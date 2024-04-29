"""
ReFT trainer functions pulled from
https://github.com/stanfordnlp/pyreft/blob/main/pyreft/reft_model.py.
"""

import pyvene as pv
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (Trainer, TrainingArguments, DataCollator,
                          DataCollatorForSeq2Seq, AutoTokenizer)
from transformers.trainer_utils import (EvalPrediction, has_length,
                                        denumpify_detensorize)
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from tqdm import tqdm
import os
import torch
import re
import evaluate
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class ReftDataCollator(object):
    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs[
            "intervention_locations"][..., :max_seq_length]
        return batch_inputs


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(dataset: Dataset, batch_size: int,
                    collate_fn: DataCollatorForSeq2Seq,
                    shuffle: bool) -> DataLoader:
    return DataLoader(dataset,
                      shuffle=shuffle,
                      batch_size=batch_size,
                      collate_fn=collate_fn)


class ReftTrainer(Trainer):

    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_intervention(
            save_directory=f"{output_dir}/intervenable_model",
            include_model=True)

    def _load_best_model(self):
        logger.warning(
            f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model",
            include_model=True)

    def compute_loss(self,
                     intervenable: pv.IntervenableModel,
                     inputs,
                     return_outputs=False):
        # run intervened forward pass
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations={
                "sources->base":
                (None, inputs["intervention_locations"].permute(1, 0,
                                                                2).tolist())
            },
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist()
            if "subspaces" in inputs else None)
        # return
        return (cf_outputs.loss,
                cf_outputs) if return_outputs else cf_outputs.loss


class ReftTrainerForCausalLM(ReftTrainer):

    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset,
                               self._train_batch_size,
                               self.data_collator,
                               shuffle=True)


def count_parameters(model):
    """Count parameters of a model that require gradients"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ReftModel(pv.IntervenableModel):
    """
    Base model for Reft methods.
    """

    def __init__(self, config, model, **kwargs):
        super().__init__(config, model, **kwargs)

    @staticmethod
    def _convert_to_reft_model(intervenable_model):
        reft_model = ReftModel(intervenable_model.config,
                               intervenable_model.model)
        # Copy any other necessary attributes
        for attr in vars(intervenable_model):
            setattr(reft_model, attr, getattr(intervenable_model, attr))
        return reft_model

    @staticmethod
    def load(*args, **kwargs):
        model = pv.IntervenableModel.load(*args, **kwargs)
        return ReftModel._convert_to_reft_model(model)

    def print_trainable_parameters(self):
        """
        Print trainable parameters.
        """
        _linked_key_set = set([])
        trainable_intervention_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v[0], pv.TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[
                            k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        trainable_intervention_parameters += count_parameters(
                            v[0])
                else:
                    trainable_intervention_parameters += count_parameters(v[0])

        trainable_model_parameters = sum(p.numel()
                                         for p in self.model.parameters()
                                         if p.requires_grad)

        all_model_parameters = sum(p.numel() for p in self.model.parameters())

        total_trainable_parameters = trainable_intervention_parameters + trainable_model_parameters

        print(
            f"trainable intervention params: {trainable_intervention_parameters:,d} || trainable model params: {trainable_model_parameters:,d}\n"
            f"model params: {all_model_parameters:,d} || trainable%: {100 * total_trainable_parameters / all_model_parameters}"
        )