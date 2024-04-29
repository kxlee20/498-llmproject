import pyvene as pv
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (Trainer, TrainingArguments, DataCollator,
                          DataCollatorForSeq2Seq, AutoTokenizer)
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
from config_args import *
from tqdm import tqdm
import os
import torch
import re
import evaluate
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
from transformers.trainer_utils import (EvalPrediction, has_length,
                                        denumpify_detensorize)
from model import ReftDataCollator

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.get_logger(__name__)


def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def extract_output(pred, trigger=''):
    if not trigger:
        return pred
    start = pred.find(trigger)
    if start < 0:
        return '' #generation too long
    output = pred[start + len(trigger):].lstrip()
    return output


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


def eval_metrics(task: str,
                    dataset_name: str,
                    intervenable: pv.IntervenableModel,
                    tokenizer: AutoTokenizer,
                    eval_dataset: Dataset,
                    data_items: list,
                    trigger_tokens: str,
                    run_name: str,
                    batch_size: int = 4,
                    data_collator=None,
                    split=None,
                    greedy_decoding=False,
                    temperature=None,
                    top_p=None,
                    top_k=None):
    tokenizer.padding_side = "left"  # switch padding side for collator
    num_beams = 4

    data_collator = data_collator if data_collator is not None else \
        make_data_collator(tokenizer, intervenable.model)
    eval_dataloader = make_dataloader(eval_dataset,
                                      batch_size,
                                      data_collator,
                                      shuffle=False)
    correct_count = 0
    total_count = 0
    generations = []
    eval_iterator = tqdm(eval_dataloader, position=0, leave=True)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for step, inputs in enumerate(eval_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # [layers, batch_size, positions]
            intervention_locations = inputs["intervention_locations"].permute(
                1, 0, 2)
            left_padding = (
                inputs["input_ids"] == tokenizer.bos_token_id).nonzero(
                    as_tuple=True)[1]
            left_padding = left_padding.reshape(1, -1, 1).to(
                device)  # [1, batch_size, 1]
            intervention_locations += left_padding
            intervention_locations -= 1

            intervention_locations = intervention_locations.repeat_interleave(
                num_beams, dim=1).tolist()

            generation_args = {
                "base": {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                },
                "unit_locations": {
                    "sources->base": (None, intervention_locations)
                },
                "intervene_on_prompt": True,
                "eos_token_id": tokenizer.eos_token_id,
                "early_stopping": True,
            }
            if "generation_args" in config_args[task]:
                generation_args.update(
                    config_args[task]["generation_args"][greedy_decoding])

            # generate answers
            _, steered_response = intervenable.generate(**generation_args)
            actual_preds = tokenizer.batch_decode(steered_response,
                                                    skip_special_tokens=True)

            for id, pred in zip(inputs["id"].tolist(), actual_preds):
                example = data_items[id]
                try:
                    raw_generation = extract_output(pred, trigger_tokens)
                except:
                    print("get not split based on trigger tokens: ",
                            raw_generation)
                    raw_generation = "WRONG"

                # check if generation is correct
                answer = example["answer"]
                generation = raw_generation[:]
                    # #TODO:
                    # print("generation:", generation)
                if generation.strip() == answer.strip():
                    correct_count += 1

                total_count += 1
                metric_str = round(correct_count / total_count, 3)
                eval_iterator.set_postfix({"em": metric_str})
                instruction = example[
                    "question"] if task == "gsm8k" else example[
                        "instruction"]
                generations += [{
                    "instruction": instruction,
                    "raw_generation": raw_generation,
                    "generation": generation,
                    "answer": answer
                }]

    return generations, {
            f"eval/{dataset_name}": correct_count / total_count
        }
