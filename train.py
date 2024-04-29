import os
import torch
import argparse
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
    TrainingArguments
)
from transformers.trainer_utils import EvalPrediction
import wandb
import evaluate
import datetime
import json
import math
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import Dataset

from config_args import *
from dataset import LoreftDataset
from eval_metrics import eval_metrics
from loreft import *
from model import *
from reft_dataset import *
from utils import *

CUDA_LAUNCH_BLOCKING=1

device = "cuda" if torch.cuda.is_available() else "cpu"

class LoReftConfig(pv.IntervenableConfig):
    """
    Reft config for Reft methods.
    """
    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

def fetch_eval_results(data_sets, task, out_dir, model, tokenizer,
                       trigger, run_id, batch_size, collator,
                       n_params, greedy, temp, p_top, k_top,
                       logging):

    results = {}
    for ds_name in data_sets:
        for split, (eval_set, data_items) in data_sets[ds_name].items():

            gens, stats = eval_metrics(
                task, ds_name, model, tokenizer, eval_set,
                data_items, trigger, run_id, batch_size,
                collator, split, greedy, temp, p_top, k_top
            )
            results.update(stats)
            data_to_save = stats if gens is None else gens
            file_path = f"{out_dir}/{run_id}/{ds_name}_{split}_outputs.json"
            with open(file_path, 'w') as file:
                json.dump(data_to_save, file, indent=4)

    final_path = f"{out_dir}/{run_id}/eval_results.json"
    results["n_params"] = n_params
    with open(final_path, 'w') as file:
        json.dump(results, file, indent=4)

    return results


def load_datasplits(task, data_dir, tokenizer, seed, layers, position,
              max_n_train_example, max_n_eval_example, share_weights,
              test_split):
    train_datasets = config_args[task]["train_datasets"]
    eval_datasets = config_args[task]["eval_datasets"]

    train_datapath = os.path.join(data_dir, train_datasets[0])
    train_dataset = LoreftDataset(
        task, train_datapath,
        tokenizer, data_split="train", seed=seed, max_n_example=max_n_train_example,
        **{"num_interventions": len(layers), "position": position,
           "share_weights": share_weights, "test_split": test_split}
    )

    all_eval_datasets = {}

    for eval_dataset in eval_datasets:
        test_splits = test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            eval_datapath = os.path.join(data_dir, eval_dataset)
            raw_eval = LoreftDataset(
                task, eval_datapath,
                tokenizer, data_split=split, seed=seed, max_n_example=max_n_eval_example,
                **{"num_interventions": len(layers), "position": position,
                   "share_weights": share_weights}
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets
    return train_dataset, eval_datasets

def loreft(
):
    # ---- GET CONFIGURATIONS ----
    task = config_args['task']
    data_dir = config_args['data_dir']
    train_dataset = config_args['train_dataset']
    eval_dataset = config_args['eval_dataset']
    model = config_args['model']
    seed = config_args['seed']
    layers_config = config_args['layers']
    rank = config_args['rank']
    position = config_args['position']
    epochs = config_args['epochs']
    is_wandb = config_args['is_wandb']
    wandb_name = config_args['wandb_name']
    save_model = config_args['save_model']
    max_n_train_example = config_args['max_n_train_example']
    max_n_eval_example = config_args['max_n_eval_example']
    intervention_type = config_args['type']
    gradient_accumulation_steps = config_args['gradient_accumulation_steps']
    batch_size = config_args['batch_size']
    eval_batch_size = config_args['eval_batch_size']
    output_dir = config_args['output_dir']
    lr = config_args['lr']
    schedule = config_args['schedule']
    warmup_ratio = config_args['wu']
    weight_decay = config_args['wd']
    dropout = config_args['dropout']
    act_fn = config_args['act_fn']
    add_bias = config_args['add_bias']
    test_split = config_args['test_split']
    train_on_inputs = config_args['train_on_inputs']
    max_length = config_args['max_length']
    use_normalized_template = config_args['use_normalized_template']
    allow_cls_grad = config_args['allow_cls_grad']
    metric_for_best_model = config_args['metric_for_best_model']
    dtype = config_args['dtype']
    logging_steps = config_args['logging_steps']
    wandb_dir = config_args['wandb_dir']
    wandb_proj = config_args['wandb_proj']
    share_weights = config_args['share_weights']
    greedy_decoding = config_args['greedy_decoding']
    temperature = config_args['temperature']
    top_p = config_args['top_p']
    top_k = config_args['top_k']

    dtype = torch.bfloat16
    set_seed(seed)

    print_logistics(task, model, layers_config, rank, position, epochs,
                           max_length, allow_cls_grad)

    model_name = model
    model_str = model.split("/")[-1]
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    run_name = f"{model_str}.{task}.{now}"

    layers = [int(l) for l in layers_config.split(";")]
    if "+" in position and not share_weights:
        layers += layers

    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    # load dataset splits
    train_dataset, eval_datasets = load_datasplits(task, data_dir, tokenizer, seed, layers, position,
              max_n_train_example, max_n_eval_example, share_weights,
              test_split)

    trigger_tokens = train_dataset.trigger_tokens
    num_labels = train_dataset.num_labels

    model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype if dtype != "float8" else None,  # save memory
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )
    config = model.config
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    intervention_type = LoreftIntervention

    data_collator = ReftDataCollator(data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest"
        ))

    intervention_dtype = torch.bfloat16 if isinstance(dtype, str) else dtype

    representations = [{
            "layer": l, "component": "block_output",
            "low_rank_dimension": rank,
            "intervention": intervention_type(
                embed_dim=config.hidden_size, low_rank_dimension=rank,
                dropout=dropout, dtype=intervention_dtype, act_fn=act_fn, device=device,
                add_bias=add_bias
            )
        } for l in layers]

    reft_config = LoReftConfig(representations=representations)

    reft_model = ReftModel(reft_config, model)
    if not isinstance(dtype, str):
        reft_model.set_device(model.device)
    reft_model.disable_model_gradients()

    reft_model.print_trainable_parameters()
    reft_model.model.train()
    n_params = reft_model.count_parameters(include_model=False)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="no",
        save_strategy="no",
        metric_for_best_model=None,
        load_best_model_at_end=False,
        logging_strategy="steps",
        save_total_limit=1,
        logging_steps=logging_steps,
        lr_scheduler_type=schedule,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        weight_decay=weight_decay,
        report_to=[],
        use_cpu=False if device == "cuda" else True,
        seed=seed,
        remove_unused_columns=False
    )
    trainer = ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )
    trainer.train()

    # save config so you don't forget if this thing finally works
    args_dict = config_args
    args_dict["n_params"] = n_params
    json_file_name = f"{output_dir}/{run_name}/args.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    # ----- EVAL ------
    reft_model.model.eval()
    for k,v in reft_model.interventions.items():
        _ = v[0].eval()

    print({"n_params": n_params})
    # do eval
    eval_results = fetch_eval_results(eval_datasets, task, output_dir, reft_model, tokenizer,
                     trigger_tokens, run_name, eval_batch_size, data_collator,
                     n_params, greedy_decoding, temperature, top_p, top_k,
                     is_wandb)

    result_json_file_name = f"{output_dir}/{run_name}/eval_results.json"
    eval_results["n_params"] = n_params
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)

    print(f"Results saved {output_dir}/{run_name}")

def main():
    loreft()

if __name__ == "__main__":
    main()