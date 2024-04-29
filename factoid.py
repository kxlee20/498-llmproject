import copy, json, random, re
import logging
from dataclasses import dataclass, field
import torch
import transformers

from typing import Dict, Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_minimal
from matplotlib.ticker import MaxNLocator
from datasets import Dataset
from transformers import Trainer

from config_args import *
from dataset import LoreftDataset
from compute_metrics import compute_metrics
from loreft import *
from model import *
from reft_dataset import *
from utils import *

IGNORE_INDEX = -100

device = "cuda" if torch.cuda.is_available() else "cpu"

make_supervised_data_module = make_last_position_supervised_data_module

# load model (take 1 min)
base_model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map=device)

# get tokenizer
model_max_length = 2048
tokenizer = transformers.AutoTokenizer.from_pretrained(
    base_model,
    model_max_length=model_max_length,
    padding_side="right",
    use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

dataset = load_dataset("aisafe/FACTOID")

# Split the dataset into training and validation + test
train_testvalid = dataset["train"].train_test_split(
    test_size=0.2)  # 80% train, 20% for test+validation

# Further split the test+validation into separate test and validation sets
test_valid_split = train_testvalid['test'].train_test_split(
    test_size=0.5)  # Split the 20% into 10% test and 10% validation

# Assign splits to variables
train_set = train_testvalid['train']
valid_set = test_valid_split['test']
test_set = test_valid_split['train']
print(valid_set[:5])
positive_examples = []
negative_examples = []
neutral_examples = []
for example in train_set:
    if example["label"] == 'neutral':
        neutral_examples += [
            f"Sentence Pair: {example['sentence_1']};{example['sentence_2']}"
        ]
    elif example["label"] == 'support':
        positive_examples += [
            f"Sentence Pair: {example['sentence_1']};{example['sentence_2']}"
        ]
    elif example["label"] == 'refute':
        negative_examples += [
            f"Sentence Pair: {example['sentence_1']};{example['sentence_2']}"
        ]
targ_layer = 15

import wandb

wandb.login()
wandb.init(project='factoid-reft-3')

results = []
for seed in [42]:
    random.seed(seed)
    # K_SHOTS = [4, 10, 20, 30, 40, 50]
    K_SHOTS = [3, 6, 18, 27]
    for K in K_SHOTS:
        print("evaluating: ", K)
        wandb.init(project='factoid-reft', config={"K": K, "seed": seed})
        # creating training dataset for ReFT
        half_k = K // 3
        pos_demo = random.sample(positive_examples, k=half_k)
        neg_demo = random.sample(negative_examples, k=half_k)
        neu_demo = random.sample(neutral_examples,
                                 k=half_k)  # Samples from the neutral class

        # Combine all samples and create storage access ids
        storage_access_ids = [f"{w}->" for w in pos_demo + neg_demo + neu_demo]

        # Assuming you also need to adjust memo_tokens or any similar structure:
        memo_tokens = ["positive"] * half_k + ["negative"] * half_k + [
            "neutral"
        ] * half_k

        # create ICL baseline
        icl_prompt = []
        for w in pos_demo:
            icl_prompt += [f"{w}->positive"]
        for w in neg_demo:
            icl_prompt += [f"{w}->negative"]
        for w in neu_demo:
            icl_prompt += [f"{w}->neutral"]
        random.shuffle(icl_prompt)
        icl_prompt = "\n".join(icl_prompt)

        # get reft model
        reft_config = ReftConfig(
            representations={
                "layer":
                targ_layer,
                "component":
                "block_output",
                "intervention":
                LoreftIntervention(embed_dim=model.config.hidden_size,
                                   low_rank_dimension=1)
            })
        reft_model = get_reft_model(model, reft_config)
        reft_model.print_trainable_parameters()

        # get training data to train our intervention to remember the following sequence
        data_module = make_last_position_supervised_data_module(
            tokenizer, model, storage_access_ids, memo_tokens)

        # train
        training_args = transformers.TrainingArguments(
            max_steps=200,
            output_dir="./tmp",
            learning_rate=2e-3,
            report_to='wandb',
            per_device_train_batch_size=8,
            logging_steps=50,
            save_strategy="no",
            evaluation_strategy="no")
        trainer = ReftTrainerForCausalLM(model=reft_model,
                                         tokenizer=tokenizer,
                                         args=training_args,
                                         **data_module)
        _ = trainer.train()

        # evaluate ReFT
        print(f"Evaluating ReFT for K={K}")
        correct_count = 0
        for e in valid_set:
            w = f"Sentence Pair: {e['sentence_1']};{e['sentence_2']}"
            prompt = tokenizer(f"{w}->", return_tensors="pt").to(device)
            base_unit_location = prompt["input_ids"].shape[-1] - 1
            _, steered_response = reft_model.generate(
                prompt,
                unit_locations={
                    "sources->base": (None, [[[base_unit_location]]])
                },
                intervene_on_prompt=True,
                max_new_tokens=10,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True)
            retrieved_storage = tokenizer.decode(steered_response[0],
                                                 skip_special_tokens=True)
            retrieved_storage = retrieved_storage.split("->")[-1]
            if e["label"] == "support" and retrieved_storage == "positive":
                correct_count += 1
            elif e["label"] == 'refute' and retrieved_storage == "negative":
                correct_count += 1
            elif e["label"] == 'neutral' and retrieved_storage == "neutral":
                correct_count += 1
        reft_acc = round((correct_count) / (len(valid_set)), 2)

        # evaluate ICL baseline
        print(f"Evaluating ICL for K={K}")
        correct_count = 0
        for e in valid_set:
            w = f"Sentence Pair: {e['sentence_1']};{e['sentence_2']}"
            prompt = tokenizer(f"{icl_prompt}\n{w}->",
                               return_tensors="pt").to(device)
            steered_response = model.generate(
                **prompt,
                max_new_tokens=10,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True)
            retrieved_storage = tokenizer.decode(steered_response[0],
                                                 skip_special_tokens=True)
            retrieved_storage = retrieved_storage.split("->")[-1].split(
                "\n")[0]
            if e["label"] == "support" and retrieved_storage == "positive":
                correct_count += 1
            elif e["label"] == 'refute' and retrieved_storage == "negative":
                correct_count += 1
            elif e["label"] == 'neutral' and retrieved_storage == "neutral":
                correct_count += 1

        icl_acc = round((correct_count) / (len(valid_set)), 2)
        print((K, reft_acc, icl_acc))
        wandb.log({"Validation Accuracy REFT": reft_acc})
        wandb.log({"Validation Accuracy ICL": icl_acc})
        results += [(K, reft_acc, icl_acc)]
        wandb.finish()

result_json_file_name = './cmp.json'
with open(result_json_file_name, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved {result_json_file_name} ")