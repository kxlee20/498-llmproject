import os
from copy import deepcopy
import torch
import random
import transformers
from datasets import load_dataset  #HF
from collections import defaultdict

from config_args import *
from reft_dataset import *

task_keys = ("sentence_1", "sentence_2")


def parse_positions(positions: str):

    first_n, last_n = 0, 0
    first_n = int(positions.split("+")[0].strip("f"))
    last_n = int(positions.split("+")[1].strip("l"))
    return first_n, last_n


class LoreftDataset(ReftDataset):

    def preprocess(self, kwargs):
        print(kwargs)
        # basic setup
        self.raw_dataset, self.trigger_tokens, self.num_labels = None, None, None
        dataset_config = config_args[self.task]
        self.task_prompt_template = dataset_config["task_prompt_template"]
        self.trigger_tokens = dataset_config["trigger_tokens"]
        self.original_data_split = self.data_split
        self.test_split = kwargs[
            "test_split"] if "test_split" in kwargs else None

        self.data_path = os.path.join(self.data_path,
                                      self.data_split + ".json")

    def postprocess(self, kwargs):
        original_dataset_size = len(self.task_dataset)
        self.raw_dataset = self.task_dataset
        return

    def tokenize(self, data_item):
        result = {}
        base_prompt = self.task_prompt_template % (
                data_item['instruction'])
        base_input = base_prompt + self.trigger_tokens + data_item[
            "answer"] + self.tokenizer.eos_token

        # tokenize
        base_prompt_ids = self.tokenizer(
            base_prompt,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        # print(f"base_prompt_length : {base_prompt_length }")
        if self.original_data_split == "train":
            base_input_ids = self.tokenizer(
                base_input,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")["input_ids"][0]

            output_ids = deepcopy(base_input_ids)
            output_ids[:base_prompt_length] = IGNORE_INDEX

            result["input_ids"] = base_input_ids
            result["labels"] = output_ids
        else:
            # print("Assuming test split for now")
            result["input_ids"] = base_prompt_ids
        last_position = base_prompt_length

        return result, last_position
