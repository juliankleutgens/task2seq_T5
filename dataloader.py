
import pandas as pd

# Importing libraries
import os
from omegaconf import OmegaConf
import logging
import torch
import torchsummary
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from T5mapping import *



class DataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the neural network for finetuning the model.
    """
    def __init__(self, dataframe, tokenizer, source_len, target_len, extra_tokens):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.target_text = self.data['target']
        self.source_text = self.data['input']
        self.name = self.data['name']
        self.local_path = self.data['local_path']
        self.extra_token = extra_tokens

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        name = self.name[index]
        path = self.local_path[index]
        source_text = str(self.source_text[index])
        target_string = self.target_text[index]

        # ------- Tokenize source text  ------
        #source_text_tokens = self.tokenizer.tokenize(source_text)
        #print(len(source_text_tokens))
        source_encoded = self.tokenizer.encode_plus(
            source_text,
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        # convert back
        #source_decode = self.tokenizer.convert_ids_to_tokens(source_encoded['input_ids'].squeeze())
        #print("tokenized_task_sample_length",len(self.tokenizer.tokenize(source_text)))
        #print("encoded_task_sample_length", source_encoded['input_ids'].shape[1])

        # -------------- tokenizes target ------------
        # so convert it to embedding and tensor
        target_token,_ = map_to_t5_token(target_string,extra_token=self.extra_token, tokenizer=self.tokenizer, loading_new_mappings=False)
        target_token_ids = self.tokenizer.convert_tokens_to_ids(target_token)

        # print(f"The length of the tokenized solver.py functions sample is: {len(target_token_ids)}")
        # Prepare target tokens tensor
        target_ids = torch.tensor(target_token_ids, dtype=torch.long)

        # Padding target to ensure fixed size tensor
        if len(target_ids) < self.target_len:
            padding_length = self.target_len - len(target_ids)
            target_ids = torch.cat([target_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
        if len(target_ids) > self.target_len:
            l = len(target_ids)
            target_ids = target_ids[:self.target_len]
            print(f"Target Length Exceeded for {self.name[index]} with length {l} is longer than {self.target_len}, gets truncated.")
        # Create attention masks for target
        target_mask = torch.where(target_ids == self.tokenizer.pad_token_id, 0, 1)

        return {
            'source_ids': source_encoded['input_ids'].squeeze(),
            'source_mask': source_encoded['attention_mask'].squeeze(),
            'target_ids': target_ids,
            'target_mask': target_mask,
            'name': name,
            'local_path': path,
        }
