"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""
import shutil
import pandas as pd

# Importing libraries
import os
from omegaconf import OmegaConf
import logging
import torch
import torch.nn as nn
import torchsummary
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from get_datasetframe import *
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

from rich.table import Column, Table
from rich import box
from rich.console import Console
from utils import *
from torch import cuda
from T5mapping import *
from torch.utils.data import DataLoader, Subset
import random
from rich import box
from rich.console import Console
from rich.table import Table
import wandb
from dataloader import DataSetClass
from trainer import  display_df, T5Trainer
import yaml

# ------------------- load config -------------------
test_mode = True # set to True if you want to test the code on your local computer
# Load the configuration from the YAML file
if test_mode:
    with open("config_test.yaml", "r") as file:
        yaml_cfg = yaml.safe_load(file)
else:
    with open("config.yaml", "r") as file:
        yaml_cfg = yaml.safe_load(file)



# Convert the dictionary to an OmegaConf object
cfg = OmegaConf.create(yaml_cfg)
max_samples = cfg["max_samples"]
train_paths = cfg["train_paths"]
test_paths = cfg["test_paths"]
load_new_mappings = cfg["load_new_mappings"]
extra_token = cfg["extra_token"]
wandb_cfg = cfg["wandb"]

wandb.init(
    entity=wandb_cfg["wandb_user"],
    project=wandb_cfg["wandb_project"],
    name=wandb_cfg.get("wandb_run_name", None),
    notes=wandb_cfg.get("wandb_notes", None),
    config=OmegaConf.to_container(cfg, resolve=True),
)

# ------------------- get Training data -------------------
# paths = ['/data/ct_schema' , '/data/gl_schema']#, '/data/or_schema', '/Users/juliankleutgens/training_data']
# paths = ['/data/gl_schema', '/Users/juliankleutgens/training_data']
paths = train_paths
dfs_train_list = []
for path in paths:
    df = load_data(path, maxsamples=max_samples)
    dfs_train_list.append(df)

# ------------------- get Testing data -------------------
# paths_test = ['/data_test/ct_schema', '/data_test/gl_schema', '/data_test/or_schema']
# paths_test = ['/data_test/ct_schema','/data_test/gl_schema']
paths_test = test_paths
dfs_test_list = []
for path in paths_test:
    df = load_data(path, maxsamples=max_samples)
    dfs_test_list.append(df)

# ------------------- get new mapping if necessary -------------------
# set to True if you want to load new mappings
# but leave it to False if you have already loaded the mappings,
# it takes time to build the mappings
if load_new_mappings:
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    dfs = dfs_train_list + dfs_test_list
    save_new_mapping_from_df(dfs, extra_token, tokenizer)

# concatenate all the data
for i, df in enumerate(dfs_train_list):
    df = pd.DataFrame(df)
    if i == 0:
        dfs_train = df
    else:
        dfs_train = pd.concat([dfs_train, df], ignore_index=True)

for i, df in enumerate(dfs_test_list):
    df = pd.DataFrame(df)
    if i == 0:
        dfs_test = df
    else:
        dfs_test = pd.concat([dfs_test, df], ignore_index=True)
    dfs_test_list[i] = df

# get log file
training_logger = Table(Column("Epoch", justify="center"),
                        Column("Steps", justify="center"),
                        Column("Loss", justify="center"),
                        title="Training Status", pad_edge=False, box=box.ASCII)
# Apply the reformatting function
# all_dsl_tokens = get_all_dsl_tokens()

# ---- init console ----
# define a rich console logger
console = Console(record=True)
# ---------------- Log the model loading ----------------
current_time = datetime.now().strftime("%Y%m%d_%H%M")  # Generate the current timestamp
base_output_dir = cfg["output_dir"]
new_output_dir = os.path.join(base_output_dir, f"output_{current_time}")
os.makedirs(new_output_dir, exist_ok=True)
cfg["output_dir"] = new_output_dir
output_dir = cfg["output_dir"]
# copy past the dsl_token_mapping.json file and the config.yaml to the output directory
shutil.copy("dsl_token_mappings_T5.json", output_dir)
# use cfg to save the config.yaml file
with open(os.path.join(output_dir, "config_used.yaml"), "w") as file:
    OmegaConf.save(cfg, file)

T5Trainer(cfg=cfg, dataframe_train=dfs_train, dataframe_test_list=dfs_test_list, console=console,
          training_logger=training_logger)
# can I save the output folder to weights and biases?
wandb.save(output_dir)

