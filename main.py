"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Julian Kleutgens
Date: June 2024
"""
import shutil
import pandas as pd
from datetime import datetime
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
from data_scripts.get_datasetframe import *
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

from rich.table import Column, Table
from rich import box
from rich.console import Console
from utils import *
from torch import cuda
from tokenization.T5mapping import *
from torch.utils.data import DataLoader, Subset
import random
from rich import box
from rich.console import Console
from rich.table import Table
import wandb
from data_scripts.dataloader import DataSetClass
from trainer import display_df, T5Trainer
import yaml

# ------------------- load config -------------------
# the config file is it configuration/config.yaml
os.makedirs("configuration", exist_ok=True)
# open directly the config.yaml file
with open("configuration/config.yaml", "r") as file:
    yaml_cfg = yaml.safe_load(file)
cfg = OmegaConf.create(yaml_cfg)

test_mode = cfg["test_mode"] # set to True if you want to test the code on your local computer
if test_mode:
    with open("configuration/config_test.yaml", "r") as file:
        yaml_cfg = yaml.safe_load(file)
    cfg = OmegaConf.create(yaml_cfg)
    print("Test mode is on")




# Convert the dictionary to an OmegaConf object


max_samples = cfg["max_samples"]
train_paths = cfg["train_paths"]
test_paths = cfg["test_paths"]
load_new_mappings = cfg["load_new_mappings"]
extra_token = cfg["extra_token"]
wandb_cfg = cfg["wandb"]
sparse_type = cfg["sparse_type"]
type_of_mapping = cfg["type_of_mapping"]

wandb.init(
    entity=wandb_cfg["wandb_user"],
    project=wandb_cfg["wandb_project"],
    name=wandb_cfg.get("wandb_run_name", None),
    notes=wandb_cfg.get("wandb_notes", None),
    config=OmegaConf.to_container(cfg, resolve=True),
)
# ---------------- Log the model loading ----------------
current_time = datetime.now().strftime("%Y%m%d_%H%M")  # Generate the current timestamp
base_output_dir = cfg["output_dir"]
new_output_dir = os.path.join(base_output_dir, f"output_{current_time}")
os.makedirs(new_output_dir, exist_ok=True)
cfg["output_dir"] = new_output_dir
output_dir = cfg["output_dir"]
# use cfg to save the config.yaml file
with open(os.path.join(output_dir, "config_used.yaml"), "w") as file:
    OmegaConf.save(cfg, file)
print(f"New output directory created at: {new_output_dir}")


# ------------------- get Training data -------------------
# paths = ['/data/ct_schema' , '/data/gl_schema']#, '/data/or_schema', '/Users/juliankleutgens/training_data']
# paths = ['/data/gl_schema', '/Users/juliankleutgens/training_data']
paths = train_paths
dfs_train_list = []
for path in paths:
    df = load_data(path, maxsamples=max_samples, sparse_type=sparse_type)
    dfs_train_list.append(df)
print("Training Data loaded successfully")

# ------------------- get Testing data -------------------
# paths_test = ['/data_test/ct_schema', '/data_test/gl_schema', '/data_test/or_schema']
# paths_test = ['/data_test/ct_schema','/data_test/gl_schema']
paths_test = test_paths
dfs_test_list = []
max_test_samples = cfg["test_samples"]
for path in paths_test:
    df = load_data(path, maxsamples=max_test_samples, sparse_type=sparse_type)
    dfs_test_list.append(df)
print("Testing Data loaded successfully")
# ------------------- get new mapping if necessary -------------------
# set to True if you want to load new mappings
# but leave it to False if you have already loaded the mappings,
# it takes time to build the mappings
already_loaded = False
if load_new_mappings and not os.path.exists(cfg["model_params"]["fined_tuned_dir"]):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    dfs = dfs_train_list + dfs_test_list
    save_new_mapping_from_df(dfs, extra_token, tokenizer , type_of_mapping)
elif os.path.exists(cfg["model_params"]["fined_tuned_dir"]):
    idx = cfg["model_params"]["fined_tuned_dir"].rfind("/")
    path_to_mapping = cfg["model_params"]["fined_tuned_dir"][:idx+1] + "dsl_token_mappings_T5.json"
    print(f" We are using a pre-trained model which was already fine tuned with the mapping file, we are copy pasting the file from: {path_to_mapping}")
    # overwrite the path to the mapping file
    shutil.copy(path_to_mapping, output_dir)
    cfg["path_to_mapping"] = os.path.join(output_dir, "dsl_token_mappings_T5.json")
    already_loaded = True

if not already_loaded:
    # copy past the dsl_token_mapping.json file and the config.yaml to the output directory
    shutil.copy("dsl_token_mappings_T5.json", output_dir)
    cfg["path_to_mapping"] = os.path.join(output_dir, "dsl_token_mappings_T5.json")
    print(f"Copied dsl_token_mappings_T5.json and config.yaml to: {output_dir}")
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


# ---- init console ----
# define a rich console logger
console = Console(record=True)



# ------------------- Start Training -------------------
T5Trainer(cfg=cfg, dataframe_train=dfs_train, dataframe_test_list=dfs_test_list, console=console)
wandb.save(output_dir)

