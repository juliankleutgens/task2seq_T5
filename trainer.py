
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
import shutil
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from get_datasetframe import *
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

from datetime import datetime
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
from engine import train, validate


def display_df(df):
  """display dataframe in ASCII format"""

  console=Console()
  table = Table(Column("source_text", justify="center" ), Column("target_text", justify="center"), title="Sample Data",pad_edge=False, box=box.ASCII)

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  console.print(table)

def create_loader(dataset, train_params, max_samples=10000):
    # Randomly sample indices without replacement
    indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    subset = Subset(dataset, indices)
    return DataLoader(subset, **train_params)



def T5Trainer(cfg,dataframe_train,dataframe_test_list, console=Console(), training_logger=Table(title="Training Logs", box=box.ASCII)):
    """
    T5 trainer
    """
    # Set random seeds and deterministic pytorch for reproducibility
    # torch.manual_seed(model_params["SEED"]) # pytorch random seed
    # np.random.seed(model_params["SEED"]) # numpy random seed
    torch.backends.cudnn.deterministic = True
    # Setup logging
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)



    # ------------------- load device -------------------
    model_params = cfg["model_params"]
    if cfg["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif cfg["device"] == "cpu":
        device = torch.device("cpu")
    elif cfg["device"] == "mps":
        device = torch.device("mps")
    else:
        raise ValueError("Invalid device. Choose from 'cuda', 'cpu', 'mps'")
    print(f"------- Using device {device} -------")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")
    wandb.log({"model_loading": model_params["MODEL"]})
    print(f"Loading {model_params['MODEL']}...")


    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    source_text="input"
    target_text="target"

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)
    # logging
    console.log(f"[Data]: Reading data...\n")
    wandb.log({"data_reading": True})

    # ----------------- Testing dataloader -----------------
    dataframe_train = dataframe_train[[source_text, target_text, 'name']]
    # Assuming dataframe_train and dataframe_test are already defined
    train_dataset = dataframe_train.reset_index(drop=True)
    # Logging
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = DataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
    }
    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)


    # ----------------- Validation dataloader -----------------
    val_loader_list = []
    for i, dataframe_test in enumerate(dataframe_test_list):
        dataframe_test = dataframe_test[[source_text, target_text, 'name']]
        val_dataset = dataframe_test.reset_index(drop=True)
        console.print(f"TEST Dataset: {val_dataset.shape}\n")
        val_set = DataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                   model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
        val_params = {
            'batch_size': model_params["VALID_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': 0
        }
        val_loader = DataLoader(val_set, **val_params)
        val_loader_list.append(val_loader)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_params["LEARNING_RATE"])

    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')

    # ------------------- Training and Validation Loop -------------------
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer=tokenizer, model=model, device=device, loader=training_loader, optimizer=optimizer,
            console=console, cfg=cfg, val_loader_list=val_loader_list)


    console.save_text(os.path.join(output_dir, 'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")

