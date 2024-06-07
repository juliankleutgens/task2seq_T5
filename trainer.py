
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
from torch.utils.data import DataLoader, Subset, DistributedSampler
import random
from rich import box
from rich.console import Console
from rich.table import Table
import wandb
from dataloader import DataSetClass
from engine import train, validate
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP


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
    torch.backends.cudnn.deterministic = True
    output_dir = cfg["output_dir"]
    model_params = cfg["model_params"]
    extra_tokens = cfg["extra_token"]

    # Print CUDA availability information
    try:
        print(f"The line torch.cuda.is_available() is {torch.cuda.is_available()}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error checking CUDA availability: {e}")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    source_text="input"
    target_text="target"

    # Define the model and send it to the appropriate device
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])

    # ------------------------------ load device ------------------------------
    # Set the environment variable to use specified GPUs if device is set to cuda
    # Set the environment variable to use only the n-th GPU
    # n = "0" => use the first GPU
    # n = "1" => use the second GPU
    # n = "5,6" => use the 5th and 6th GPU
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', init_method='env://')
    print(f"Rank: {dist.get_rank()}")
    device = torch.device(f'cuda:{dist.get_rank()}')

    if cfg["device"] == "cuda" and "n_gpu" in cfg:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["n_gpu"]

    if cfg["device"] == "cuda" and torch.cuda.is_available():
        #device = torch.device("cuda")
        model = model.to(device)
        # Check if we need to train on multiple GPUs
        if cfg.get("train_on_multiple_gpus", False) and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = DDP(model, device_ids=[dist.get_rank()])
        else:
            print("Using a single GPU")
    elif cfg["device"] == "mps":
        device = torch.device("mps")
        print("Using Multi-Process Service (MPS) for model parallelism")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("Using CPU for model parallelism")
        model = model.to(device)

    print(f"------- Using device {device} -------")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
    else:
        print(f"Using CPU, CUDA is not available")

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")
    wandb.log({"model_loading": model_params["MODEL"]})
    print(f"Loading {model_params['MODEL']}...")



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
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text, extra_tokens)
    # Defining the parameters for creation of dataloaders
    if cfg["train_on_multiple_gpus"]:
        train_sampler = DistributedSampler(training_set)
        train_params = {
            'batch_size': model_params["TRAIN_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': 0,
            'sampler': train_sampler
        }
    else:
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
                                   model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text, extra_tokens)

        if cfg["train_on_multiple_gpus"]:
            validation_sampler = DistributedSampler(val_set)
            val_params = {
                'batch_size': model_params["VALID_BATCH_SIZE"],
                'shuffle': False,
                'num_workers': 0,
                'sampler': validation_sampler
            }
        else:
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
        metrics_blue, metrics_leven = train(epoch, tokenizer=tokenizer, model=model, device=device, loader=training_loader, optimizer=optimizer,
            console=console, cfg=cfg, val_loader_list=val_loader_list)

        # --------- save the data to a plot file ---------
        # Convert metrics to DataFrame
        df_metrics_blue = pd.DataFrame([metrics_blue])
        df_metrics_leven = pd.DataFrame([metrics_leven])

        if epoch == 0:
            all_metrics_blue = df_metrics_blue
            all_metrics_leven = df_metrics_leven
        else:
            all_metrics_blue = pd.concat([all_metrics_blue, df_metrics_blue], ignore_index=True)
            all_metrics_leven = pd.concat([all_metrics_leven, df_metrics_leven], ignore_index=True)

    # Plot the metrics for the training
    def plot_metrics(all_metrics, metric_name, output_dir):
        """Plot the metrics"""
        plt.figure()
        for column in all_metrics.columns:
            if column == 'epoch':
                continue
            plt.plot(all_metrics[column], label=column)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over Epochs')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric_name}_metrics.png'))
        plt.show()

    console.log(f"[Plotting Metrics]...\n")
    plot_metrics(all_metrics_blue, 'BLEU Score', output_dir)
    plot_metrics(all_metrics_leven, 'Levenshtein Distance', output_dir)

    console.save_text(os.path.join(output_dir, 'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")
    dist.destroy_process_group()

