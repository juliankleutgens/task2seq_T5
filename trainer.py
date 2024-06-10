
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
from engine import train_and_validate, validate
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



def T5Trainer(cfg,dataframe_train,dataframe_test_list, console=Console()):
    """
    T5 trainer
    """
    torch.backends.cudnn.deterministic = True
    output_dir = cfg["output_dir"]
    model_params = cfg["model_params"]
    extra_tokens = cfg["extra_token"]
    train_on_multiple_gpus = cfg["train_on_multiple_gpus"]

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

    # Define the model and send it to the appropriate device
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])

    # ------------------------------ load device ------------------------------
    # Set the environment variable to use specified GPUs if device is set to cuda
    # Set the environment variable to use only the n-th GPU
    # n = "0" => use the first GPU
    # n = 1 => use the second GPU
    if cfg["device"] == "cuda" and "n_gpu" in cfg:
        device = torch.device(f"cuda:{str(cfg['n_gpu'])}")
    else:
        device = torch.device("cuda")


    if cfg["device"] == "cuda" and torch.cuda.is_available():
        model = model.to(device)
        # Check if we need to train on multiple GPUs
        if train_on_multiple_gpus and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
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
    train_dataset = dataframe_train.reset_index(drop=True)
    # Logging
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = DataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], extra_tokens, cfg=cfg)
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
        val_dataset = dataframe_test.reset_index(drop=True)
        console.print(f"TEST Dataset: {val_dataset.shape}\n")
        val_set = DataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                   model_params["MAX_TARGET_TEXT_LENGTH"], extra_tokens,cfg=cfg)


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
    first_epoch = True
    # ------------------- Training and Validation Loop -------------------
    for epoch in range(1,model_params["TRAIN_EPOCHS"]+1):
        metrics_blue, metrics_leven, accuracy = train_and_validate(epoch, tokenizer=tokenizer, model=model, device=device,
                                        loader=training_loader, optimizer=optimizer,
                                        console=console, cfg=cfg, val_loader_list=val_loader_list)

        if epoch % cfg["model_params"]["VAL_EPOCHS"] == 0:

            # --------- save the data to a plot file ---------
            # Convert metrics to DataFrame
            df_metrics_blue = pd.DataFrame([metrics_blue])
            df_metrics_leven = pd.DataFrame([metrics_leven])
            df_accuracy = pd.DataFrame([accuracy])

            if first_epoch:
                all_metrics_blue = df_metrics_blue
                all_metrics_leven = df_metrics_leven
                all_accuracy = df_accuracy
                first_epoch = False
            else:
                all_metrics_blue = pd.concat([all_metrics_blue, df_metrics_blue], ignore_index=True)
                all_metrics_leven = pd.concat([all_metrics_leven, df_metrics_leven], ignore_index=True)
                all_accuracy = pd.concat([all_accuracy, df_accuracy], ignore_index=True)




    console.save_text(os.path.join(output_dir, 'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")



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
    plot_metrics(all_accuracy, 'Accuracy', output_dir)

