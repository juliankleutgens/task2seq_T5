"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""

import pandas as pd

# Importing libraries
import os
import logging
import torch
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


def display_df(df):
  """display dataframe in ASCII format"""

  console=Console()
  table = Table(Column("source_text", justify="center" ), Column("target_text", justify="center"), title="Sample Data",pad_edge=False, box=box.ASCII)

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  console.print(table)

class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the neural network for finetuning the model.
    """
    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.target_text = self.data[target]
        self.source_text = self.data[source_text]
        self.name = self.data['name']

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_string = self.target_text[index]

        # ------- Tokenize source text  ------
        source_text_tokens = self.tokenizer.tokenize(source_text)
        print(len(source_text_tokens))
        source_encoded = self.tokenizer.encode_plus(
            source_text,
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        )

        # -------------- tokenizes target ------------
        # so convert it to embedding and tensor
        extra_token = ['sym_aft_func', 'BoF', 'EoF', 'var_to_num']
        target_token,_ = map_to_t5_token(target_string,extra_token = extra_token, tokenizer=self.tokenizer, loading_new_mappings = False)
        target_token_ids = self.tokenizer.convert_tokens_to_ids(target_token)
        # Prepare target tokens tensor
        target_ids = torch.tensor(target_token_ids, dtype=torch.long)

        # Padding target to ensure fixed size tensor
        if len(target_ids) < self.target_len:
            padding_length = self.target_len - len(target_ids)
            target_ids = torch.cat([target_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])

        # Create attention masks for target
        target_mask = torch.where(target_ids == self.tokenizer.pad_token_id, 0, 1)

        return {
            'source_ids': source_encoded['input_ids'].squeeze(),
            'source_mask': source_encoded['attention_mask'].squeeze(),
            'target_ids': target_ids,
            'target_mask': target_mask
        }

def train(epoch, tokenizer, model, device, loader, optimizer):

  """
  Function to be called for training with the parameters passed from main function

  """

  model.train()
  for _,data in enumerate(loader, 0):
    y = data['target_ids'].to(device, dtype = torch.long)
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
    ids = data['source_ids'].to(device, dtype = torch.long)
    mask = data['source_mask'].to(device, dtype = torch.long)

    outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
    loss = outputs[0]

    if _%10==0:
      training_logger.add_row(str(epoch), str(_), str(loss))
      console.print(training_logger)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def validate(epoch, tokenizer, model, device, loader):
    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = []
            for gen_id in generated_ids:
                one_sample_pred = []
                for _id in gen_id:
                    token = tokenizer.decode(_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    if token != 'pad':
                        one_sample_pred.append(token)
                preds.append(one_sample_pred)

            # --- convert the targets into a list ----- #
            target = []
            for t in y:
                one_sample_target = []
                for _id in t:
                    token = tokenizer.decode(_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    if token != 'pad':
                        one_sample_target.append(token)
                our_token_sample = map_back(one_sample_target)
                target.append(one_sample_target)

                #target.append(tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            if _ % 10 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def create_loader(dataset, train_params, max_samples=10000):
    # Randomly sample indices without replacement
    indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    subset = Subset(dataset, indices)
    return DataLoader(subset, **train_params)

def T5Trainer(dataframe_train,dataframe_test, source_text, target_text, model_params, output_dir="./outputs/" ):
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
    # Log the model loading

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)
    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe_train = dataframe_train[[source_text, target_text, 'name']]
    dataframe_test = dataframe_test[[source_text, target_text, 'name']]
#    display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    # Assuming dataframe_train and dataframe_test are already defined
    train_dataset = dataframe_train.reset_index(drop=True)
    val_dataset = dataframe_test.reset_index(drop=True)

    # Logging
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                               model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': model_params["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_params["LEARNING_RATE"])

    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch=epoch, tokenizer=tokenizer, model=model, device=device, loader=val_loader)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        #print(final_df)
        final_df.to_csv(os.path.join(output_dir, 'predictions.csv'))

    console.save_text(os.path.join(output_dir, 'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")


if __name__ == "__main__":
    """
    Example on how to get the data
    df = pd.read_csv("https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv")

    df.sample(10)

    df["text"] = "summarize: " + df["text"]

    df.head()
    """
    # ------------------- get Training data -------------------
    paths = ['/data/ct_schema', '/data/ct_schema_all', '/data/gl_schema', '/data/or_schema']
    paths = ['/data/ct_schema','/data/or_schema']
    dfs_train_list = []
    for path in paths:
        df = load_data(path, maxsamples=None)
        dfs_train_list.append(df)

    # ------------------- get Testing data -------------------
    paths_test = ['/data_test/ct_schema', '/data_test/gl_schema', '/data_test/or_schema']
    paths_test = ['/data_test/ct_schema','/data_test/or_schema']
    dfs_test_list = []
    for path in paths_test:
        df = load_data(path, maxsamples=None)
        dfs_test_list.append(df)

    # ------------------- get new mapping if necessary -------------------
    load_new_mappings = False
    # set to True if you want to load new mappings
    # but leave it to False if you have already loaded the mappings,
    # it takes time to build the mappings
    if load_new_mappings:
        extra_token = ['sym_aft_func', 'BoF', 'EoF', 'var_to_num']
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



    # get log file
    training_logger = Table(Column("Epoch", justify="center"),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status", pad_edge=False, box=box.ASCII)
    # Apply the reformatting function
    #all_dsl_tokens = get_all_dsl_tokens()
    #  Setting up the device for GPU usage
    device = 'cuda' if cuda.is_available() else 'cpu'

    # ---- init console ----
    # define a rich console logger
    console=Console(record=True)
    model_params={
        "MODEL":"t5-small",             # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE":8,          # training batch size
        "VALID_BATCH_SIZE":8,          # validation batch size
        "TRAIN_EPOCHS":1,              # number of training epochs
        "VAL_EPOCHS":1,                # number of validation epochs
        "LEARNING_RATE":1e-4,          # learning rate
        "MAX_SOURCE_TEXT_LENGTH":2048,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH":256,   # max length of target text
        "SEED": 42                     # set seed for reproducibility
    }

    T5Trainer(dataframe_train=dfs_train, dataframe_test=dfs_test, source_text="input", target_text="target", model_params=model_params, output_dir="outputs")

