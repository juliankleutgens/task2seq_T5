

import pandas as pd

# Importing libraries
import os
from omegaconf import OmegaConf
import logging
import torch
import torchsummary
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
from rouge import Rouge
import torch
from tqdm import tqdm
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


def train(epoch, tokenizer, model, device, loader, optimizer, training_logger, console=Console(), cfg=None,
          val_loader_list=None):
    """
    Function to be called for training with the parameters passed from main function

    """
    output_dir = cfg["output_dir"]
    test_paths = cfg["test_paths"]
    model_params = cfg["model_params"]

    model.train()
    #training_logger = Table(title=f"Training Epoch {epoch}")  # Assuming `training_logger` is a table for logging
    #training_logger.add_column("Epoch")
    #training_logger.add_column("Step")
    #training_logger.add_column("Loss")

    # Add tqdm progress bar for the training loop
    for step, data in tqdm(enumerate(loader, 0), total=len(loader), desc=f"Training Epoch {epoch}", leave=False):
        if step > cfg["num_of_itr"] and cfg["num_of_itr"] != -1:
            print(f"Number of iterations {step} reached. Stopping the training")
            break

        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if step % 10 == 0:
            #training_logger.add_row(str(epoch), str(step), str(loss.item()))
            #console.print(training_logger)
            wandb.log({"epoch": epoch, "step": step, "loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")

    first_iteration = True
    # Initialize accumulators for scores
    total_bleu_score = 0
    total_rouge_1_f = 0
    total_rouge_2_f = 0
    total_rouge_l_f = 0
    num_datasets = len(val_loader_list)
    for i, val_loader in enumerate(val_loader_list):
        console.print(f"Validation for dataset {test_paths[i]}")
        predictions, actuals, avg_bleu_score, avg_rouge_score = validate(epoch=epoch, tokenizer=tokenizer, model=model, device=device,
                                        loader=val_loader,
                                        model_params=model_params, num_batches=cfg["model_params"]["VALID_BATCH_SIZE"])
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        # print(final_df)
        if first_iteration:
            final_df.to_csv(os.path.join(output_dir, 'predictions.csv'), mode='w', header=True, index=False)
            first_iteration = False
        else:
            final_df.to_csv(os.path.join(output_dir, 'predictions.csv'), mode='a', header=False, index=False)
        # Accumulate scores
        total_bleu_score += avg_bleu_score
        total_rouge_1_f += avg_rouge_score["rouge-1"]
        total_rouge_2_f += avg_rouge_score["rouge-2"]
        total_rouge_l_f += avg_rouge_score["rouge-l"]

    # Calculate average scores across all datasets
    avg_bleu_score_overall = total_bleu_score / num_datasets
    avg_rouge_1_f_overall = total_rouge_1_f / num_datasets
    avg_rouge_2_f_overall = total_rouge_2_f / num_datasets
    avg_rouge_l_f_overall = total_rouge_l_f / num_datasets

    # Log the overall average metrics to wandb
    wandb.log({
        "avg_bleu_score": avg_bleu_score_overall,
        "rouge-1_f": avg_rouge_1_f_overall,
        "rouge-2_f": avg_rouge_2_f_overall,
        "rouge-l_f": avg_rouge_l_f_overall,
    })
    print(f"avg_bleu_score: {avg_bleu_score_overall}")
    print(f"rouge-1_f: {avg_rouge_1_f_overall}")
    print(f"rouge-2_f: {avg_rouge_2_f_overall}")
    print(f"rouge-l_f: {avg_rouge_l_f_overall}")

def validate(epoch, tokenizer, model, device, loader, model_params, num_batches):
    """
    Function to evaluate model for predictions
    """
    model.eval()
    predictions = []
    actuals = []
    bleu_scores = []
    rouge = Rouge()
    rouge_scores = []
    iteration = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=num_batches, desc="Eval Batch", leave=False):
            iteration += 1

            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=model_params["MAX_TARGET_TEXT_LENGTH"],
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
                    one_sample_pred.append(token)
                our_token_sample = map_back(one_sample_pred)
                preds.append(our_token_sample)

            # Convert the targets into a list
            target = []
            for t in y:
                one_sample_target = []
                for _id in t:
                    token = tokenizer.decode(_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    one_sample_target.append(token)
                our_token_sample = map_back(one_sample_target)
                target.append(our_token_sample)

            predictions.extend(preds)
            actuals.extend(target)
            #if iteration == int(len(loader.dataset)/num_batches):
                #break
            # Calculate BLEU scores
            for pred, act in zip(preds, target):
                score = sentence_bleu([act], pred, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(score)

            # Calculate ROUGE scores
            for pred, act in zip(preds, target):
                rouge_score = rouge.get_scores(' '.join(pred), ' '.join(act))
                rouge_scores.append(rouge_score[0])

        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        avg_rouge_score = {
            "rouge-1": sum([s["rouge-1"]["f"] for s in rouge_scores]) / len(rouge_scores),
            "rouge-2": sum([s["rouge-2"]["f"] for s in rouge_scores]) / len(rouge_scores),
            "rouge-l": sum([s["rouge-l"]["f"] for s in rouge_scores]) / len(rouge_scores),
        }

    return predictions, actuals, avg_bleu_score, avg_rouge_score