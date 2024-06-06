

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
import Levenshtein
from dataloader import DataSetClass


def train(epoch, tokenizer, model, device, loader, optimizer, console=Console(), cfg=None,
          val_loader_list=None):
    """
    Function to be called for training with the parameters passed from main function

    """
    output_dir = cfg["output_dir"]
    test_paths = cfg["test_paths"]
    model_params = cfg["model_params"]

    model.train()
    print(f"The model is on the device: {next(model.parameters()).device}")


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
    metrics_blue = {"epoch": epoch}
    metrics_leven = {"epoch": epoch}
    # Initialize accumulators for scores
    total_bleu_score = 0
    total_levenshtein_distance = 0
    num_datasets = len(val_loader_list)
    for i, val_loader in enumerate(val_loader_list):
        test_set = test_paths[i][test_paths[i].rfind('/'):]
        if test_set == '/training':
            test_set = 'Real ARC Training Data'
        console.print(f"Validation for dataset {test_paths[i]}")
        predictions, actuals, avg_bleu_score, bleu_scores, avg_levenshtein_distance, levenshtein_distances= validate(epoch=epoch,
                                        tokenizer=tokenizer, model=model, device=device, loader=val_loader,
                                        model_params=model_params, num_batches=cfg["model_params"]["VALID_BATCH_SIZE"])
        final_df = pd.DataFrame({'Epoch':epoch,
                                 'Testset': test_set,
                                 'Average Blue Score':bleu_scores,
                                 'Levenshtein': levenshtein_distances,
                                 'Generated Text': predictions,
                                 'Actual Text': actuals})

        if epoch == 0 and i == 0:
            final_df.to_csv(os.path.join(output_dir, 'predictions.csv'), mode='w', header=True, index=False)
        else:
            final_df.to_csv(os.path.join(output_dir, 'predictions.csv'), mode='a', header=False, index=False)
        # Accumulate scores
        total_bleu_score += avg_bleu_score
        total_levenshtein_distance += avg_levenshtein_distance
        # Log dataset-specific metrics to WandB
        #wandb.log({
        #    f"{test_set}/avg_bleu_score": avg_bleu_score,
        #    f"{test_set}/avg_levenshtein_distance": avg_levenshtein_distance,
        #    "epoch": epoch
        #})
        # Collect dataset-specific metrics
        metrics_blue[f"{test_set}/bleu_score"] = avg_bleu_score
        metrics_leven[f"{test_set}/levenshtein_distance"] = avg_levenshtein_distance

    # Calculate average scores across all datasets
    avg_bleu_score_overall = total_bleu_score / num_datasets
    avg_levenshtein_overall = total_levenshtein_distance / num_datasets
    metrics_blue["bleu_score_overall"] = avg_bleu_score_overall
    metrics_leven["avg_levenshtein_distance_overall"] = avg_levenshtein_overall
    # Log the overall average metrics to WandB
    #wandb.log({
    #    "avg_bleu_score_overall": avg_bleu_score_overall,
    #    "avg_levenshtein_distance_overall": avg_levenshtein_overall,
    #    "epoch": epoch
    #})
    wandb.log(metrics_blue)
    wandb.log(metrics_leven)
    print(f"avg_bleu_score: {avg_bleu_score_overall}")
    print(f"avg_levenshtein_distance: {avg_levenshtein_overall}")
    return metrics_blue, metrics_leven


def validate(epoch, tokenizer, model, device, loader, model_params, num_batches):
    """
    Function to evaluate model for predictions
    """
    model.eval()
    predictions = []
    actuals = []
    bleu_scores = []
    levenshtein_distances = []
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
                    token = tokenizer.convert_ids_to_tokens(int(_id))
                    one_sample_pred.append(token)
                our_token_sample = map_back(one_sample_pred)
                preds.append(our_token_sample)

            # Convert the targets into a list
            target = []
            for t in y:
                one_sample_target = []
                for _id in t:
                    token = tokenizer.convert_ids_to_tokens(int(_id))
                    one_sample_target.append(token)
                our_token_sample = map_back(one_sample_target)
                target.append(our_token_sample)

            predictions.extend(preds)
            actuals.extend(target)
            #if iteration == int(len(loader.dataset)/num_batches):
                #break
            # Calculate BLEU scores
            for pred, act in zip(preds, target):
                # trim the list of if the token #EoF is present
                if '#EoF' in pred:
                    pred_blue = pred[:pred.index('#EoF')+1]
                else:
                    pred_blue = pred
                if '#EoF' in act:
                    act_blue = act[:act.index('#EoF')+1]
                else:
                    act_blue = act
                score = sentence_bleu([act_blue], pred_blue, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(score)

                # Flatten the lists of tokens into strings for Levenshtein distance
                pred_str = ''.join(pred)
                act_str = ''.join(act)
                levenshtein_distance = Levenshtein.distance(pred_str, act_str)
                levenshtein_distances.append(levenshtein_distance)

        avg_levenshtein_distance = sum(levenshtein_distances) / len(levenshtein_distances)
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)


    return predictions, actuals, avg_bleu_score, bleu_scores, avg_levenshtein_distance, levenshtein_distances