

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
from torch.nn import CrossEntropyLoss
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
from utils import *


def train_and_validate(epoch, tokenizer, model, device, loader, optimizer, console=Console(), cfg=None,
          val_loader_list=None):
    """
    Function to be called for training with the parameters passed from main function
    """
    output_dir = cfg["output_dir"]
    test_paths = cfg["test_paths"]
    train_on_multiple_gpus = cfg["train_on_multiple_gpus"]

    model.train()
    print(f"The model is on the device: {next(model.parameters()).device}")
    percent_of_seen_pairs = 0

    # ------------------- Training Loop -------------------
    iter_70_percent = int(len(loader) * 0.7)
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
        percent_of_seen_pairs += data["percent_of_seen_pairs"].to(torch.float).sum()


        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if cfg["weighted_loss"] == True and step > iter_70_percent:
            try:
                loss = weighted_loss(outputs, lm_labels)

            except Exception as e:
                loss = loss
        if loss.dim() != 0 & train_on_multiple_gpus:
            # the loss must be a scaler
            loss = loss.mean()

        if step % 10 == 0:
            wandb.log({"epoch": epoch, "step": step, "loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    console.log(f"[Saving Model]...\n")
    percent_of_seen_pairs = percent_of_seen_pairs/len(loader.dataset) if cfg["num_of_itr"] == -1 else percent_of_seen_pairs/(step*cfg["model_params"]["TRAIN_BATCH_SIZE"])
    print(f"Training: The T5 Model saw on average {percent_of_seen_pairs*100}% of the input output pairs")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")

    if train_on_multiple_gpus:
        # Save the model correctly by accessing the underlying model from DataParallel
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(path)
    else:
        model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # ------------------- Validation Loop -------------------
    if not epoch % (cfg["model_params"]["VAL_EPOCHS"]) == 0:
        return 0,0,0
    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    metrics_blue = {"epoch": epoch}
    metrics_leven = {"epoch": epoch}
    accuracy = {}
    num_reconstructed_codes = {}
    num_generated_outputs = {}
    percent_of_seen_pairs = {}
    # Initialize accumulators for scores
    total_bleu_score = 0
    total_levenshtein_distance = 0
    num_datasets = len(val_loader_list)

    for i, val_loader in enumerate(val_loader_list):
        test_set = test_paths[i][test_paths[i].rfind('/'):]
        if test_set == '/training':
            test_set = 'Real ARC Training Data'
        console.print(f"Validation for dataset {test_paths[i]}")
        try:
            output = validate(epoch=epoch,tokenizer=tokenizer, model=model, device=device, loader=val_loader,
                            cfg=cfg, num_batches=cfg["model_params"]["VALID_BATCH_SIZE"], dataset_path=test_paths[i])
        except Exception as e:
            print(f"Error validating: {e}")
            continue
        final_df = pd.DataFrame({'Epoch':epoch,
                                 'Testset': test_set,
                                 'Average Blue Score':output['bleu_scores'],
                                 'Levenshtein': output['levenshtein_distances'],
                                 'Name': output['names'],
                                 'Generated Text': output['predictions'],
                                 'Actual Text': output['actuals'],
                                 'Codes': output['codes'],
                                 'Accuracy': output['accuracies'],
                                 'Code Reconstructed': output['codes_reconstructed'],
                                 'Code Initializable': output['codes_initializable'],
                                 'Output Generated': output['outputs_generated'],
                                 'Error': output['errors'],
                                 'Number of Seen Pairs': output["percent_of_seen_pairs"]})

        if not os.path.exists(os.path.join(output_dir, 'predictions.csv')):
            final_df.to_csv(os.path.join(output_dir, 'predictions.csv'), mode='w', header=True, index=False)
        else:
            final_df.to_csv(os.path.join(output_dir, 'predictions.csv'), mode='a', header=False, index=False)
        # Accumulate scores
        total_bleu_score += output['avg_bleu_score']
        total_levenshtein_distance += output['avg_levenshtein_distance']

        # Collect dataset-specific metrics
        metrics_blue[f"{test_set}/bleu_score"] = output['avg_bleu_score']
        metrics_leven[f"{test_set}/levenshtein_distance"] = output['avg_levenshtein_distance']
        accuracy[f"{test_set}/accuracy"] = sum(output['accuracies'])/len(output['accuracies'])
        print(f"For the dataset {test_set} then model has an accuracy of {accuracy[f'{test_set}/accuracy']*100}%")
        num_reconstructed_codes[f"{test_set}/percent_reconstructed_codes"] = sum(output['codes_reconstructed'])/len(output['codes_reconstructed'])
        print(f"For the dataset {test_set} then model reconstructed on average {num_reconstructed_codes[f'{test_set}/percent_reconstructed_codes']*100}% of the codes")
        num_generated_outputs[f"{test_set}/percent_generated_outputs"] = sum(output['outputs_generated'])/len(output['outputs_generated'])
        print(f"For the dataset {test_set} then model generated on average {num_generated_outputs[f'{test_set}/percent_generated_outputs']*100}% of the outputs")
        percent_of_seen_pairs[f"{test_set}/percent_of_seen_pairs"] = sum(output["percent_of_seen_pairs"])/len(output["percent_of_seen_pairs"])
        print(f"For the dataset {test_set} then model saw on average {percent_of_seen_pairs[f'{test_set}/percent_of_seen_pairs']*100}% of the input output pairs")


    # Calculate average scores across all datasets
    avg_bleu_score_overall = total_bleu_score / num_datasets
    avg_levenshtein_overall = total_levenshtein_distance / num_datasets
    accuracy['accuracy_overall'] = sum(accuracy.values())/len(accuracy.values())
    num_reconstructed_codes['percent_reconstructed_codes_overall'] = sum(num_reconstructed_codes.values())/len(num_reconstructed_codes.values())
    num_generated_outputs['percent_generated_outputs_overall'] = sum(num_generated_outputs.values())/len(num_generated_outputs.values())
    metrics_blue["bleu_score_overall"] = avg_bleu_score_overall
    metrics_leven["avg_levenshtein_distance_overall"] = avg_levenshtein_overall
    # Log the overall average metrics to WandB
    wandb.log(metrics_blue)
    wandb.log(metrics_leven)
    wandb.log(accuracy)
    wandb.log(num_reconstructed_codes)
    wandb.log(num_generated_outputs)


    print(f"avg_bleu_score: {avg_bleu_score_overall}")
    print(f"avg_levenshtein_distance: {avg_levenshtein_overall}")
    print(f"accuracy: {accuracy['accuracy_overall']}")
    print(f"percent_reconstructed_codes: {num_reconstructed_codes['percent_reconstructed_codes_overall']}")
    print(f"percent_generated_outputs: {num_generated_outputs['percent_generated_outputs_overall']}")

    return metrics_blue, metrics_leven, accuracy


def validate(epoch, tokenizer, model, device, loader, cfg, num_batches, dataset_path):
    """
    Function to evaluate model for predictions
    """
    model_params = cfg["model_params"]
    train_on_multiple_gpus = cfg["train_on_multiple_gpus"]
    first_time_validation = epoch == cfg["model_params"]["VAL_EPOCHS"]
    if train_on_multiple_gpus:
        model = get_model(model)
    model.eval()
    predictions = []
    actuals = []
    bleu_scores = []
    levenshtein_distances = []
    names = []
    codes = []
    accuracies = []
    codes_reconstructed = []
    codes_initializable = []
    outputs_generated = []
    errors = []
    percents_of_seen_pairs = []
    iteration = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=num_batches, desc="Eval Batch", leave=False):
            iteration += 1

            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            name = data['name']
            path = data['local_path']

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
            j = 0
            for gen_id in generated_ids:
                one_sample_pred = []
                for _id in gen_id:
                    token = tokenizer.convert_ids_to_tokens(int(_id))
                    one_sample_pred.append(token)
                our_token_sample = map_back(one_sample_pred, cfg["path_to_mapping"])
                preds.append(our_token_sample)
                try:
                    result = reconstruct_and_execute_code(our_token_sample, path[j], name[j], path_to_mapping=cfg["path_to_mapping"])
                    # concatenate the results

                except Exception as e:
                    result = {
                        'code': '',
                        'accuracy': 0,
                        'code_reconstructed': False,
                        'code_initializable': False,
                        'output_generated': False,
                        'error_message': e
                    }
                codes.append(result['code'])
                accuracies.append(result['accuracy'])
                codes_reconstructed.append(int(result['code_reconstructed']))
                codes_initializable.append(int(result['code_initializable']))
                outputs_generated.append(int(result['output_generated']))
                errors.append(result['error_message'])
                j += 1


            # Convert the targets into a list
            # and reconstruct the code for each sample in the batch
            target = []
            j = 0
            for t in y:
                one_sample_target = []
                for _id in t:
                    token = tokenizer.convert_ids_to_tokens(int(_id))
                    one_sample_target.append(token)

                our_token_sample = map_back(one_sample_target, cfg["path_to_mapping"])
                target.append(our_token_sample)
                if first_time_validation:
                    try:
                        r = reconstruct_and_execute_code(our_token_sample, path[j], name[j], path_to_mapping=cfg["path_to_mapping"])
                        if not r['accuracy'] == 1:
                            print(f"The ground true code for {name[j]} is not correct!!!")
                    except Exception as e:
                        print(f"Error in task {name[j]} reconstructing and executing: {e}")
                j += 1
            predictions.extend(preds)
            actuals.extend(target)
            percents_of_seen_pairs.extend(data["percent_of_seen_pairs"])
            names.extend(name)

            try:
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
                    score = sentence_bleu([act_blue], pred_blue,
                                          smoothing_function=SmoothingFunction().method1)
                    bleu_scores.append(score)

                    # Flatten the lists of tokens into strings for Levenshtein distance
                    pred_str = ''.join(pred)
                    act_str = ''.join(act)
                    levenshtein_distance = Levenshtein.distance(pred_str, act_str)
                    levenshtein_distances.append(levenshtein_distance)
            except Exception as e:
                print(f"Error calculating BLEU scores and Levenshtein distances: {e}")
                for _ in range(len(preds)):
                    levenshtein_distances.append(0)
                    bleu_scores.append(0)
        try:
            if len(levenshtein_distances) > 0:
                avg_levenshtein_distance = sum(levenshtein_distances) / len(levenshtein_distances)
            else:
                avg_levenshtein_distance = 0
                print(f"No levenshtein distances calculated, for the task {data['name']}")
            if len(bleu_scores) > 0:
                avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
            else:
                avg_bleu_score = 0
                print(f"No bleu scores calculated, for the task {data['name']}")
        except Exception as e:
            print(f"Error calculating average BLEU scores and Levenshtein distances: {e}")
            avg_bleu_score = 0
            avg_levenshtein_distance = 1000

    # Return the predictions, actuals, and scores as dictionary
    output = {'predictions': predictions, 'actuals': actuals, 'avg_bleu_score': avg_bleu_score,
            'bleu_scores': bleu_scores, 'avg_levenshtein_distance': avg_levenshtein_distance,
            'levenshtein_distances': levenshtein_distances, 'names': names,
            'codes': codes, 'accuracies': accuracies, 'codes_reconstructed': codes_reconstructed,
            'codes_initializable': codes_initializable, 'outputs_generated': outputs_generated, 'errors': errors,
            'percent_of_seen_pairs': percents_of_seen_pairs}

    return output