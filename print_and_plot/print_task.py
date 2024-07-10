import shutil
import importlib
from inspect import getsource
import copy
import os
import json
import tqdm
import random
import zipfile
from data_scripts.get_datasetframe import load_data

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from DSL.dsl import *
from DSL.constants import *
from DSL.arc_types import *

from typing import Dict, List, Tuple, Optional

colormapping = {
    0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR',
    5: 'FIVE', 6: 'SIX', 7: 'SEVEN', 8: 'EIGHT', 9: 'NINE'
}

color_name_mapping = {
    0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
    5: 'grey', 6: 'pink', 7: 'orange', 8: 'light blue', 9: 'dark red'
}


def plot_task(
        task: Dict[str, List[Dict[str, List[List[int]]]]]
) -> None:
    """ plots a task """
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}


    train_examples = task.get('train', [])
    test_examples = task.get('test', [])
    if type(test_examples) == dict:
        all_examples = train_examples + [test_examples]
    else:
        all_examples = train_examples + test_examples
    height = 2
    width = len(all_examples)
    figure_size = (width * 4, height * 4)
    figure, axes = plt.subplots(height, width, figsize=figure_size)

    if width == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # Make axes 2D
        axes[0, 0].imshow(all_examples[0]['input'], **args)
        axes[1, 0].imshow(all_examples[0]['output'], **args)
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
    else:
        for column, example in enumerate(all_examples):
            axes[0, column].imshow(example['input'], **args)
            axes[1, column].imshow(example['output'], **args)
            axes[0, column].axis('off')
            axes[1, column].axis('off')

    figure.set_facecolor('#1E1E1E')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == '__main__':
    # Load the task data
    path_1 = '/Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/training_generated'
    path_test_data = ['/Users/juliankleutgens/PycharmProjects/task2seq_T5/data_test/ct_schema/tasks',
    '/Users/juliankleutgens/PycharmProjects/task2seq_T5/data_test/gl_schema/tasks',
    '/Users/juliankleutgens/PycharmProjects/task2seq_T5/data_test/or_schema/tasks',
    '/Users/juliankleutgens/PycharmProjects/arc-dsl-main/abstraction-and-reasoning-challenge/training']
    df = load_data(path_1, maxsamples=-1, sparse_type='None')
    names_unique_outputs = []
    num_same_out_input_grid = 0
    num_empty_grids = 0
    for i in range(len(df['input'])):
        task_grid = df['input'][i]
        input_data = task_grid[::2]
        output_data = task_grid[1::2]
        same_girds = []
        empty_grids = []
        for j in range(len(output_data)):
            same_girds.append(np.array_equal(input_data[j], output_data[j]))
            empty_grids.append(np.all(output_data[j] == 0))
        if all(same_girds):
            num_same_out_input_grid += 1
            continue
        if all(empty_grids):
            num_empty_grids += 1
            continue
        names_unique_outputs.append(df['local_path'][i])
    print(f"Number of tasks with same input and output grid: {num_same_out_input_grid}")
    print(f"Number of tasks with empty output grid: {num_empty_grids}")
    tasks_paths = [#'/Users/juliankleutgens/PycharmProjects/task2seq_T5/data/training_generated/X/00d62c1b_5etxccht.json',
                   '/Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/training_generated/X/0a0f4ris_odek5ids.json']
    random_idx = random.sample(range(len(names_unique_outputs)), 5)
    for task_path in [names_unique_outputs[i] for i in random_idx]:
        with open(task_path,'r') as file:
            task = json.load(file)
        name = task_path[task_path.rfind('/') + 1:task_path.rfind('_')]
        for test_sets in path_test_data:
            if os.path.exists(os.path.join(test_sets, name + '.json')):
                with open(os.path.join(test_sets, name + '.json'), 'r') as file:
                    task_ground_truth = json.load(file)
                    plot_task(task_ground_truth)
                break

        # Plot the task
        plot_task(task)
