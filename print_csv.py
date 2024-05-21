"""
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.device_count())  # Should print: Number of available GPUs
print(torch.cuda.get_device_name(0))  # Should print: Name of the first GPU


import pandas as pd
import os

# Load the CSV file
# current folder
# file_path = 'predictions.csv'
my_path = current_path = os.getcwd()
# /Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/outputs/predictions.csv
file_path = os.path.join(my_path, 'outputsgpuserver/outputs/predictions.csv')
data = pd.read_csv(file_path)

genetext = data["Generated Text"][0]
targettext = data["Actual Text"][0]
for index, row in data.iterrows():
    print(f"Generated Tokens: {row['Generated Text']}")
    print(f"Actual Text: {row['Actual Text']}")
    print()  # Add a blank line for better readability
"""
from get_datasetframe import *
from transformers import T5Tokenizer
import matplotlib.pyplot as plt

max_samples = 10000
sparse_types = ['repeated2words']
path = '/Users/juliankleutgens/training_data'
dfs = {}
for sparse_type in sparse_types:
    df = load_data(path, maxsamples=max_samples, sparse_type=sparse_type)
    dfs[sparse_type] = df



# tokenized all the data with the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
for type, df in dfs.items():
    len_tasks = []
    for i in range(len(df['input'])):
        # Tokenize the input and target
        input = tokenizer(df['input'][i], return_tensors='pt', padding='max_length', truncation=True, max_length=10000)
        # Add the tokenized input and target to the dataframe
        len_tasks.append(input['attention_mask'].sum())
    # print the average length of the tasks
    print(f'Average length of tasks for {type}: {sum(len_tasks) / len(len_tasks)}')
    # plot the distribution of the lengths
    plt.hist(len_tasks, bins=50)
    plt.title(f'Distribution of task lengths for {type}')
    plt.xlabel('Task length')
    plt.ylabel('Number of tasks')
    plt.show()


# print the list of tokens for one example
x = tokenizer(dfs['repeated2words']['input'][0])
x = x['input_ids']
# I want to see it as a list
tokens = tokenizer.convert_ids_to_tokens(x)
#print(tokens)

#print('Example of tokens for the first task:')
#print('Input:', tokenizer.decode(x))
#print()
