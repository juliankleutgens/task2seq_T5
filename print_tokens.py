from get_datasetframe import *
from transformers import T5Tokenizer
import matplotlib.pyplot as plt

max_samples = 1
sparse_types = ['repeated2words', ]#'codeit']
paths = ['/Users/juliankleutgens/PycharmProjects/arc-dsl-main/abstraction-and-reasoning-challenge/training',
        '/Users/juliankleutgens/PycharmProjects/arc-dsl-main/abstraction-and-reasoning-challenge/evaluation',
        '/Users/juliankleutgens/PycharmProjects/arc-dsl-main/abstraction-and-reasoning-challenge/test'
         ]
#paths = ['/Users/juliankleutgens/PycharmProjects/arc-dsl-main/abstraction-and-reasoning-challenge/training']
paths = ['/Users/juliankleutgens/training_data']
dfs = {}
for path in paths:
    for sparse_type in sparse_types:
        df = load_data(path, maxsamples=max_samples, sparse_type=sparse_type)
        name = sparse_type
        dfs[name] = df

tokenizer = T5Tokenizer.from_pretrained('t5-small')
"""
# tokenized all the data with the T5 tokenizer
j = 0
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
    plt.title(f'Distribution of task lengths for {type} for dataset {paths[j][-10:]}')
    plt.xlabel('Task length')
    plt.ylabel('Number of tasks')
    plt.show()
    j += 1
"""
all_tokens = []


# print the list of tokens for one example
for df in dfs.values():
    for i in range(len(df['input'])):
        x = tokenizer(dfs['repeated2words']['input'][i])
        x = x['input_ids']
        # I want to see it as a list
        tokens = tokenizer.convert_ids_to_tokens(x)

        all_tokens += tokens

# ptint tokens as a string and not as alist
print(' '.join(all_tokens))
print(set(all_tokens))
tokens_from_task_encoder = get_tokens_from_task_encoder()
# check if all tokens are in the task encoder
for token in set(all_tokens):
    if token not in set(tokens_from_task_encoder):
        print(f'Token {token} not in task encoder')

#print('Example of tokens for the first task:')
#print('Input:', tokenizer.decode(x))
#print()
