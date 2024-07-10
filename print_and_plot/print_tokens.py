from data_scripts.get_datasetframe import *
from transformers import T5Tokenizer
import matplotlib.pyplot as plt

max_samples = 100000
sparse_types = ['codeit', ]#'codeit']
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
all_tokens = []


# print the list of tokens for one example
number_of_tokens_in_task = []
for df in dfs.values():
    for i in range(len(df['input'])):
        x = tokenizer(dfs['codeit']['input'][i])
        x = x['input_ids']
        # I want to see it as a list
        tokens = tokenizer.convert_ids_to_tokens(x)

        number_of_tokens_in_task.append(len(tokens))
        all_tokens += tokens
# plot the distribution of the number of tokens in a task
plt.hist(number_of_tokens_in_task, bins=100)
plt.show()
print(f"The average number of tokens in a task is: {sum(number_of_tokens_in_task)/len(number_of_tokens_in_task)}")
