import torch
import pandas as pd
import os
import random
import ast
from utils import *

# Load the CSV file
# current folder
# file_path = 'predictions.csv'
my_path = current_path = os.getcwd()
# /Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/outputs/predictions.csv
# old results
path = '/Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/output_20240603_1912/predictions.csv'
path = '/Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/output_20240605_0824/predictions.csv'
# new
path = '/Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/output_20240610_2024/predictions.csv'
#path = '/Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/output_20240610_2018/predictions.csv'
file_path = os.path.join(my_path, )
data = pd.read_csv(path)

#genetext = data["Generated Text"][0]
#targettext = data["Actual Text"][0]
# Filter for the 10th Epoch and the specific test set
min_max_scores = []
epoch = 1
datasets = ['/ct_schema','/or_schema','/gl_schema']
average_accuracy_overall = data['Accuracy'].mean()
average_Ini_overall = data['Code Initializable'].mean()
average_gene_output= data['Output Generated'].mean()
print(f"Average Accuracy overall: {average_accuracy_overall}")
print(f"Average Code Initializable overall: {average_Ini_overall}")
print(f"Average Output Generated overall: {average_gene_output}")
for data_set in ['/ct_schema','/or_schema','/gl_schema']:
    filtered_dataset = data[(data['Testset'] == data_set)]
    filtered_data = data[(data['Epoch'] == epoch) & (data['Testset'] == data_set) & (data['Accuracy'] > 0)]
    # Calculate the Average Blue Score (ABS)
    average_blue_score = filtered_data['Average Blue Score'].mean()
    print(f"Average Blue Score for {data_set} in epoch {epoch}: {average_blue_score}")
    average_accuracy = filtered_dataset['Accuracy'].mean()
    print(f"Average Accuracy for {data_set} in epoch {epoch}: {average_accuracy}")
    # find maximum and minimum values and get the argument index
    #max_blue_score = filtered_data['Average Blue Score'].idxmax()
    #min_blue_score = filtered_data['Average Blue Score'].idxmin()
    #min_max_scores.append((max_blue_score, min_blue_score))
    # print the generated text and the actual text
    """
    print('-------------------- Max --------------------')
    print(f"Value of Blue Score: {filtered_data['Average Blue Score'][max_blue_score]}")
    print(f"Generated Text: {filtered_data['Generated Text'][max_blue_score]}")
    print(f"Actual Text: {filtered_data['Actual Text'][max_blue_score]}")
    print()
    """


print_will_be_done = False
if print_will_be_done:
# Print the generated text and the actual text
    for i in range(3):
        print(f"-------------------- Dataset {data['Testset'][min_max_scores[i][0]]} --------------------")
        print('-------------------- Max --------------------')
        print(f"Value of Blue Score: {data['Average Blue Score'][min_max_scores[i][0]]}")
        print(f"The Levenshtein distance is: {data['Levenshtein'][min_max_scores[i][0]]}")
        print(f"Generated Text: {data['Generated Text'][min_max_scores[i][0]]}")
        print(f"Actual Text: {data['Actual Text'][min_max_scores[i][0]]}")
        print()
        print('-------------------- Min --------------------')
        print(f"Value of Blue Score: {data['Average Blue Score'][min_max_scores[i][1]]}")
        print(f"The Levenshtein distance is: {data['Levenshtein'][min_max_scores[i][1]]}")
        print(f"Generated Text: {data['Generated Text'][min_max_scores[i][1]]}")
        print(f"Actual Text: {data['Actual Text'][min_max_scores[i][1]]}")
        print()


# Print the generated text and the actual text
# random sample
# get the last epoch
epoch = data['Epoch'].max()
filtered_data = data[(data['Epoch'] == epoch) & (data['Accuracy'] == 0) & (data['Average Blue Score'] < 0.4)]
filtered_data = data[(data['Epoch'] == epoch) & (data['Code Reconstructed'] == 1) & (data['Output Generated'] == 0)]
#filtered_data = data[(data['Epoch'] == epoch) & (data['Levenshtein'] == 4504) & (data['Average Blue Score'] > 0.4444)& (data['Average Blue Score'] < 0.45)]
l = len(filtered_data.index)
print()
print()
print(f"Number of samples with that filter: {l} out of all {len(data.index)} samples")

i = random.choice(filtered_data.index)

print()


print(f"-------------------- Dataset {data['Testset'][i]} {data['Epoch'][i]}--------------------")
print('-------------------- Random --------------------')
print(f"Value of Blue Score: {data['Average Blue Score'][i]}")
print(f"The Levenshtein distance is: {data['Levenshtein'][i]}")
print(f"Accuracy: {data['Accuracy'][i]}")
print(f"The Number of Seen Pairs is: {data['Number of Seen Pairs'][i]}")
print()
print(f"Generated Text: \n{data['Codes'][i]}")
print()
print()
idx_end=data['Actual Text'][i].index('#EoF')
actual_list = ast.literal_eval(data['Actual Text'][i])
path_mapping = path[:path.rfind('/')] + '/dsl_token_mappings_T5.json'
name = data['Codes'][i][data['Codes'][i].index('solve_')+6:data['Codes'][i].index('(')]
reconstruct_code_true = reconstruct_code(actual_list, name_idx=name, path_to_mapping=path_mapping)
print(f"Actual Text: \n{reconstruct_code_true} ")

print()
print()
print(f" predicted tokens: {data['Generated Text'][i]}")
print()
print()
print(f"Actual tokens: {data['Actual Text'][i][:idx_end+15]} ... ")




#for index, row in data.iterrows():
    #print(f"Generated Tokens: {row['Generated Text']}")
    #print(f"Actual Text: {row['Actual Text']}")
    #print()  # Add a blank line for better readability
