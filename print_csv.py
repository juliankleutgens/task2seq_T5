import torch
import pandas as pd
import os
import random

# Load the CSV file
# current folder
# file_path = 'predictions.csv'
my_path = current_path = os.getcwd()
# /Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/outputs/predictions.csv
path = '/Users/juliankleutgens/PycharmProjects/task2seq_T5/outputsgpuserver/output_20240522_2150/predictions.csv'
file_path = os.path.join(my_path, )
data = pd.read_csv(path)

#genetext = data["Generated Text"][0]
#targettext = data["Actual Text"][0]
# Filter for the 10th Epoch and the specific test set
min_max_scores = []
epoch = 9
datasets = ['/ct_schema','/or_schema','/gl_schema']
for data_set in ['/ct_schema','/or_schema','/gl_schema']:
    filtered_data = data[(data['Epoch'] == epoch) & (data['Testset'] == data_set)]
    # Calculate the Average Blue Score (ABS)
    average_blue_score = filtered_data['Average Blue Score'].mean()
    #print(f"Average Blue Score for {data_set} in epoch {epoch}: {average_blue_score}")
    # find maximum and minimum values and get the argument index
    max_blue_score = filtered_data['Average Blue Score'].idxmax()
    min_blue_score = filtered_data['Average Blue Score'].idxmin()
    min_max_scores.append((max_blue_score, min_blue_score))
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
        print(f"Generated Text: {data['Generated Text'][min_max_scores[i][0]]}")
        print(f"Actual Text: {data['Actual Text'][min_max_scores[i][0]]}")
        print()
        print('-------------------- Min --------------------')
        print(f"Value of Blue Score: {data['Average Blue Score'][min_max_scores[i][1]]}")
        print(f"Generated Text: {data['Generated Text'][min_max_scores[i][1]]}")
        print(f"Actual Text: {data['Actual Text'][min_max_scores[i][1]]}")
        print()


# Print the generated text and the actual text
# random sample
epoch = 9
filtered_data = data[(data['Epoch'] == epoch) & (data['Average Blue Score'] > 0.25)]
# get a random index
first_index = filtered_data.index[0]
last_index = filtered_data.index[-1]
i = random.randint(first_index, last_index)

print(f"-------------------- Dataset {data['Testset'][i]} {data['Epoch'][i]}--------------------")
print('-------------------- Max --------------------')
print(f"Value of Blue Score: {data['Average Blue Score'][i]}")
print()
print(f"Generated Text: {data['Generated Text'][i]}")
print()
print()
print(f"Actual Text: {data['Actual Text'][i]}")




#for index, row in data.iterrows():
    #print(f"Generated Tokens: {row['Generated Text']}")
    #print(f"Actual Text: {row['Actual Text']}")
    #print()  # Add a blank line for better readability
