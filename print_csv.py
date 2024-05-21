import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.device_count())  # Should print: Number of available GPUs
print(torch.cuda.get_device_name(0))  # Should print: Name of the first GPU

"""
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