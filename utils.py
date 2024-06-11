# Let's read the content of the uploaded Python file to see what needs to be reformatted.

import inspect

# %load_ext autoreload
# %autoreload 2
# full_demo()
import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Optional
from constants import COLORMAP, NORM, TASK_DICT, Example, Task
from numpy.typing import NDArray
import numpy as np
import os
import dsl

from tqdm import tqdm
from transformers import T5Tokenizer

def tokenize_expression(expression):
    """Helper function to tokenize all variable references in an expression."""
    tokens = expression.split()
    new_tokens = []
    for token in tokens:
        if 'x' in token and re.match(r"x\d+", token):
            # If token is a variable, tokenize it
            new_tokens.append(tokenize_variable(token))
        else:
            # Otherwise, add it unchanged
            new_tokens.append(token)
    return ' '.join(new_tokens)


def tokenize_variable(var_name):
    """Helper function to tokenize variable names with digit separation."""
    match = re.match(r"(x)(\d+)", var_name)
    if match:
        var_prefix, var_numbers = match.groups()
        # Create a space-separated string of the prefix and each digit
        digit_tokens = ' '.join(var_numbers)  # Splitting the number into separate digits
        return f"{var_prefix} {digit_tokens}"
    return var_name  # Return original if it does not match expected pattern

# Let's reformat the file content based on the simplified DSL syntax described by the user
def reformat_dsl_code(original_code, extra_token = None):
    lines = original_code.split('\n')
    new_code = []
    previous = []  # Track variables defined in previous lines
    for line in lines:
        if line.strip().startswith('x') and '=' in line:  # Process lines defining variables
            parts = line.split('=')
            var_name = parts[0].strip()
            expression = parts[1].strip().replace('(', ' ').replace(')', '').replace(',', ' ')

            if extra_token is not None and 'sym_aft_func' in extra_token:
                tokens = expression.split()
                tokens.insert(1, ';')
                expression = ' '.join(tokens)

            if extra_token is not None and 'underscore' in extra_token:
                tokens = expression.split()
                tokens[0] = '_' + tokens[0]
                expression = ' '.join(tokens)

            new_line = f'#newline {var_name} {expression}'

            if extra_token is not None and 'var_to_num' in extra_token:
                # Tokenize the variable name
                var_tokenized = tokenize_variable(var_name)
                # Tokenize all variables within the expression
                expression_tokenized = tokenize_expression(expression)
                new_line = f'#newline {var_tokenized} {expression_tokenized}'

            if not (extra_token is not None and 'prev' in extra_token):
                new_code.append(new_line)

            # ----------- Handle the case when 'prev' is in extra_token ------------ #
            else: # if prev is in extra_token
                tokens = expression.split()
                function = tokens[0]
                args = tokens[1:]
                modified_args = []
                for arg in args:
                    if arg in previous[:-1] and arg[0] =='x':  # Checks if the argument is at least two steps back
                        # if arg in previous[:-1]:  # Checks if the argument is at least two steps back
                        # , but then previous.append(var_name)

                        modified_args.append('prev ' + arg)
                    else:
                        modified_args.append(arg)
                if extra_token is not None and 'var_to_num' in extra_token:
                    modified_args_tokenized = tokenize_expression(' '.join(modified_args))
                    new_line = f'#newline {var_tokenized} {function} {modified_args_tokenized}'
                else:
                    new_line = f'#newline {var_name} {function} ' + ' '.join(modified_args)
                new_code.append(new_line)
                previous.append(arg)  # Remember this var as defined

        elif line.strip().startswith('O ='):  # Handle the output assignment line
            parts = line.split('=')
            expression = parts[1].strip().replace('(', ' ').replace(')', '').replace(',', ' ')

            if extra_token is not None and 'sym_aft_func' in extra_token:
                tokens = expression.split()
                tokens.insert(1, ';')
                expression = ' '.join(tokens)

            if extra_token is not None and 'var_to_num' in extra_token:
                expression_tokenized = tokenize_expression(expression)
                new_line = f'#newline O {expression_tokenized}'
            else:
                new_line = f'#newline O {expression}'
            new_code.append(new_line)

        # ----------- Handle the case when for the begining of the function is in extra_token ------------ #
        elif line.strip().startswith('def'):  #BoF begining of Function
            new_line = "#BoF"
            previous = [] # the previous_vars are reset because we are in a new function
            if extra_token is not None and 'BoF' in extra_token:
                new_code.append(new_line) # Function definition line

        elif line.find('return') != -1:  # Handle the return statement
            new_line = "#EoF"
            if extra_token is not None and 'EoF' in extra_token:
                new_code.append(new_line)

    string_print = '\n'.join(new_code)
    list_of_tokens = string_print.split()
    return string_print, list_of_tokens

def get_all_dsl_tokens():
    # Get all functions from the DSL module
    dsl_functions = inspect.getmembers(dsl, inspect.isfunction)
    # Extract the function names
    dsl_tokens = [func[0] for func in dsl_functions]
    return dsl_tokens


def read_solver_file(path):
    # Get the current working directory
    if path[-12:] == 'verifiers.py':
        with open(path, 'r') as file:
            data = file.read()
        return data
    if path[:7] == "/Users/" or path[:6] == "/home/":
        path_solver = path + '/solvers.py'
    else:
        path_solver = os.getcwd() + path + '/solvers.py'
    #path_solver = os.getcwd() + path  + '/solvers.py'
    with open(path_solver, 'r') as file:
        data = file.read()
    return data


def decode_json_task(file_path: str) -> Task:
    with open(file_path) as f:
        data = json.load(f)

    examples = data["train"] + data["test"]

    task: Task = []
    for example in examples:
        input = example["input"]
        output = example["output"]
        example = Example(
            input=np.array(input, dtype=np.uint8),
            output=np.array(output, dtype=np.uint8),
        )
        task.append(example)

    return task
def decode_json_task_test(file_path: str) -> Task:
    with open(file_path) as f:
        data = json.load(f)

    examples = data["train"] + data["test"]

    task: Task = []
    for example in examples:
        input = example["input"]
        example = Example(
            input=np.array(input, dtype=np.uint8),
            output=np.array(input, dtype=np.uint8),
        )
        task.append(example)

    return task

def normalize_task(
        task: Task, h: int = 30, w: int = 30, max_input_output_pairs: int = 4
) -> NDArray[np.uint8]:
    """
    Given a task normalize the task examples so that all of them are placed
    at the top left corner of a hxw grid. We use 10 as value to flag a pixel
    not being part of the task. Dimensions: [N_Example, 0 | 1, h, w], where
    0 := input, 1 := output. e.g To get output of the example 2, X[1, 1, :, :]

    Args:
        task: Task to be normalized
        h: Height of the normalized grid
        w: Width of the normalized grid
        max_input_otput_pairs: Maximum input output pairs of the normalized task.
    """
    MAX_PAIRS = max_input_output_pairs  # noqa

    N = len(task)  # noqa

    if N > MAX_PAIRS:
        raise ValueError(f"A task cannot have more than {MAX_PAIRS} examples")

    res = np.empty(shape=(MAX_PAIRS, 2, h, w), dtype=np.uint8)

    # TODO: We use the 10 value as a flag. Maybe I need
    # to test if the examples contain values that are not
    # within [0, 9].
    res.fill(10)

    for i in range(N):
        inp_shape = task[i].input.shape
        out_shape = task[i].output.shape

        if inp_shape[0] > h or inp_shape[1] > w:
            raise ValueError(f"Input {i} exceeds the dimension of the normalized grid.")

        if out_shape[0] > h or out_shape[1] > w:
            raise ValueError(
                f"Output {i} exceeds the dimension of the normalized grid."
            )

        res[i, 0, : inp_shape[0], : inp_shape[1]] = task[i].input
        res[i, 1, : out_shape[0], : out_shape[1]] = task[i].output

    return res






def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import json
from pathlib import Path
import inflect

def number_to_words(number):
    p = inflect.engine()
    return p.number_to_words(number)
def number_to_words_one_till_ten(number):
    dic = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
    return dic[number]

def number_to_color(number):
    dic = {0: '▁Black', 1: '▁Blue', 2: '▁Red', 3: '▁Green', 4: '▁Yellow', 5: '▁Gray', 6: '▁Purple', 7: '▁Orange', 8: '▁Azure', 9: '▁Brown', 10: '▁White'}
    return dic[number]

def get_tokens_from_task_encoder():
    # in this function we will pass the task tokens and get the tokens that are not to be mapped to
    # since the task to T5 tokens is a standard mapping, we have a constant and fix tokens which are use
    special_sym = ['</s>', '▁new', '|', ';', '|', 'x', '▁pair', '▁input', '▁output']
    word_numbers = ['▁zero', '▁one', '▁two', '▁three', '▁four', '▁five', '▁six', '▁seven', '▁eight', '▁nine', '▁ten']
    # the numbers 0 till 50 as a sting in a list
    color_words = ['▁Black', '▁Blue', '▁Red', '▁Green', '▁Yellow', '▁Gray', '▁Purple', '▁Orange', '▁Azure', '▁Brown', '▁White']
    numbers = [str(i) for i in range(51)]
    return special_sym + word_numbers + numbers + color_words


def convert2sparse_repeated_numbers(task):
    def fromArray2(narray):
        # Get the dimensions of the narray
        dim_ = narray.shape
        # Create a flattened string representation of the narray with color indexes
        flatgrid_str = f"{dim_[0]}x{dim_[1]}:  "
        flatgrid_str = ''
        for row_ in narray:
            idx = 0
            while idx < len(row_):
                count = 0
                while idx + count < len(row_) and row_[idx] == row_[idx + count]:
                    count += 1
                if count >= 4:
                    flatgrid_str += ' ' + str(number_to_color(row_[idx])) + 'x' + str(count)
                else:
                    for num in row_[idx:idx + count]:
                        word = number_to_color(num)
                        flatgrid_str += ' ' + word
                idx += count
            flatgrid_str += ';'
        return flatgrid_str

    input_ = task[0]
    output_ = task[1]
    sparse_task = ' input' + fromArray2(input_) + ' output' + fromArray2(output_)
    return sparse_task


def load_token_mappings_utils(filename="dsl_token_mappings_T5.json"):
    """Loads token mappings from a JSON file, handling potential errors."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing token mappings found.")
        return None

def convert2sparse(task):
    def fromTuple(tuple_):
        dim_ = (len(tuple_),len(tuple_[0]))
        color_ = set(sum(tuple_,()))
        color_indx = {color: index for index, color in enumerate(color_)}
        flatgrid_str = ''
        for row_ in tuple_:
            row_ = tuple(color_indx[elem] for elem in row_)
            flatgrid_str = flatgrid_str + ' ' + str(row_)[1:-1].replace(" ", "")
        out = str(dim_[0])+'x'+str(dim_[1])+' bg='+' '.join(str(item) for item in color_)+'='+flatgrid_str[1:]
        return out

    def fromArray(narray):
        # Get the dimensions of the narray
        dim_ = narray.shape

        # Find unique colors and assign indexes
        unique_colors = np.unique(narray)
        color_indx = {color: index for index, color in enumerate(unique_colors)}

        # Create a flattened string representation of the narray with color indexes
        flatgrid_str = ''
        for row_ in narray:
            indexed_row = [color_indx[elem] for elem in row_]
            flatgrid_str += ' ' + ', '.join(map(str, indexed_row))

        # Format output string
        out = f"{dim_[0]}x{dim_[1]} bg=" + ' '.join(map(str, unique_colors)) + '=' + flatgrid_str[1:]
        return out

    if isinstance(task, dict):
        input_ = task['input']
        output_ = task['output']
        return fromTuple(input_)+'|'+fromTuple(output_)
    else: # if task is a narray
        input_ = task[0]
        output_ = task[1]
        return fromArray(input_)+'|'+fromArray(output_)


def convert2grid(sparse_task):
    def fromSparse(sparse):
        sparse = sparse.split('=')
        color_ = sparse[1].split(' ')
        grid_ = sparse[2]
        for i in range(len(color_)): # this is okay if number of colors are less than 11 (color from 0 to 9)
            grid_ = grid_.replace(str(i),color_[i])
        grid_ = '(('+grid_.replace(' ','),(')+'))'
        grid_ = ast.literal_eval(grid_)
        return grid_
    task_ = sparse_task.split('|')
    input_ = fromSparse(task_[0])
    output_ = fromSparse(task_[1])
    task_dict = {'input': input_, 'output': output_}
    return task_dict

def get_model(model):
    """
    Utility function to get the underlying model if wrapped in DataParallel.
    """
    return model.module if hasattr(model, 'module') else model


def reconstruct_code(token_list, name_idx='005t822n', path_to_mapping='dsl_token_mappings_T5.json'):
    tokens = token_list
    output = []
    current_function = []
    # search for #newline from the back of the list
    last_idx_newline = len(tokens) - tokens[::-1].index('#newline') - 1
    i = 0

    # make the function header with the name of the function
    import_statement = "from dsl import * \nfrom constants import *\n"
    output.append(import_statement)
    function_header = f"\n\ndef solve_{name_idx}(I: Grid) -> Grid:"
    current_function.append(function_header)
    while i < len(tokens):
        token = tokens[i]

        if token == '#newline' or i==0 or token == '<pad>':
            try:
                current_function.append('\n')  # Add newline

                # get the indexes of the beginning and the end of the line
                if i == last_idx_newline:
                    # for last line in the code, it is only one time true in the end
                    idx_end_line = tokens.index('#EoF', i + 1)
                else:
                    idx_end_line = tokens.index('#newline', i + 1)
                    if '#EoF' in tokens[i + 1:idx_end_line]:
                        idx_of_end_func = tokens.index('#EoF', i + 1)
                        if idx_end_line > idx_of_end_func:
                            idx_end_line = idx_of_end_func


                code_line = rebuild_the_line(tokens, i, idx_end_line, path_to_mapping)
                current_function.extend(code_line)
                i = idx_end_line  # Move index to next relevant token
            except:
                current_function.append('\n')
                current_function.append('    # there was an error in the code in this line')
                i += 1
        elif token == '#EoF':
            current_function.append("\n    return O")
            output.append(''.join(current_function))
            current_function = []
            break # Move to next token

        else:
            current_function.append(f"\n    # generated {token}")
            i += 1  # Skip unrecognized tokens or handle other cases if needed

    if current_function:
        output.append(''.join(current_function))

    # Return a single string that concatenates all function definitions
    return ''.join(output)


def rebuild_the_line(tokens, idx_beginning, idx_end_line, path_to_mapping):
    # load mapping
    dsl_token_mappings = load_token_mappings_utils(filename=path_to_mapping) # tokens[idx_beginning:idx_end_line]
    idx_of_break = tokens.index(';', idx_beginning + 1, idx_end_line)
    function_name = tokens[idx_of_break - 1]
    sub_tokens = tokens[idx_beginning:idx_of_break]
    if sub_tokens.count('x') > 1 or (sub_tokens.count('x') + sub_tokens.count('O')) > 1:
        reversed_sub_tokens = sub_tokens[::-1]
        reverse_index = reversed_sub_tokens.index('x')
        idx_var_name = idx_of_break - reverse_index - 1
        function_name = ''.join(tokens[idx_var_name:idx_of_break])
        idx_var_name = idx_var_name + 1
    else:
        idx_var_name = idx_of_break

    arguments = tokens[idx_of_break + 1:idx_end_line]
    args = []
    unrecognized_tokens = []
    # get all the arguments which are the inputs to the function
    for1 = 0
    # there is a unrecognized tokens
    while for1 < len(arguments):
        if not arguments[for1] in dsl_token_mappings:
            unrecognized_tokens.append(arguments[for1])
            for1 += 1
        elif arguments[for1] == 'x':
            for2 = for1 + 1
            argument = 'x'
            while for2 < len(arguments):
                if arguments[for2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    argument += arguments[for2]
                    for2 += 1
                else:
                    break
            args.extend([argument])
            for1 = for2

        else:
            args.extend([arguments[for1]])
            for1 += 1

    if tokens[idx_beginning + 1] == 'O':
        var_name = 'O'
        code_line = f"    {var_name} = {function_name}({', '.join(args)})"
    else:
        var_name = ''.join(tokens[idx_beginning+1:idx_var_name-1])
        code_line = f"    {var_name} = {function_name}({', '.join(args)})"
    if unrecognized_tokens:
        comment = ', '.join(unrecognized_tokens)
        code_line = code_line + f"   # there was an unrecognized token in the code: " + comment
    return code_line



def reconstruct_and_execute_code(tokens, path, name, path_to_mapping):
    """Executes Python code containing a solver function against a JSON task file.

    Args:
        code (str): The Python code to execute.
        path (str): Path to the JSON task file.
        name (str): Name of the solver function within the code (e.g., "solve_task1").

    Returns:
        dict: A dictionary containing the number of tasks solved correctly,
              and flags indicating various states of execution.
    """
    solvers_module = {}
    result = {
        'code': '',
        'accuracy': 0,
        'code_reconstructed': False,
        'code_initializable': False,
        'output_generated': False,
        'error_message': ''
    }
    try:
        code = reconstruct_code(tokens, name, path_to_mapping)
        result['code_reconstructed'] = True
        result['code'] = code
    except Exception as e:
        result['error_message'] += f'Error reconstructing code: {e}. '
        return result
    task = decode_json_task(path)

    try:
        exec(code, solvers_module)
        solver = solvers_module.get(f'solve_{name}')
        if not solver:
            result['error_message'] += f'Function solve_{name} not found in solvers_module. '
            return result
        result['code_initializable'] = True
    except Exception as e:
        result['error_message'] += f'Error initializing code: {e}. '
        return result

    try:
        for i, ex in enumerate(task, start=1):
            input_data = tuple(map(tuple, ex[0]))  # Assume first element is input
            output_data = tuple(map(tuple, ex[1]))  # Assume second element is output

            try:
                result['output_generated'] = True
                if solver(input_data) == output_data:
                    result['accuracy'] += 1
            except Exception as e:
                result['error_message'] += f'Error solving task {i}: {e}. '
                result['output_generated'] = False

    except Exception as e:
        result['error_message'] += f'Unexpected error during task execution: {e}. '

    result['accuracy'] = result['accuracy'] / len(task)
    return result


if __name__ == "__main__":
    # Apply the reformatting function
    file_content = read_solver_file('/Users/juliankleutgens/PycharmProjects/task2seq_T5/data_test/ct_schema')
    string_print, list_of_tokens = reformat_dsl_code(file_content,  extra_token = ['underscore', 'prev', 'sym_aft_func', 'var_to_num', 'BoF'])
    print(string_print)
    print(Counter(list_of_tokens))
    print(len(Counter(list_of_tokens)))
    #all_dsl_tokens = get_all_dsl_tokens()
    #print(all_dsl_tokens)







