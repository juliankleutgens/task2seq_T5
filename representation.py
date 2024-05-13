import dsl
import inspect


# %load_ext autoreload
# %autoreload 2
# full_demo()
import json
import re
from pathlib import Path
from collections import Counter

def read_solver_file():
    # Get the current working directory
    current_dir = Path.cwd()
    # Construct the path to the file within the current directory
    file_path = current_dir / 'ct_schema' / 'solvers.py'
    with open(file_path, 'r') as file:
        data = file.read()
    return data


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

            # ----------- Handle the case when 'prev' is in extra_token ------------
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
            line = "#BoF"
            previous = [] # the previous_vars are reset because we are in a new function
            if extra_token is not None and 'BoF' in extra_token:
                new_code.append(line) # Function definition line

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

def embedding_one_hot(tokens):
    # Get all unique tokens
    unique_tokens = set(tokens)
    # Create a dictionary to hold the one-hot encodings
    one_hot_dict = {token: [0] * len(unique_tokens) for token in unique_tokens}
    # Assign a unique index to each token
    token_to_index = {token: i for i, token in enumerate(unique_tokens)}
    # Generate the one-hot encodings
    for token, one_hot in one_hot_dict.items():
        one_hot[token_to_index[token]] = 1
    return one_hot_dict

if __name__ == "__main__":
    # Apply the reformatting function
    file_content = read_solver_file()
    string_print, list_of_tokens = reformat_dsl_code(file_content,  extra_token=['sym_aft_func', 'var_to_num', 'BoF', 'EoF'])
    #print(string_print)
    dsl_tokens = get_all_dsl_tokens()
    tokens_all = dsl_tokens  #+ ['UNK']
    one_hot_dict = embedding_one_hot(tokens_all)
    # save the one_hot_dict to a json file
    with open('one_hot_dict_dsl_func.json', 'w') as file:
        json.dump(one_hot_dict, file)
    print(Counter(list_of_tokens))
    print(len(Counter(list_of_tokens)))



    #all_dsl_tokens = get_all_dsl_tokens()
    #print(all_dsl_tokens)







