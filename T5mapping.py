import dsl
from representation import *
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import T5Tokenizer
import json
from fuzzywuzzy import process
import numpy as np

def map_to_t5_token(string_solver,extra_token = ['sym_aft_func', 'BoF', 'EoF'], tokenizer=T5Tokenizer.from_pretrained('t5-small'), loading_new_mappings = True):
    string_print, list_of_tokens = reformat_dsl_code(string_solver, extra_token=extra_token)
    dsl_token_mappings = load_token_mappings(filename="dsl_token_mappings_T5.json")
    if list_of_tokens is None:
        error_message = "The list of tokens is empty. Please check the reformat_dsl_code function."
        return error_message

    if not (dsl_token_mappings is not None and not loading_new_mappings):
        # Load the T5 tokenizer
        all_tokens = list(tokenizer.get_vocab().keys())
        # List of custom tokens
        # all tokens from the dsl.py file + the tokens from the solver.py file
        dsl_func = get_all_dsl_tokens()
        tokenstoMap = list(set(list_of_tokens)) + dsl_func

        if 'sym_aft_func' in extra_token:
            tokenstoMap.append(';')
        if 'BoF' in extra_token:
            tokenstoMap.append('#BoF')
        if 'EoF' in extra_token:
            tokenstoMap.append('#EoF')
        tokenstoMap.append('#newline')

        tokenstoMap = list(set(tokenstoMap))
        #all_tokens_minus_used_solver = list(set(all_tokens) - set(string_tokens_to_t5))
        dsl_token_mappings = get_mapping(custom_tokens=tokenstoMap, T5_tokens=all_tokens)
        save_token_mappings(dsl_token_mappings, filename="dsl_token_mappings_T5.json")



    processed_text = preprocess_text(string_print, dsl_token_mappings)
    list_of_tokens_T5 = map_list(list_of_tokens, dsl_token_mappings)
    T5_tokens_list = tokenizer.tokenize(processed_text)
    if loading_new_mappings:
        print("--------------------- Example DSL ---------------------")
        print('---- DSL with our Tokens ----')
        idx_BoF = string_print.index('#BoF')
        idx_EoF = string_print.index('#EoF')
        print(string_print[idx_BoF:idx_EoF+4])

        print('---- DSL List Tokens ----')
        idx_BoF = list_of_tokens.index('#BoF')
        idx_EoF = list_of_tokens.index('#EoF')
        print(list_of_tokens[idx_BoF:idx_EoF + 1])


        print("------ T5 list -------")
        idx_BoF = list_of_tokens_T5.index(dsl_token_mappings['#BoF'])
        idx_EoF = list_of_tokens_T5.index(dsl_token_mappings['#EoF'])
        print(list_of_tokens_T5[idx_BoF:idx_EoF+1])

    return list_of_tokens_T5, dsl_token_mappings

def map_list(list_of_tokens, dsl_token_mappings):
    T5_tokens_list = []
    for token in list_of_tokens:
        if token in dsl_token_mappings:
            T5_tokens_list.append(dsl_token_mappings[token])
        else:
            raise ValueError(f"Input list contains a token that is not in the mapping: {token}, set loading_new_mappings to True to generate a new mapping.")
    return T5_tokens_list
def save_token_mappings(token_mappings, filename="dsl_token_mappings_T5.json"):
    """Saves token mappings to a JSON file, handling potential overwrite."""
    # Save mappings only if they've changed or the file doesn't exist
    # always save the mappings
    try:
        with open(filename, "w") as f:
            json.dump(token_mappings, f, indent=4)
            print(f"Token mappings saved to {filename}.")
    except TypeError as e:
        print("Failed to serialize token_mappings due to a TypeError:", e)
    except Exception as e:
        print("An error occurred:", e)


def load_token_mappings(filename="dsl_token_mappings_T5.json"):
    """Loads token mappings from a JSON file, handling potential errors."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing token mappings found.")
        return None


def get_mapping(custom_tokens, T5_tokens):
    """Generates mappings of custom tokens to T5 vocabulary tokens, ensuring unique mappings and excluding '</s>'."""


    token_mappings = {}  # Initialize an empty dictionary for token mappings
    used_matches = set()  # Initialize an empty set to track used T5 tokens
    all_tokens = set(T5_tokens) - {'<pad>', '</s>', '<unk>'}  # Exclude '</s>' from potential matches
    i = 0
    for token in custom_tokens:
        i += 1
        # Find the best match that is not already used
        best_match, score = None, None
        potential_matches = process.extract(token, all_tokens, limit=10)  # Get top 10 matches to find an unused best match
        for match, match_score in potential_matches:
            if match not in used_matches:
                best_match, score = match, match_score
            else:
                print(f"{i}: Match {match} is already used. Skipping.")
                continue

        if best_match:
            token_mappings[token] = best_match
            used_matches.add(best_match)  # Mark this match as used
            print(f"{i}: Best match for {token} is {best_match} with a score of {score}")
        else:
            print(f"{i}: No available unique match for {token}.")
            # then choose one randomly from the all tokens
            best_match = list(all_tokens)[np.random.randint(0, len(all_tokens))]
            token_mappings[token] = best_match
            used_matches.add(best_match)
        all_tokens.remove(best_match)  # Remove the match from potential matches

    save_token_mappings(token_mappings)
    return token_mappings

def save_new_mapping_from_df(dfs, extra_token = ['sym_aft_func', 'BoF', 'EoF', 'var_to_num'], tokenizer=T5Tokenizer.from_pretrained('t5-small')):
    string_solver = ''
    task_tokens = []
    print('Computing a new mappings')
    print('First, concatenate all the solvers and tokenize them')

    for df in tqdm(dfs, desc="DataFrames"):
        for solver, task, name in tqdm(zip(df['target'], df['input'], df['name']), desc="Tasks", leave=False):
            # concatenate the solver
            string_solver = string_solver + solver + '\n'
            task_input = tokenizer(task, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
            task_input = tokenizer.convert_ids_to_tokens(task_input['input_ids'][0])
            task_tokens = task_tokens + task_input
    string_print, list_of_tokens = reformat_dsl_code(string_solver, extra_token=extra_token)
    all_tokens = list(tokenizer.get_vocab().keys())

    # ------------------- filter the T5 tokens I don't want to be mapped to -------------------
    print('There are ', len(all_tokens), ' tokens in the T5 vocabulary, time to filter them')
    # I want to get rid of all the tokens with an underscore before the word/sentencepiece
    sentence_piece_char = '\u2581'
    # Filter out tokens that start with the SentencePiece character
    task_tokens_set = set(task_tokens)  # Convert to set for faster lookup
    filtered_tokens = [
        token for token in all_tokens
        if not token.startswith(sentence_piece_char) and
           token not in map(str, range(50)) and
           token not in task_tokens_set
    ]
    print('There are ', len(filtered_tokens), ' tokens left after filtering')


    # ------------------ make a list of all tokens I need to map to the T5 tokens -------------------
    dsl_func = get_all_dsl_tokens()
    tokenstoMap = list(set(list_of_tokens)) + dsl_func
    if 'sym_aft_func' in extra_token:
        tokenstoMap.append(';')
    if 'BoF' in extra_token:
        tokenstoMap.append('#BoF')
    if 'EoF' in extra_token:
        tokenstoMap.append('#EoF')
    tokenstoMap.append('#newline')
    tokenstoMap = tokenstoMap + ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'UNITY', 'ORIGIN', 'TWO_BY_TWO', 'NEG_ONE', 'UP', 'DOWN_LEFT', 'DOWN_RIGHT', 'DOWN', 'LEFT', 'RIGHT', 'UP_LEFT', 'UP_RIGHT', 'DOWN_LEFT_ONE', 'DOWN_RIGHT_ONE', 'DOWN_ONE', 'LEFT_ONE', 'RIGHT_ONE']
    # list of numbers from 0 to 99
    tokenstoMap = tokenstoMap + [str(i) for i in range(10)]
    tokenstoMap = list(set(tokenstoMap))
    print('There are ', len(tokenstoMap), ' tokens to map')
    # all_tokens_minus_used_solver = list(set(all_tokens) - set(string_tokens_to_t5))
    dsl_token_mappings = get_mapping(custom_tokens=tokenstoMap, T5_tokens=filtered_tokens)
    save_token_mappings(dsl_token_mappings, filename="dsl_token_mappings_T5.json")
    # check if it worked by loading the mapping
    dsl_token_mappings = load_token_mappings(filename="dsl_token_mappings_T5.json")
    if len(dsl_token_mappings) == len(tokenstoMap):
        print('The mapping was successful')
    else:
        print('The mapping was not successful')



# Example usage:
# custom_tokens = ['exampleToken1', 'exampleToken2']
# all_tokens = ['sampleToken1', 'sampleToken2', 'exampleToken1']
# get_mapping(custom_tokens, all_tokens)
def preprocess_text(text, token_mappings):
    # Split the text into words (or use your own tokenizer)
    words = text.split()
    # Replace custom tokens with their mapped values
    processed_words = [token_mappings.get(word, word) for word in words]
    return ' '.join(processed_words)


def reconstruct_dsl_code(reformatted_code):
    original_code = []
    for line in reformatted_code:
        if line.startswith('#newline'):
            # Remove the line marker
            parts = line.replace('#newline ', '').split(' ')
            var_name = parts[0]
            expression = ' '.join(parts[1:])

            # Undo specific token modifications (as examples)
            expression = expression.replace(';', '')  # Removing semicolons
            expression = expression.lstrip('_')  # Removing leading underscores
            # More undo operations based on your reformatted structure

            original_code.append(f'{var_name} = {expression}')
        elif line == '#BoF':
            original_code.append('def function():')  # Adjust to actual function syntax
        elif line == '#EoF':
            original_code.append('return something')  # Adjust to actual return statement
        # Continue for other markers and cases

    return '\n'.join(original_code)

def map_back(list_of_tokens):
    T5_map = load_token_mappings(filename="dsl_token_mappings_T5.json")
    # use the dictionary to map back the tokens
    # switch the key and values
    T5_map = {v: k for k, v in T5_map.items()}
    original_tokens = []
    for token in list_of_tokens:
        if token in T5_map:
            original_tokens.append(T5_map[token])
        else:
            original_tokens.append(token)

    return original_tokens


def reconstruct_code(token_list, name_idx='005t822n'):
    tokens = token_list
    output = []
    current_function = []
    current_variable_index = 1
    function_count = 1
    in_function = False
    # search for #newline from the back of the list
    last_idx_newline = len(tokens) - tokens[::-1].index('#newline') - 1
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == '#BoF':

            in_function = True
            function_header = f"\n\ndef solve_{name_idx}(\n    I: Grid\n) -> Grid:"
            current_function.append(function_header)

            function_count += 1
            i += 1  # go to next token after BoF

        elif token == '#newline':
            current_function.append('\n')  # Add newline

            if i == last_idx_newline:
                idx_end_line = len(tokens)
            else:
                idx_end_line = tokens.index('#newline', i + 1)
                idx_of_end_func = tokens.index('#EoF', i + 1)
                if idx_end_line > idx_of_end_func:
                    idx_end_line = idx_of_end_func
            idx_of_break = tokens.index(';', i+1, idx_end_line)
            function_name = tokens[idx_of_break-1]

            arguments = tokens[idx_of_break+1:idx_end_line]
            args = []

            # get all the arguments which are the inputs to the function

            for1 = 0
            while for1 < len(arguments):

                if arguments[for1] == 'x':
                    if arguments[for1+1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        args.extend([''.join(arguments[for1:for1+2])])
                        for1 += 2
                    elif arguments[for1+2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        args.extend([''.join(arguments[for1:for1+3])])
                        for1 += 3
                    elif arguments[for1+3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        args.extend([''.join(arguments[for1:for1+4])])
                        for1 += 4
                else:
                    args.extend([arguments[for1]])
                    for1 += 1

            if tokens[idx_of_break-2] == 'O':
                var_name = 'O'
                current_function.append(f"    {var_name} = {function_name}({', '.join(args)})")
            else:
                var_name = ''.join(tokens[i+1:idx_of_break-1])
                current_function.append(f"    {var_name} = {function_name}({', '.join(args)})")

            i = idx_end_line  # Move index to next relevant token

        elif token == '#EoF':
            current_function.append("\n    return O")
            output.append(''.join(current_function))
            current_function = []
            in_function = False
            i += 1  # Move to next token

        else:
            i += 1  # Skip unrecognized tokens or handle other cases if needed

    if current_function:
        output.append(''.join(current_function))

    # Return a single string that concatenates all function definitions
    return ''.join(output)




if __name__ == "__main__":
    # get the file content and the solver.py file
    file_content = read_solver_file()

    #all_functions_string = "\n".join(all_rewritten_functions.values())
    #extra_token = ['BoF','prev','sym_aft_func','var_to_num']
    extra_token = ['sym_aft_func', 'BoF', 'EoF', 'var_to_num']
    string_print, list_of_tokens = reformat_dsl_code(file_content,  extra_token=extra_token)


    # Load the T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    T5_tokens_text, dsl_token_mappings = map_to_t5_token(file_content, extra_token=extra_token, tokenizer=tokenizer, loading_new_mappings=False)
    original_tokens = map_back(T5_tokens_text)
    print("---------- Original Tokens ----------")
    print(original_tokens[:200])
    recon_code = reconstruct_code(original_tokens, name_idx='005t822n')

    print(file_content[:2000])
    print("---------- Reconstructed Code ----------")
    print(recon_code[:2000])
    print()
