import dsl
from representation import *
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import T5Tokenizer
import json
from rapidfuzz import process
import time
import numpy as np
from utils import *
from Levenshtein import distance as levenshtein_distance

def map_to_t5_token(string_solver,extra_token = ['sym_aft_func', 'BoF', 'EoF'], tokenizer=T5Tokenizer.from_pretrained('t5-small'), loading_new_mappings = True):
    string_print, list_of_tokens = reformat_dsl_code(string_solver, extra_token=extra_token)
    dsl_token_mappings = load_token_mappings(filename="dsl_token_mappings_T5.json")
    if list_of_tokens is None:
        error_message = "The list of tokens is empty. Please check the reformat_dsl_code function."
        return error_message
    list_of_tokens_T5 = map_list(list_of_tokens, dsl_token_mappings)

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


def get_best_match_levenstein(token, all_tokens, used_matches):
    distances = [(other, levenshtein_distance(token, other)) for other in all_tokens]
    distances.sort(key=lambda x: x[1])

    for match, dist in distances:
        if match not in used_matches:
            return match, dist
    return None, None


def get_mapping(custom_tokens, T5_tokens, extra_token , type_of_mapping, tokenizer):
    """Generates mappings of custom tokens to T5 vocabulary tokens, ensuring unique mappings and excluding '</s>'."""


    token_mappings = {}  # Initialize an empty dictionary for token mappings
    used_matches = set()  # Initialize an empty set to track used T5 tokens
    all_tokens = set(T5_tokens) - {'<pad>', '</s>', '<unk>'}
    xnum2alp = alphabet_mapping(all_tokens)# Exclude '</s>' from potential matches
    if type_of_mapping == 'val2alphabet':
        used_matches = set(xnum2alp.values())
    elif type_of_mapping == 'val2num':
        used_matches = set([str(i) for i in range(100)])

    i = 0
    tokens_with_nomatch = []
    for token in custom_tokens:
        i += 1
        if token == '#EoF':
            token_mappings[token] = '</s>' # Map '#EoF' to '</s>'
            print(f"{i}: The {token} must be mapped to </s>")
            continue

        if token == 'x' and type_of_mapping == 'x2y':
            token_mappings[token] = 'y'
            print(f"{i}: {token} must be mapped to y")
            continue
        used_matches.add('y')

        if token == 'T' or token == 'F':
            if token == 'T':
                new_token = function_is_token('True', all_tokens)
            elif token == 'F':
                new_token = function_is_token('False', all_tokens)
            if new_token is not None:
                token_mappings[token] = new_token
                used_matches.add(new_token)
                print(f"{i}: {token} is mapped to {new_token}")
                continue

        if token.startswith('x') and type_of_mapping == 'val2num':
            token_mappings[token] = str(int(token[1:]))
            used_matches.add(str(int(token[1:])))
            print(f"{i}: {token} is mapped to {str(int(token[1:]))}")
            continue

        if token.startswith('x') and type_of_mapping == 'val2alphabet':
            token_mappings[token] = xnum2alp[token]
            used_matches.add(xnum2alp[token])
            print(f"{i}: {token} is mapped to {token_mappings[token]}")
            continue

        new_token = function_is_token(token, all_tokens)
        if new_token is not None:
            if not new_token in used_matches:
                token_mappings[token] = new_token
                used_matches.add(new_token)
                print(f"{i}: Perfect Match: {token} is mapped to {new_token}")
                continue
            else:
                print(f"{i}: Want to map {token} to {new_token}, but it is already used. Must choose differently.")
        tokens_with_nomatch.append(token)
        i = i - 1

    for token in tokens_with_nomatch:
        i += 1
        # could not find a perfect match so use Levenshtein distance to find
        # the next similar match in the T5 tokens
        best_match, score = get_best_match_levenstein(token, all_tokens, used_matches)
        if best_match:
            token_mappings[token] = best_match
            used_matches.add(best_match)
            print(f"{i}: Best match for {token} is {best_match} with a score of {score}")
        else:
            print(f"{i}: No available unique match for {token}.")
            best_match = list(all_tokens)[np.random.randint(0, len(all_tokens))]
            token_mappings[token] = best_match
            print(f"{i}: Randomly selected match for {token} is {best_match}.")
            used_matches.add(best_match)
        all_tokens.remove(best_match)  # Remove the match from potential matches

    save_token_mappings(token_mappings)
    return token_mappings

def alphabet_mapping(model_tokens):
    # capital letters
    alphabet = []

    # small letters
    # for i in range(1, 27):
    #   alphabet.append(chr(i + 96))

    # capital letters with sentence piece
    # sentence_piece_char = '\u2581'
    # for i in range(1, 27):
    # alphabet.append(sentence_piece_char + chr(i+64))

    # capital letters
    for i in range(1, 27):
        alphabet.append(chr(64 + i))
    len(alphabet)
    for i in range(1, 27):
        for j in range(1, 27):
            character = chr(64 + i) + chr(64 + j)
            if character in model_tokens and character not in alphabet:
                alphabet.append(character)
    dic = {}
    for i in range(1,300):
        val = 'x'+str(i)
        dic[val] = alphabet[i-1]
    return dic


def function_is_token(token,model_tokens):
    sentence_piece_char = '\u2581'
    token0 = token
    token1 = sentence_piece_char + token
    # make the word lower case
    token2 = token.lower()
    token3 = sentence_piece_char + token2
    # make only the first letter capital
    token4 = token2.capitalize()
    token5 = sentence_piece_char + token4
    # make all the letters capital
    token6 = token2.upper()
    token7 = sentence_piece_char + token6
    # look if it is in model tokens
    check_tokens = [token0, token1, token2, token3, token4, token5, token6, token7]
    for token in check_tokens:
        if token in model_tokens:
            return token
    return None



def save_new_mapping_from_df(dfs, extra_token = ['sym_aft_func', 'BoF', 'EoF', 'var_to_num'], tokenizer=T5Tokenizer.from_pretrained('t5-small'), type_of_mapping='x2y'):
    string_solver = ''
    task_tokens = []
    print('Computing a new mappings')
    print('First, concatenate all the solvers and tokenize them')
    # debug stuff
    if (type_of_mapping == 'val2num' or type_of_mapping == 'val2alphabet') and 'var_to_num' in extra_token:
        raise ValueError("You can't use val2num or val2alphabet as type of mapping and var_to_num as extra tokens: us extra_token = ['sym_aft_func', 'EoF'] instead")
    if type_of_mapping == 'x2y' and not 'var_to_num' in extra_token:
        raise ValueError("You must use var_to_num as extra token when using x2y as type of mapping")
    # Initialize the total time counters
    # Calculate the total number of tasks to be processed
    total_tasks = sum(len(df) for df in dfs)

    # this is just for getting some strings solver.py in theory should be one enough and add the tokens we have there to be mapped to the T5
    j = 0
    with tqdm(total=total_tasks, desc="Getting string of Solver.py") as pbar:
        for df in dfs:
            for solver, task, name in zip(df['target'], df['input'], df['name']):
                if j < 10000:
                    string_solver = string_solver + solver + '\n'
                    # Update the progress bar
                    pbar.update(1)
                    j += 1
    start_time = time.time()
    string_print, list_of_tokens = reformat_dsl_code(string_solver, extra_token=extra_token)
    time_to_reformat = time.time() - start_time
    print('Time to reformat the DSL code: ', time_to_reformat)

    all_tokens = list(tokenizer.get_vocab().keys())
    task_tokens = get_tokens_from_task_encoder()

    # ------------------- filter the T5 tokens I don't want to be mapped to -------------------
    print('There are ', len(all_tokens), ' tokens in the T5 vocabulary, time to filter them')
    # I want to get rid of all the tokens with an underscore before the word/sentencepiece
    sentence_piece_char = '\u2581'
    # Filter out tokens that start with the SentencePiece character
    task_tokens_set = set(task_tokens)  # Convert to set for faster lookup
    if type_of_mapping != 'val2num':
        filtered_tokens = [
            token for token in all_tokens
            if token not in map(str, range(50)) and
               token not in task_tokens_set
        ]
    else:
        # not token.startswith(sentence_piece_char) and
        filtered_tokens = [
            token for token in all_tokens
            if token not in task_tokens_set
        ]
    print('There are ', len(filtered_tokens), ' tokens left after filtering')


    # ------------------ make a list of all tokens I need to map to the T5 tokens -------------------
    dsl_func = get_all_dsl_tokens()
    dsl_constants = get_all_dsl_constants()
    tokenstoMap = list(set(list_of_tokens)) + dsl_func + dsl_constants
    tokenstoMap.append('#newline')
    if 'sym_aft_func' in extra_token:
        tokenstoMap.append(';')
    if 'BoF' in extra_token:
        tokenstoMap.append('#BoF')
    if 'EoF' in extra_token:
        tokenstoMap.append('#EoF')
    if 'var_to_num' in extra_token:
        tokenstoMap = tokenstoMap + [str(i) for i in range(10)] + ['x']
    else:
        tokenstoMap = tokenstoMap + ['x' + str(i) for i in range(1,300)]

    tokenstoMap = list(set(tokenstoMap))
    print('There are ', len(tokenstoMap), ' tokens to map')

    dsl_token_mappings = get_mapping(custom_tokens=tokenstoMap, T5_tokens=filtered_tokens,extra_token=extra_token,
                                     type_of_mapping=type_of_mapping, tokenizer=tokenizer)
    save_token_mappings(dsl_token_mappings, filename="dsl_token_mappings_T5.json")

    # check if it worked by loading the mapping
    dsl_token_mappings = load_token_mappings(filename="dsl_token_mappings_T5.json")
    if len(dsl_token_mappings) == len(tokenstoMap):
        print('The mapping was successful')
    else:
        print('The mapping was not successful')


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




if __name__ == "__main__":
    # get the file content and the solver.py file
    file_content = read_solver_file('/Users/juliankleutgens/PycharmProjects/task2seq_T5/data_test/ct_schema')

    #all_functions_string = "\n".join(all_rewritten_functions.values())
    #extra_token = ['BoF','prev','sym_aft_func','var_to_num']
    extra_token = ['sym_aft_func', 'EoF']
    string_print, list_of_tokens = reformat_dsl_code(file_content,  extra_token=extra_token)


    # Load the T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    T5_tokens_text, dsl_token_mappings = map_to_t5_token(file_content, extra_token=extra_token, tokenizer=tokenizer, loading_new_mappings=False)
    original_tokens = map_back(T5_tokens_text)
    #  get one function, find #EoF
    example_function = original_tokens[0:original_tokens.index('#EoF')+4]
    # add noise to the function
    random = np.random.randint(0, len(example_function))

    # add the noise token but do not change the tokens add it only in the sequence
    example_function.insert(random, '!NOISE!')


    print("---------- Original Tokens ----------")
    print(example_function)
    recon_code = reconstruct_code(example_function, name_idx='005t822n')

    print("---------- Reconstructed Code ----------")
    print(recon_code)
    print()
