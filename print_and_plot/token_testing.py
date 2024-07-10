import DSL.dsl
from tokenization.representation import *
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import T5Tokenizer
import json
from fuzzywuzzy import process
import time
import numpy as np
from utils import *
def get_two_letter_alphabet():
    # capital letters
    toknizer = T5Tokenizer.from_pretrained('t5-small')
    alphabet = []
    sentence_piece_char = '\u2581'
    #for i in range(1, 27):
     #   alphabet.append(chr(i + 96))
    for i in range(1, 27):
        alphabet.append(chr(64 + i))
    #for i in range(1, 27):
        #alphabet.append(sentence_piece_char + chr(i+64))
    len(alphabet)
    vocabulary = toknizer.get_vocab()
    vocabulary = set(list(vocabulary.keys()))
    for i in range(1, 27):
        for j in range(1, 27):
            if chr(64 + i) + chr(64 + j) in vocabulary and len(alphabet) < 100:
                alphabet.append(chr(64 + i) + chr(64 + j))

    return alphabet


dsl = get_all_dsl_tokens()
dsl = ['A','a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z',]
dsl = get_two_letter_alphabet()
#print(dsl)
#dsl = ['Black', 'Blue', 'Red', 'Green', 'Yellow', 'Gray', 'Purple', 'Orange', 'Azure', 'Brown', 'White']
#dsl = ['Input', 'Output', 'input', 'output']
#dsl = [str(i) for i in range(1000)]
# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define the DSL solver function as a string
dsl_solver = """
def solve_00d62c1b(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    O = fill(I, FOUR, x5)
    return O
"""
extra_token = ['sym_aft_func', 'BoF', 'EoF']
string_print, list_of_tokens = reformat_dsl_code(dsl_solver, extra_token=extra_token)
# Encode the DSL solver function
encoded_solver = tokenizer.encode(string_print)
tokens_solver = tokenizer.convert_ids_to_tokens(encoded_solver)

# Print the encoded function
print(string_print)
print(tokens_solver)



"""
for token in dsl:
    print()
    print(token)
    print(toknizer.encode(token))
    # get the id back to token
    print(toknizer.convert_ids_to_tokens(toknizer.encode(token)))
"""



