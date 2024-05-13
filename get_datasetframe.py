from T5mapping import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import os
from typing import List
from constants import Task
from utils import *

def _read_generated_json_files(path: str, max_sampels:int) -> List[Task]:
    path_json = os.getcwd() + path + '/tasks/'
    print('we are at: ', path_json)
    files = sorted(os.listdir(path_json))

    tasks: List[Task] = []
    solvers = []
    #  read the solver.py file in a string
    solvers_file = read_solver_file(path)
    file_names = []
    i = 0
    for file in tqdm(files, desc="Decoding json files", leave=False):
        if i == max_sampels:
            break
        try:
            task = decode_json_task(os.path.join(path_json, file))
            function = 'def solve_' + file[:-5] + '('
            idx_start = solvers_file.find(function)
            idx_end = solvers_file.find('def solve_', idx_start + 1)
            solver = solvers_file[idx_start:idx_end]

        except Exception as e:
            print(e)
        else:
            tasks.append(task)
            solvers.append(solver)
            file_names.append(file)
        i += 1

    return tasks, solvers, file_names

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

def load_data_with_T5_tokens(path='ct_schema', maxsamples=None, extra_token=['sym_aft_func', 'BoF', 'EoF', 'var_to_num'],
              tokenizer = T5Tokenizer.from_pretrained('t5-small')):
    dataset_dict = {'input': [], 'target': [], 'name': []}
    # Load the T5 tokenizer
    tasks, solvers, file_names = _read_generated_json_files(path=path, max_sampels=maxsamples)


    counter = 0
    for task, solver, name in zip(tasks, solvers, file_names):
        trimmed_func = solver
        T5_tokens_list, _ = map_to_t5_token(solver, extra_token=extra_token, tokenizer=tokenizer,
                                                             loading_new_mappings=False)

        for i in task:
            task_desc = convert2sparse(i)
            dataset_dict['input'].append(task_desc)
            dataset_dict['target'].append(T5_tokens_list)
            dataset_dict['name'].append(name)

    return dataset_dict

def load_data(path='ct_schema', maxsamples=None):
    dataset_dict = {'input': [], 'target': [], 'name': []}
    # Load the T5 tokenizer
    tasks, solvers, file_names = _read_generated_json_files(path=path, max_sampels=maxsamples)

    counter = 0
    for task, solver, name in zip(tasks, solvers, file_names):
        trimmed_func = solver
        task_all_pairs = ''
        for i in task:
            task_desc = convert2sparse(i)
            task_all_pairs += task_desc + '\n'
        dataset_dict['input'].append(task_all_pairs)
        dataset_dict['target'].append(trimmed_func)
        dataset_dict['name'].append(name)

    return dataset_dict

if __name__ == "__main__":
    extra_token = ['sym_aft_func', 'BoF', 'EoF', 'var_to_num']

    dataset_dict = load_data(path='ct_schema', maxsamples=10, extra_token=extra_token)
    print(dataset_dict)