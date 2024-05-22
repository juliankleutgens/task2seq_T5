from T5mapping import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import os
from typing import List
from constants import Task
from utils import *
import inflect
def _read_arc_json_files(path: str, max_sampels: int) -> List[Task]:
    """
    Given a directory path returns a list of arc
    Tasks.
    """
    if path[:7] == "/Users/" or path[:6] == "/home/":
        path_json = path
    else:
        path_json = os.getcwd() + path
    print('we are at: ', path_json)
    files = sorted(os.listdir(path_json))
    tasks: List[Task] = []
    solvers = []
    #  read the solver.py file in a string
    if not path[-5] == '/test':
        idx = path.find('abstraction-and-reasoning-challenge')
        solvers_file = read_solver_file(path[:idx])

    file_names = []
    i = 0
    for file in tqdm(files, desc="Decoding json files", leave=False):
        if i == max_sampels:
            break
        try:
            if not path[-5:] == '/test':
                task = decode_json_task(os.path.join(path_json, file))
                function = 'def solve_' + file[:-5] + '('
                idx_start = solvers_file.find(function)
                idx_end = solvers_file.find('def solve_', idx_start + 1)
                solver = solvers_file[idx_start:idx_end]
            else:
                task = decode_json_task_test(os.path.join(path_json, file))
                solver = ''

        except Exception as e:
            print(e)
        else:
            tasks.append(task)
            solvers.append(solver)
            file_names.append(file[:-5])
        i += 1

    return tasks, solvers, file_names

def _read_generated_json_files(path: str, max_sampels:int) -> List[Task]:
    if path[:7] == "/Users/" or path[:6] == "/home/":
        path_json = path + '/tasks/'
    else:
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
            file_names.append(file[:-5])
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

def convert_task(task, sparse_type='repeated2words'):
    task_all_pairs = ''
    for i in task:
        if sparse_type == 'codeit':
            task_all_pairs += ' new pair'
            task_desc = convert2sparse(i)
        elif sparse_type == 'repeated2words':
            task_all_pairs += ' new pair'
            task_desc = convert2sparse_repeated_numbers(i)
        else:
            print('decode the task with given sparse type not found in configuration file.')
            print('Please check the configuration file, for now using: repeated2words.')
            task_desc = convert2sparse_repeated_numbers(i)
        task_all_pairs += task_desc
    return task_all_pairs

def load_data(path='ct_schema', maxsamples=-1, sparse_type='repeated2words'):
    dataset_dict = {'input': [], 'target': [], 'name': []}
    # Load the T5 tokenizer
    if path.find('data_test') != -1:
        maxsamples = 3000 # we have 3000 test samples
    if path.find('training_data') != -1:
        tasks, solvers, file_names = _read_generated_json_files_saperatly(path=path, max_sampels=maxsamples)
    elif path.find('abstraction-and-reasoning-challenge') != -1:
        tasks, solvers, file_names = _read_arc_json_files(path=path, max_sampels=maxsamples)
    else:
        tasks, solvers, file_names = _read_generated_json_files(path=path, max_sampels=maxsamples)

    print(f"Read the data successfully from {path}")
    print(f"Number of tasks: {len(tasks)}")
    print("Continue with the preprocessing step")

    counter = 0
    for task, solver, name in tqdm(zip(tasks, solvers, file_names), desc="Pre Preprocessing step", leave=False):
        trimmed_func = solver
        task_all_pairs = convert_task(task, sparse_type=sparse_type)
        dataset_dict['input'].append(task_all_pairs)
        dataset_dict['target'].append(trimmed_func)
        dataset_dict['name'].append(name)

    return dataset_dict

def _read_generated_json_files_saperatly(path: str, max_sampels:int) -> List[Task]:

    path_json = path + '/X/'
    path_solver = path + '/Y/'
    print('we are getting data from: ', path_json)
    files = sorted(os.listdir(path_json))

    tasks: List[Task] = []
    solvers = []
    #  read the solver.py file in a string

    file_names = []
    i = 0
    current_dir = Path.cwd()
    # Construct the path to the file within the current directory
    for file_i in tqdm(files, desc="Decoding json files", leave=False):
        if i == max_sampels:
            break
        try:
            task = decode_json_task(os.path.join(path_json, file_i))
            file_path = path_solver + file_i[:-5] + '.py'
            with open(file_path, 'r') as file:
                data = file.read()
            solver = data

        except Exception as e:
            print(e)
        else:
            tasks.append(task)
            solvers.append(solver)
            file_names.append(file_i[:-5])
        i += 1
    return tasks, solvers, file_names


if __name__ == "__main__":
    extra_token = ['sym_aft_func', 'BoF', 'EoF', 'var_to_num']

    dataset_dict = load_data(path='ct_schema', maxsamples=10, extra_token=extra_token)
    print(dataset_dict)