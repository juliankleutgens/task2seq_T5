from T5mapping import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import os
from typing import List
from constants import Task
from utils import *
import inflect

def modify_variable_names(file_content):
    # Function to increment variable names
    def increment_var(match):
        var = match.group(0)
        num = int(var[1:])
        return f'x{num + 1}'

    # Replace variable names by incrementing their indices
    modified_content = re.sub(r'\bx\d+\b', increment_var, file_content)

    # Replace the last variable with 'O'
    lines = modified_content.split('\n')
    i = 0
    while i < len(lines):
        if 'return' in lines[i]:
            # replace the variable xN with O
            last_var = re.findall(r'\bx\d+\b', lines[i])[-1]
            lines[i] = re.sub(rf'\b{last_var}\b', 'O', lines[i])
            lines[i-1] = re.sub(rf'\b{last_var}\b', 'O', lines[i-1])
        i += 1
    # combine the lines
    modified_content = '\n'.join(lines)

    return modified_content




def _read_reverse_engineering_json_files(path: str, max_sampels: int) -> List[Task]:
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
    idx = path.find('generated_tasks')
    path_solver = path[:idx] + 'verifiers.py'
    solvers_file = read_solver_file(path_solver)
    # switch the word verify with solver in solvers_file
    solvers_file_buffer = solvers_file.replace('verify', 'solve')
    solvers_file = modify_variable_names(solvers_file_buffer)
    file_names = []
    paths = []
    i = 0
    for file in tqdm(files, desc="Decoding json files", leave=False):
        if i == max_sampels:
            break
        try:
            if not path[-5:] == '/test':
                task = decode_json_task(os.path.join(path_json, file))
                solver_name_for_task = file[:file.find('_')]
                function = 'def solve_' + solver_name_for_task + '('
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
            paths.append(os.path.join(path_json, file))
        i += 1

    return tasks, solvers, file_names, paths
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
    paths = []
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
                p = os.path.join(path_json, file)
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
            paths.append(p)
        i += 1

    return tasks, solvers, file_names, paths

def _read_generated_json_files(path: str, max_sampels:int) -> List[Task]:
    if path[:7] == "/Users/" or path[:6] == "/home/":
        path_json = path + '/tasks/'
    else:
        path_json = os.getcwd() + path + '/tasks/'
    print('we are at: ', path_json)
    files = sorted(os.listdir(path_json))

    tasks: List[Task] = []
    solvers = []
    paths = []
    #  read the solver.py file in a string
    solvers_file = read_solver_file(path)
    file_names = []
    i = 0
    for file in tqdm(files, desc="Decoding json files", leave=False):
        if i == max_sampels:
            break
        try:
            path_dir = os.path.join(path_json, file)
            task = decode_json_task(path_dir)
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
            paths.append(path_dir)
        i += 1

    return tasks, solvers, file_names, paths



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
            task_desc = convert2sparse_repeated_numbers(i)
            dataset_dict['input'].append(task_desc)
            dataset_dict['target'].append(T5_tokens_list)
            dataset_dict['name'].append(name)

    return dataset_dict

def convert_task(task, sparse_type='repeated2words'):
    task_all_pairs = ''
    whole_task_list = []
    for input_output_pair in task:
        if sparse_type == 'codeit':
            task_desc = codeit(input_output_pair)
            task_all_pairs += 'new ' + task_desc
        elif sparse_type == 'repeated2words':
            task_desc = convert2sparse_repeated_numbers(input_output_pair)
            task_all_pairs += task_desc
        elif sparse_type == 'None':
            task_desc = input_output_pair # this is the case when we want to keep the task as it is
            whole_task_list += task_desc
        else:
            print('decode the task with given sparse type not found in configuration file.')
            print('Please check the configuration file, for now using: repeated2words.')
            task_desc = convert2sparse_repeated_numbers(input_output_pair)
            task_all_pairs += task_desc
    if sparse_type == 'None':
        return whole_task_list
    return task_all_pairs

def load_data(path='ct_schema', maxsamples=-1, sparse_type='repeated2words'):
    """
    dataset_dict = {'input': [], 'target': [], 'name': []}
    # Load the T5 tokenizer
    if path.find('data_test') != -1 and maxsamples == -1:
        maxsamples = 500 # we have 3000 test samples
    if path.find('training_data') != -1:
        tasks, solvers, file_names, paths = _read_generated_json_files_saperatly(path=path, max_sampels=maxsamples)
    elif path.find('abstraction-and-reasoning-challenge') != -1:
        tasks, solvers, file_names, paths = _read_arc_json_files(path=path, max_sampels=maxsamples)
    elif path.find('reverse_engineering') != -1:
        tasks, solvers, file_names, paths = _read_reverse_engineering_json_files(path=path, max_sampels=maxsamples)
    else:
        tasks, solvers, file_names, paths = _read_generated_json_files(path=path, max_sampels=maxsamples)
    """

    # Extract dataset name using regular expression
    dataset_name_match = re.search(r'(training|abstraction-and-reasoning-challenge|reverse_engineering|data_test)',
                                   path)
    dataset_name = dataset_name_match.group() if dataset_name_match else None

    # Adjust maxsamples based on dataset
    maxsamples_test = -1 if maxsamples == -1 else maxsamples
    maxsamples_adjustments = {
        'data_test': maxsamples_test,
        'training_data': maxsamples,
        'abstraction-and-reasoning-challenge': maxsamples,
        'reverse_engineering': maxsamples
    }
    maxsamples = maxsamples_adjustments.get(dataset_name, maxsamples)

    # Function dispatching
    read_functions = {
        'training': _read_generated_json_files_separately,
        'abstraction-and-reasoning-challenge': _read_arc_json_files,
        'reverse_engineering': _read_reverse_engineering_json_files
    }
    read_function = read_functions.get(dataset_name, _read_generated_json_files)

    # Load data using the selected function
    tasks, solvers, file_names, paths = read_function(path, maxsamples)

    print(f"Read the data successfully from {path}")
    print(f"Number of tasks: {len(tasks)}")
    print("Continue with the preprocessing step")

    counter = 0
    dataset_dict = {'input': [], 'target': [], 'name': [], 'local_path': []}
    for task, solver, name, p in tqdm(zip(tasks, solvers, file_names, paths), desc="Pre Preprocessing step", leave=False):
        if check_validity(task):
            counter += 1
            continue
        trimmed_func = solver

        task_all_pairs = convert_task(task, sparse_type=sparse_type)
        dataset_dict['input'].append(task_all_pairs)
        dataset_dict['target'].append(trimmed_func)
        dataset_dict['name'].append(name)
        dataset_dict['local_path'].append(p)

    if counter > 0:
        print(f"Number of invalid tasks which are cut out of the dataset: {counter}")
    return dataset_dict


def check_validity(task):
    for i in task:
        input_ = np.array(i[0])
        output_ = np.array(i[1])

        if np.any(input_ > 9) or np.any(input_ < 0):
            return True
        if np.any(output_ > 9) or np.any(output_ < 0):
            return True
    return False


def _read_generated_json_files_separately(path: str, max_sampels:int) -> List[Task]:

    path_json = path + '/X/'
    path_solver = path + '/Y/'
    print('we are getting data from: ', path_json)
    files = sorted(os.listdir(path_json))

    tasks: List[Task] = []
    solvers = []
    #  read the solver.py file in a string
    paths = []
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
            paths.append(os.path.join(path_json, file_i))
        i += 1
    return tasks, solvers, file_names, paths


if __name__ == "__main__":
    extra_token = ['sym_aft_func', 'BoF', 'EoF', 'var_to_num']

    #dataset_dict = load_data(path='ct_schema', maxsamples=10, extra_token=extra_token)
    #print(dataset_dict)