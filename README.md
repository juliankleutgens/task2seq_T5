## Abstract:

Humans can generalize sparse experiences into principles and apply them in unfamiliar contexts. In contrast, artificial agents often struggle in novel situations, even with large amounts of data. François Chollet argues this happens because artificial agents are trained to be skillful rather than genuinely intelligent. To address this and steer research toward true AI, he introduced the Abstraction and Reasoning Corpus (ARC), a benchmark designed to measure general intelligence. The ARC dataset evaluates a system’s ability to deduce rules from limited input-output pairs and apply them to new, unseen data.
In this repository is a method using a Domain Specific Language (DSL) to represent solutions for ARC tasks. This involves training Large Language Models (LLMs) to generate DSL programs based on ARC dataset input-output pairs. 
This approach is using the T5 model, a pre-trained LLM, to generate the DSL solvers.
Results show the T5 model can learn to generate correct and generalizable DSL solvers, though challenges remain in achieving high accuracy and consistent output generation.

## Setup the Environment for this Repo:
**Prerequisites**

- Python 3.6 or higher
- Conda

**Step-by-Step Instructions**

1. **Create a Conda Environment:**

   Open a terminal and create a new Conda environment with Python 3.8:

   ```bash
   conda create --name task2seq python=3.8
   ```

2. **Activate the Conda Environment:**

   ```bash
   conda activate task2seq
   ```

3. **Install PyTorch according to the server:**

   Install PyTorch, Torchvision, and Torchaudio with CUDA support:

   This is for the GPU server "grewegpud1.ethz.ch" at ETH, where we are using the GPU Quadro RTX 6000 with CUDA Version: 11.7
   
   ```bash
   pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```

   For any other case (like your local computer) use the following line:
   ```bash
   pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
   ```


   
5. **Install Related Libraries:**
   ```bash
   pip install omegaconf torchsummary numpy pandas transformers rich wandb fuzzywuzzy sentencepiece matplotlib rouge nltk rapidfuzz inflect Levenshtein
   ```

## Configuration
In the folder "configuration," there are two files: `config.yaml` (used for running on the GPU server with different data addresses) and `config_test.yaml` (used for running the code on my local computer with a CPU). This distinction is made to save time when pushing/pulling and debugging. To use `config_test.yaml`, set `test_mode: true` in `config.yaml`.

## Data Structure
The data should be structured in two ways:

1. Save each task as a separate JSON file with the ID in the file name (e.g., `y65h51dw.json`). Then, save all the solvers in one large `solvers.py` file, where you can find the `def solver_y65h51dw` function.
2. Save both tasks and solvers separately. So, store the task in the `/X/` folder, such as `0k99no11.json`, and put the corresponding solver function file into `/Y/` folder like `0k99no11.py`. For the script to recognize this structure, include "training" in the folder name. If you don't want to label it with "training," update the method name in the load_data function in data_scripts/get_datasetframe.py.

```
data/
├── arc_tasks/
│   ├── tasks/
│   │   ├── y65h51dw.json
│   │   ├── ... 
│   ├── solvers.py
├── arc_tasks_2/
├── arc_tasks_test/
├── training_arc_task/
│   ├── X/
│   │   ├── 0k99no11.json
│   │   ├── 1r0ig26z.json
│   │   ├── ... 
│   ├── Y/
│   │   ├── 0k99no11.py
│   │   ├── 1r0ig26z.py
│   │   ├── ... 
```


## Code Structure: 
<img width="1253" alt="Screenshot 2024-07-10 at 14 43 50" src="https://github.com/juliankleutgens/task2seq_T5/assets/164042205/8936a55e-7412-4a75-92f9-04fcc5c1dc8e">
Certainly! Here’s a more detailed breakdown of the code structure and functionality for each script without any bold formatting:

### 1. main.py
Function:
- This is the entry point of the application. It coordinates the overall workflow.

Details:
- Initialization: Loads the configuration settings from `config.yaml` or `config_test.yaml` depending on the mode.
- Data Loading: Calls functions from `get_datasetframe.py` to load and preprocess the dataset into a dataframe.
- Training: Initializes the training process by calling functions from `trainer.py`.
- Evaluation: Post-training, it triggers the evaluation process handled by `engine.py`.
- Logging: Directs output to log files and handles any additional logging setup for monitoring (e.g., wandb).

### 2. get_datasetframe.py
Function:
- Loads and preprocesses the dataset, transforming it into a dataframe format.

Details:
- Load Data: Reads JSON files containing ARC tasks and organizes them.
- Dataframe Creation: Converts the loaded data into a dataframe for easy manipulation and access.
- Configuration Handling: Adjusts data loading paths and settings based on the provided configuration, distinguishing between training and testing modes.

### 3. dataloader.py
Function:
- Tokenizes the data and prepares it for training.

Details:
- Tokenization: Converts input-output pairs from the dataframe into tokenized formats that the model can process.
- Batching: Groups data into batches, ensuring they are correctly formatted for model input.
- Data Augmentation: Applies any necessary data transformations or augmentations as specified in the configuration.

### 4. trainer.py
Function:
- Manages the training process of the model.

Details:
- Model Initialization: Sets up the T5 model or any specified model architecture for training.
- Training Loop: Implements the main training loop, including forward passes, loss calculation, and backpropagation.
- Optimization: Uses specified optimizers and learning rate schedules to update model weights.
- Checkpointing: Saves model checkpoints periodically for recovery and later use.
- Metrics Monitoring: Tracks and logs training metrics for performance evaluation.

### 5. engine.py
Function:
- Handles inference and evaluation of the trained model.

Details:
- Model Loading: Loads the trained model from the saved checkpoints.
- Inference: Applies the model to new, unseen data to generate predictions.
- Evaluation: Compares model predictions with ground truth to assess performance.
- Logging: Writes evaluation results to output log files and uploads them to platforms like wandb.com for visualization and further analysis.

### 6. output.log & wandb.com
Function:
- Collects and stores logs and results from the training and evaluation processes.

Details:
- Logging: Aggregates logs from different stages of the workflow, including training metrics, evaluation scores, and any error messages.
- Visualization: Provides tools and interfaces for visualizing training progress and evaluation results.
- Monitoring: Enables continuous monitoring of model performance and health during the training and evaluation phases.

### Data Flow and Interaction

- Data Ingestion: The process begins with `main.py`, which initiates data loading through `get_datasetframe.py`.
- Preprocessing: `get_datasetframe.py` loads raw data and converts it into a structured dataframe.
- Tokenization: `dataloader.py` tokenizes this data and prepares it for the model.
- Training: `trainer.py` takes the tokenized data and runs the training loop, saving models periodically.
- Evaluation: Post-training, `engine.py` loads the trained model and performs inference on new data, evaluating the model's performance.
- Logging: Throughout this workflow, logs and results are collected and stored in `output.log` and optionally sent to `wandb.com` for further analysis.

This structure ensures a clear separation of concerns, with each script focusing on a specific aspect of the workflow. This modularity facilitates easier debugging, maintenance, and extension of the codebase.

## Hyperparameters in `config.yaml` File:

The `config.yaml` file specifies several hyperparameters and settings for training and evaluating the model. Below is an explanation of each key parameter:

**General Parameters:**
- `max_samples`: The maximum number of samples to use for training (e.g., `5000`).
- `test_samples`: The number of samples to use for testing (e.g., `10000`).
- `num_of_itr`: Number of iterations for the training loop (e.g., `5`).

**Paths:**
- `train_paths`: List of paths where the training data is located.
- `test_paths`: List of paths where the testing data is located.

**Mode and Data Loading:**
- `test_mode`: Boolean to indicate if the test mode is active, then the `config_test.yaml` file is overwriting the configurations (e.g., `true`).
- `load_new_mappings`: Boolean to decide if new mappings should be loaded (e.g., `true`).
- `type_of_mapping`: Specifies the type of mapping to be used (e.g., `val2alphabet`).
   - `val2alphabet`: the variables are mapped to the alphabet (e.g. `x1` -> `A`, `x2` -> `B`, `x3` -> `C`, ...)
   - `x2y`: the variables are split into single symbols and `x` is mapped to `y`, because the token `x` is already used to represent the input task grids. (e.g. `x1` -> `['y','1']`, `x21` -> `['y','2','1']`, ...). For that the `extra_token: ['sym_aft_func','EoF','BoF','var_to_num']`

**Device Configuration:**
- `device`: The device to run the model on (e.g., `cuda` for GPU).
- `train_on_multiple_gpus`: Boolean to indicate if training should be done on multiple GPUs (e.g., `false`).
- `n_gpu`: Number of GPUs to use if `train_on_multiple_gpus` is true (e.g., `0`). If one uses `train_on_multiple_gpus: True` the `n_gpu` should enumerate which GPUs to use like (e.g., `- 0 - 1`)

**Training Details:**
- `sparse_type`: Type of sparse representation of an ARC task (e.g., `repeated2words` or `codeit`).
   - `codeit`: The position of every square with a non-background color is written out in integers.
   - `repeated2words`: Write down the color of the squares from the top left to the bottom right. If a color is repeated more than 3 times, the squares are represented as "IntxColor" (e.g., "3xBlack").
- `output_dir`: Directory to save the output files (e.g., `./outputs/`).

**Extra Tokens:**
- `extra_token`: List of extra tokens to be used in the representation and then the tokenization (should be: `['sym_aft_func', 'EoF', 'BoF']`).

**Loss and Prompting:**
- `weighted_loss`: Boolean indicating if a weighted loss should be used after 70% of the iterations in an epoch. The generated tokens at the beginning of the sequence are weighted more compared to the others (a scalar is linearly interpolated between 1.5 and 0.5) (e.g., `False`).
- `prompting`: Boolean indicating if the prompting trick is used. In this trick, the first few tokens of the output sequence, which are the same in any case, are appended at the end of the input sequence (e.g., `False`).

**Model Parameters:**
- `MODEL`: The model name or path (e.g., `t5-small`).
- `TRAIN_BATCH_SIZE`: Batch size for training (e.g., `8`).
- `VALID_BATCH_SIZE`: Batch size for validation (e.g., `8`).
- `TRAIN_EPOCHS`: Number of epochs for training (e.g., `2`).
- `VAL_EPOCHS`: Number of epochs for validation (e.g., `1`).
- `LEARNING_RATE`: Learning rate for the optimizer (e.g., `1.0e-4`).
- `MAX_SOURCE_TEXT_LENGTH`: Maximum length of the source text (e.g., `1024`).
- `MAX_TARGET_TEXT_LENGTH`: Maximum length of the target text (e.g., `1024`).
- `NUM_BEAMS`: Number of beams for beam search (e.g., `15`).
- `fined_tuned_dir`: Directory where the fine-tuned model is saved (e.g., `/home/jkleutgens/task2seq_T5/outputs/output_20240618_2121/model_files` or `None`).

Note that if one uses an already fine-tuned model from a specific file in `fined_tuned_dir`, this overwrites the `MODEL` and also copies the mappings used for training this model. So, no new mapping is generated. 

