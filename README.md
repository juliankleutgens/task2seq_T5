### Abstract:

Humans can generalize sparse experiences into principles and apply them in unfamiliar contexts. In contrast, artificial agents often struggle in novel situations, even with large amounts of data. François Chollet argues this happens because artificial agents are trained to be skillful rather than genuinely intelligent. To address this and steer research toward true AI, he introduced the Abstraction and Reasoning Corpus (ARC), a benchmark designed to measure general intelligence. The ARC dataset evaluates a system’s ability to deduce rules from limited input-output pairs and apply them to new, unseen data.
A key contribution is developing a methodology using a Domain Specific Language (DSL) to represent solutions for ARC tasks. This involves training Large Language Models (LLMs) to generate DSL programs based on ARC dataset input-output pairs. 
Recognizing this approach’s limitations, the research shifts to using the T5 model, a pre-trained LLM, to generate DSL solvers.
Results show the T5 model can learn to generate correct and generalizable DSL solvers, though challenges remain in achieving high accuracy and consistent output generation.

### Setup the Environment for this Repo:
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

   Activate the newly created environment:

   ```bash
   conda activate task2seq
   ```

3. **Install PyTorch according to the server:**

   Install PyTorch, Torchvision, and Torchaudio with CUDA support:

   This is for the 
   ```bash
   pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```

   
4. **Install Related Libraries:**
   ```bash
   pip install omegaconf torchsummary numpy pandas transformers rich wandb fuzzywuzzy sentencepiece matplotlib rouge nltk rapidfuzz inflect Levenshtein
   ```

### Configuration
In the folder "configuration," there are two files: `config.yaml` (used for running on the GPU server with different data addresses) and `config_test.yaml` (used for running the code on my local computer with a CPU). This distinction is made to save time when pushing/pulling and debugging. To use `config_test.yaml`, set `test_mode: true` in `config.yaml`.

### Data Structure
The data should be structured in two ways:

1. Save each task as a separate JSON file with the ID in the file name (e.g., `y65h51dw.json`). Then, save all the solvers in one large `solvers.py` file, where you can find the `def solver_y65h51dw` function.
2. Save both tasks and solvers separately. So, store the task in the '/X/' folder, such as '0k99no11.json', and put the corresponding solver function file into '/Y/' folder like '0k99no11.py'. For the script to recognize this structure, include "training" in the folder name. If you don't want to label it with "training," update the method name in the load_data function in data_scripts/get_datasetframe.py.

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


   
