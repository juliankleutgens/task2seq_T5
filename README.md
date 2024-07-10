**Abstract:**

Humans can generalize sparse experiences into principles and apply them in unfamiliar contexts. In contrast, artificial agents often struggle in novel situations, even with large amounts of data. François Chollet argues this happens because artificial agents are trained to be skillful rather than genuinely intelligent. To address this and steer research toward true AI, he introduced the Abstraction and Reasoning Corpus (ARC), a benchmark designed to measure general intelligence. The ARC dataset evaluates a system’s ability to deduce rules from limited input-output pairs and apply them to new, unseen data.
A key contribution is developing a methodology using a Domain Specific Language (DSL) to represent solutions for ARC tasks. This involves training Large Language Models (LLMs) to generate DSL programs based on ARC dataset input-output pairs. 
Recognizing this approach’s limitations, the research shifts to using the T5 model, a pre-trained LLM, to generate DSL solvers.
Results show the T5 model can learn to generate correct and generalizable DSL solvers, though challenges remain in achieving high accuracy and consistent output generation.

**Setup the Environment for this Repo:**
### Prerequisites

- Python 3.6 or higher
- Conda

### Step-by-Step Instructions

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

3. **Install PyTorch and Related Libraries:**

   Install PyTorch, Torchvision, and Torchaudio with CUDA support:

   ```bash
   pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```


   
