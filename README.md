## Project Structure
 
```
Project/
├── chat_model/
│   ├── models/
│   │   └── model.py     
│   └── utils/
│       ├── data_loader.py        
│       └── train.py              
├── scripts/
│   ├── download_data.py          
│   ├── download_wtnd.py          
│   ├── preprocess.py             
│   ├── prepare_tokenizer_data.py 
│   ├── convert_to_parquet.py     
│   └── train_tokenizer.py        
├── requirements.txt
└── README.md
```
 
---
 
## Prerequisites
 
- Python 3.12
- Git
- [uv](https://github.com/astral-sh/uv) — install with:
  ```
  pip install uv
  ```
 
---
 
## Setup
 
### 1. Create virtual environment
 
```bash
# Windows CMD
py -3.12 -m venv venv
venv\Scripts\activate
 
# WSL / Linux / macOS
python3.12 -m venv venv
source venv/bin/activate
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Clone NanoChat separately
 
NanoChat is an external dependency. It lives **outside** this repo.
 
```bash
cd ..
git clone https://github.com/karpathy/nanochat.git
cd nanochat
uv venv
uv sync --extra cpu
```
 
Folder structure should look like:
```
parent_folder/
├── Project/       ← this repo
└── nanochat/      ← external dependency
```
 
---

## Model Architecture (model.py)

The core transformer model is implemented in:
```
chat_model/models/model.py
```

This file contains a GPT-style decoder-only transformer built with PyTorch. The implementation includes:
```
Multi-head self-attention
Rotary Positional Embeddings (RoPE)
KV-cache support for fast generation
Pre-LayerNorm transformer blocks
Top-k and temperature sampling during generation
```
## Main Components
| Component                      | Description                                                                |
| ------------------------------ | -------------------------------------------------------------------------- |
| `KVCache`                      | Stores past key/value tensors during inference for faster token generation |
| `precompute_rope_embeddings()` | Precomputes cosine/sine tensors used for Rotary Positional Embeddings      |
| `apply_rope_embeddings()`      | Applies RoPE rotations to query/key tensors                                |
| `AllHeadAttention`             | Multi-head causal self-attention implementation                            |
| `FeedForward`                  | Transformer MLP block with GELU activation                                 |
| `Block`                        | Single transformer block consisting of attention + feedforward             |
| `NanoChat`                     | Full decoder-only language model                                           |



## Attention Implementation

Attention uses PyTorch's optimized:
```
torch.nn.functional.scaled_dot_product_attention()
```
Which Features:
```
Flash-attention compatible backend (when supported)
Causal masking
Dropout during training
Multi-head parallel attention
```

## The generate() method

Feature	Description:
```
Temperature:	Controls randomness by making logits vector more sharp
Top-k sampling:	Restricts sampling to top-k tokens
Stop token:	Early stopping when <|end|> token appears
KV caching:	for faster generation
```
Example usage:
```
generated = model.generate(
    input_tokens,
    max_new_tokens=200,
    temperature=0.5,
    top_k=10,
    stop_token_id=stop_token_id
)

returns input_tokens with generated tokens added
```
---

## Training the model
The data loading, tokenizer training and actual model training are executed through the main function in the ```train.py``` file. 

When running the training for the first time, it is essential to set the ```load_data``` and ```init_tokenizer``` parameters to ```True```in order to download the data needed for model training and tokenize it using the tokenizer from the nanochat project. It is important to stick to the proposed directory structure above to ensure proper initialization of the tokenizer. 

Before starting the training, the virtual environment needs to be activated and you need to nevigate to the following directory in your terminal: ```Group_Project/Project/chat_model/```. 

The training can be started as follows if the data is not yet loaded and the tokenizer not trained: 
```python train.py --load_data=True --init_tokenizer=True```

Otherwise, the following command can be used to start the training: 
```python train.py```
Because ```load_data``` and ```init_tokenizer``` both default to ```False```, there is no need to set the parameters when calling the method. 

 
## Download Data separately
In order to download the data without using the training loop, call this file from inside the `Project` folder with your venv active:
 
```bash
python chat_model/utils/data_loading/download_data.py

```
 
After this we will have:
```
data/splits/
├── train.jsonl    ← 109,277 examples
├── val.jsonl      ← 12,856 examples
└── test.jsonl     ← 6,429 examples
```
  
---

## Train the Model

Note inside train.py:
```
    parser.add_argument("--load_data", default=False, type=bool, help="Whether to download and preprocess the data before training. Default is False.")
    parser.add_argument("--init_tokenizer", default=False, type=bool, help="Whether to initialize the tokenizer before training. Default is False.")

```

From inside `chat_model/` with the venv active:

```bash
python train.py
```
or if you want to download dataset in this step:
```bash
python train.py --load_data True
```
or if you want to create and train a new tokenizer:
```bash
python train.py --init_tokenizer True
```
If you do not init_tokenizer it will use the tokenizer we used to train our best model.

This trains the model and saves the best checkpoint. Trained weights are saved as `weights_<config>.pth.zip` in the working directory.

---


 ## Run evaluation

```bash
python run_eval.py
```

### Using a trained checkpoint

Open `run_eval.py` and set the `CHECKPOINT` variable at the top:

```python
CHECKPOINT = "path/to/checkpoint.pt"
```

Then run as above.

---

## What the eval measures

| Metric | Description |
|---|---|
| `perplexity` | Cross-entropy loss on the val set (lower = better) |
| `bleu` | BLEU score vs. MedQuAD reference answers |
| `rouge1_f` / `rouge2_f` / `rougeL_f` | ROUGE F1 scores |
| `safety_escalation_rate` | Fraction of high-risk prompts that produced an emergency referral |
| `safety_unsafe_rate` | Fraction of responses matching unsafe patterns |
| `safety_missing_escalation` | IDs of high-risk cases that failed to escalate |
| `safety_flagged` | IDs of responses that matched unsafe patterns |

The safety suite (`safety_cases.json`) contains 15 high-risk and 15 low-risk prompts.

## Data Sources
 
| Source | Size | Purpose |
|---|---|---|
| [MedQuAD](https://github.com/abachaa/MedQuAD) | 16,406 QA pairs | Medical Q&A from NIH |
| [ChatDoctor / MedDialog](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) | 112,165 conversations | Real doctor-patient conversations |
| [Where There Is No Doctor](https://hesperian.org) | 503 pages | Village health handbook for developing countries |
