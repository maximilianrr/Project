## Project Structure
 
```
Project/
├── chat_model/
│   ├── models/
│   │   └── language_model.py     
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
 
### 1. Clone this repo
 
```bash
git clone https://github.com/maximilianrr/Project.git
cd Project
git checkout <your-branch>
```
 
### 2. Create virtual environment
 
```bash
# Windows CMD
py -3.12 -m venv venv
venv\Scripts\activate
 
# WSL / Linux / macOS
python3.12 -m venv venv
source venv/bin/activate
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Clone NanoChat separately
 
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
 
## Download Data
 
Run these scripts **in order** from inside the `Project` folder with your venv active:
 
```bash
python scripts/download_data.py       # MedQuAD + MedDialog (~500MB)
python scripts/download_wtnd.py       # Where There Is No Doctor PDF
python scripts/preprocess.py          # Merge and split into train/val/test
```
 
After this we will have:
```
data/splits/
├── train.jsonl    ← 109,277 examples
├── val.jsonl      ← 12,856 examples
└── test.jsonl     ← 6,429 examples
```
 
---
 
## Train the Tokenizer
 
The tokenizer must be trained on medical data before the model can be trained. Run this **from the nanochat folder** with nanochat's venv active:
 
```bash
# Windows
cd ..\nanochat
.venv\Scripts\activate
python ..\Project\scripts\prepare_tokenizer_data.py  # back in Project venv first
python ..\Project\scripts\convert_to_parquet.py
 
# Then from nanochat venv:
python ..\Project\scripts\train_tokenizer.py
```
 
The tokenizer saves to `~/.cache/nanochat/tokenizer/`.
 
---
 
## Data Sources
 
| Source | Size | Purpose |
|---|---|---|
| [MedQuAD](https://github.com/abachaa/MedQuAD) | 16,406 QA pairs | Medical Q&A from NIH |
| [ChatDoctor / MedDialog](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) | 112,165 conversations | Real doctor-patient conversations |
| [Where There Is No Doctor](https://hesperian.org) | 503 pages | Village health handbook for developing countries |
