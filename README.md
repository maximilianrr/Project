# NanoChat Medical

Two-stage transformer for medical Q&A.

---

## Requirements

- Python 3.10+
- nanochat cloned at `../nanochat/`
- Git installed (for MedQuAD clone)

Clone nanochat as a sibling of this project:
git clone https://github.com/karpathy/nanochat.git ../nanochat
Expected folder structure:
dir/
├── Project/      ← this repo
└── nanochat/     ← cloned here

Install dependencies:
```
pip install -r requirements.txt
```

---

## Setup

**Windows:**
```
venv\Scripts\activate.bat
set PYTHONPATH=src
```

**Linux/Bianca:**
```
source venv/bin/activate
export PYTHONPATH=src
```

---

## Run in Order

### 1. Download data
```
python src/chat_model/datasets/download.py
```

For a quick test:
```
python src/chat_model/datasets/download.py --max-documents 50000 --max-abstracts 30000
```

### 2. Preprocess
```
python src/chat_model/datasets/preprocess.py
```

### 3. Build tokenizer corpus
```
python src/chat_model/tokenizing/prepare_data.py
```

### 4. Train tokenizer
Switch to the nanochat environment first:
```
cd ..\nanochat
.venv\Scripts\activate        (Windows)
source .venv/bin/activate     (Linux)
cd ..\Project
python src/chat_model/tokenizing/train_tokenizer.py
```

### 5. Train
Switch back to the project venv:
```
venv\Scripts\activate.bat     (Windows)
source venv/bin/activate      (Linux)
python src/chat_model/training/train.py
```

You will be prompted twice:
- **Start pre-training?** — Stage 1, climbmix + oasst2
- **Start fine-tuning?** — Stage 2, MedDialog + MedQuAD + safety examples

### 6. Chatbot interface

Requires PHP 8.3+. Install on Windows:
```
winget install PHP.PHP.8.3
```

PHP will not be on PATH after install. Set it for the current session.
To avoid doing this every time, run `setx` with the same path and restart the terminal.

Then start the server:
```
cd src\chat_model\chatbot
php -S localhost:9000
```

Visit `http://localhost:9000` in your browser.

Note: `respond.php` calls `python3` by default. On Windows this may need to be changed to `python` if `python3` is not recognised. 
The model checkpoint must be at `data/checkpoints/best_model.pth` before starting the server.

---

## Known Issues

**`num_workers` error on Windows** — multiprocessing fails with `num_workers > 0` on Windows. Set it to 0 in `loader.py`:
```python
num_workers=0,
```

**`ModuleNotFoundError: No module named 'fitz'`** — wrong venv active. Switch to the project venv, not the nanochat one.

**`ModuleNotFoundError: No module named 'nanochat'`** — add nanochat to path:
```
set PYTHONPATH=src;..\nanochat   (Windows)
export PYTHONPATH=src:../nanochat (Linux)
```

---

## Data Layout

```
data/
├── climbmix_parquet/        climbmix shards
├── pretrain_parquet/        oasst2 shards
├── pubmed_parquet/          PubMed shards
├── raw/
│   ├── MedQuAD/
│   ├── meddialog_en/
│   ├── emergency_training_cases.json
│   └── where_there_is_no_doctor_clean.txt
├── splits/                  Stage 2 fine-tune splits
├── pretrain_splits/         Stage 1 splits + climbmix_docs.txt
├── tokenizer_text.txt
├── processed/tokenized/tokenizer.pkl
└── checkpoints/
    ├── pre_trained/         Stage 1 checkpoints
    └── best_model.pth       Stage 2 best model
```

---

## On Bianca (SLURM)

```
python run_bianca.py --mode both
python run_bianca.py --mode pretrain
python run_bianca.py --mode finetune
```