# NanoChat Medical: Advanced ML Project

> Professional conversational AI system for medical domains with production-ready architecture and comprehensive documentation.

**Status:** ✅ Production Ready | **Python:** 3.10+ | **License:** MIT

---

## Quick Links

📚 **[Comprehensive Guide](README_COMPREHENSIVE.md)** - Full documentation (30+ pages)  
🚀 **[Setup Guide](SETUP.md)** - Installation instructions  
📋 **[Restructuring Summary](RESTRUCTURING_SUMMARY.md)** - Architecture changes  

---

## What is NanoChat Medical?

A specialized conversational AI system trained on medical knowledge bases (MedQuAD, MedDialog, WTND). Demonstrates professional Python project structure, efficient data pipelines, and comprehensive evaluation metrics for domain-specific language models.

### Key Features

✅ Clean, modular architecture following Python best practices  
✅ Feature-based organization (training, evaluation, data, tokenizing, chatbot)  
✅ Complete data pipeline from raw sources to trained model  
✅ Multiple evaluation metrics (Perplexity, BLEU, ROUGE, Safety)  
✅ Web interface for testing inference  
✅ Hyperparameter tuning framework  
✅ Professional packaging (`pyproject.toml`, `setup.sh`)  

---

## 5-Minute Quick Start

### 1. Install

```bash
cd Project
bash setup.sh                    # Automated setup
source venv/bin/activate
pip install rustbpe>=0.1.0      # NanoChat dependency
```

### 2. Train

```bash
python scripts/train.py          # Begin training
```

### 3. Evaluate

```bash
python scripts/eval.py           # Run evaluation metrics
```

### 4. Chatbot

```bash
cd src/chat_model/chatbot
php -S localhost:9000
# Visit: http://localhost:9000
```

---

## Project Structure

This project now follows Python best practices with a **src/ layout** and **feature-based organization** for better maintainability and scalability.

## Project Structure

This project now follows Python best practices with a **src/ layout** and **feature-based organization** for better maintainability and scalability.

```
Project/
├── src/chat_model/              Main package (src layout)
│   ├── model/                   Neural network architecture
│   ├── training/                Training loops & HP tuning
│   ├── evaluation/              Evaluation metrics
│   ├── data/                    Data loading & preprocessing
│   ├── tokenizing/              BPE tokenizer training
│   ├── chatbot/                 Web inference interface
│   ├── utils/                   Shared utilities
│   └── config.py                Centralized configuration
├── scripts/                     Entry points (train.py, eval.py)
├── tests/                       Test suite
├── data/                        Datasets & checkpoints
├── pyproject.toml               Package configuration
├── setup.sh                     Setup script
├── requirements.txt             Dependencies
└── README_COMPREHENSIVE.md      Full documentation (30+ pages)
```

### Why This Structure?

| Feature | Benefit |
|---------|---------|
| `src/` layout | Prevents import bugs, professional packaging |
| Feature modules | Clear organization, easy navigation |
| Populated `__init__.py` | Clean import API |
| Centralized config | Single source of truth |
| Entry point scripts | Multiple execution methods |

---

## Core Modules

| Module | Purpose |
|--------|---------|
| **config.py** | Centralized settings & paths |
| **model/model.py** | NanoChat transformer architecture |
| **training/train.py** | Main training loop with checkpointing |
| **training/hyperparameter_train.py** | HP tuning framework |
| **evaluation/eval.py** | Perplexity, BLEU, ROUGE, safety metrics |
| **data/loader.py** | Data loading & batching |
| **data/download.py** | Download MedQuAD, MedDialog, WTND |
| **data/preprocess.py** | Clean & split data |
| **tokenizing/train_tokenizer.py** | BPE tokenizer training |
| **chatbot/main.py** | Inference interface |

---

## Three Ways to Run

### Method 1: Entry Scripts (Recommended)
```bash
python scripts/train.py      # Simple and clear
python scripts/eval.py
```

### Method 2: Module Syntax
```bash
python -m chat_model.training.train
python -m chat_model.evaluation.run_eval
```

### Method 3: Direct Execution
```bash
python src/chat_model/training/train.py
python src/chat_model/evaluation/eval.py
```

---

## Configuration

### Default Settings

```python
# src/chat_model/config.py

# Model architecture
N_EMB = 512              # Embedding dimension
BLOCK_SIZE = 1024        # Context window
N_LAYER = 8              # Transformer layers
N_HEAD = 8               # Attention heads

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1.5e-4
EPOCHS = 10
PATIENCE = 3             # Early stopping

# Tokenizer
VOCAB_SIZE = 32768
```

### Customize

**Option A: Edit config.py** (permanent)
```python
BATCH_SIZE = 32  # Change default
```

**Option B: Command-line** (one-time)
```bash
python scripts/train.py --batch-size 32 --lr 1e-4
```

---

## Data Pipeline

### 1. Download Raw Data
```python
from chat_model.datasets import download_data
download_data()  # MedQuAD, MedDialog, WTND
```

### 2. Preprocess
```python
from chat_model.datasets import preprocess
preprocess()  # Creates train/val/test splits
```

### 3. Prepare Tokenizer Data
```python
from chat_model.tokenizing import prepare_tokenizer_data
prepare_tokenizer_data()  # Combines all data
```

### 4. Train Tokenizer
```python
from chat_model.tokenizing import train_tokenizer
train_tokenizer()  # BPE on medical vocabulary
```

### 5. Train Model
```python
from chat_model.training import main
main()  # Training loop
```

---

## Evaluation Metrics

- **Perplexity** - Language model fit on validation set
- **BLEU** - N-gram overlap with MedQuAD references
- **ROUGE** - Semantic similarity with references
- **Safety** - Rule-based medical safety validation

```bash
python scripts/eval.py  # Compute all metrics
```

---

## Dependencies

### Core
- torch >= 2.0
- transformers
- datasets
- PyYAML, Pandas, NumPy

### Tokenization
- rustbpe >= 0.1.0 (from NanoChat)

### Evaluation
- rouge-score
- sacrebleu

### Optional
- pytest, black, flake8, mypy (development)

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'chat_model'`
```bash
pip install -e .  # Install package in development mode
```

### `ModuleNotFoundError: No module named 'rustbpe'`
```bash
pip install rustbpe>=0.1.0  # Install NanoChat dependency
```

### Data files not found
```bash
python -c "from chat_model.datasets import download_data; download_data()"
```

### For more help
See [SETUP.md](SETUP.md) troubleshooting section or [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)

---

## Development

### Adding Features

1. Create module in `src/chat_model/{feature}/`
2. Populate `__init__.py` with public API
3. Add tests in `tests/test_{feature}.py`
4. Update `config.py` if new settings needed

### Testing
```bash
pytest tests/               # Run all tests
pytest tests/test_data.py   # Run specific test
pytest -v tests/            # Verbose output
```

### Code Quality
```bash
black src/ tests/           # Format code
isort src/ tests/           # Sort imports
flake8 src/ tests/          # Lint
mypy src/chat_model/        # Type check
```

---

## Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **README.md** (this file) | Quick overview & links | Everyone |
| **README_COMPREHENSIVE.md** | Complete 30+ page guide | Developers |
| **SETUP.md** | Installation & troubleshooting | New users |
| **RESTRUCTURING_SUMMARY.md** | Architecture decisions | Team leads |

📚 **→ [Read Comprehensive Guide](README_COMPREHENSIVE.md)** for:
- Detailed architecture & design decisions
- Complete module documentation
- Advanced topics (distributed training, profiling, etc.)
- Performance optimization
- Contributing guidelines

---

## Getting Help

1. **Quick questions?** See [SETUP.md](SETUP.md)
2. **Need details?** Read [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)
3. **Want to understand changes?** Check [RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md)
4. **Looking for examples?** See `tests/` folder
5. **Still stuck?** Check docstrings in source files

---

## Performance Tips

- **GPU**: Use `CUDA_VISIBLE_DEVICES` to select GPU
- **Memory**: Reduce `BATCH_SIZE` in config if out of memory
- **Speed**: Use `torch.jit.script()` for inference optimization
- **Distributed**: Use `torch.distributed.launch` for multi-GPU

---

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and write tests
3. Format: `black src/ tests/`
4. Lint: `flake8 src/ tests/`
5. Test: `pytest tests/`
6. Commit and push
7. Create pull request

---

## Project Status

✅ **Fully Functional**
- ✅ Training pipeline
- ✅ Evaluation metrics
- ✅ Data processing
- ✅ Inference interface
- ✅ Web chatbot

✅ **Well Documented**
- ✅ Comprehensive README
- ✅ Setup guide
- ✅ Inline documentation
- ✅ Test examples

✅ **Production Ready**
- ✅ Professional structure
- ✅ Error handling
- ✅ Configuration system
- ✅ Testing framework

---

## Quick Reference

```bash
# Setup
bash setup.sh

# Training
python scripts/train.py

# Evaluation  
python scripts/eval.py

# Interactive
python -c "from chat_model.training import main; main()"

# Chatbot
cd src/chat_model/chatbot && php -S localhost:9000

# Tests
pytest tests/

# Full documentation
open README_COMPREHENSIVE.md
```

---

## License

MIT License - See LICENSE file for details

---

**Last Updated:** May 2026  
**Python Version:** 3.10+  
**Status:** ✅ Production Ready  

📚 **[Full Documentation →](README_COMPREHENSIVE.md)**
