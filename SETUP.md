# Setup Guide

This guide explains how to properly set up and run the Chat Model project with the new structure.

## Quick Start (Recommended)

### Option 1: Using setup.sh (macOS/Linux)

```bash
cd Project
bash setup.sh
```

This script will:
1. Create a virtual environment
2. Install the package in development mode
3. Install all dependencies
4. Configure everything automatically

Then activate the environment and run training:
```bash
source venv/bin/activate
python scripts/train.py
```

### Option 2: Manual Setup

```bash
cd Project

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install all dependencies
pip install -r requirements.txt
```

## Why This Matters

The project uses a **src layout** which requires the package to be properly installed so Python can find the `chat_model` module. Without installation or proper path setup, you get:

```
ModuleNotFoundError: No module named 'chat_model'
```

## Running Scripts

### Method 1: Using Scripts (Recommended after installation)

```bash
# Train the model
python scripts/train.py

# Run evaluation
python scripts/eval.py

# Run Modal deployment
python scripts/run_modal.py
```

### Method 2: Using Python Module Syntax

```bash
# Train the model
python -m chat_model.training.train

# Run evaluation  
python -m chat_model.evaluation.run_eval
```

### Method 3: From Project Root with Manual Path Setup

```bash
cd Project
python -c "
import sys, os
sys.path.insert(0, 'src')
from chat_model.training.train import main
main()
"
```

## Understanding the Import Structure

### `__init__.py` Files (Newly Populated)

Each module now has properly populated `__init__.py` files that expose the public API:

```python
# src/chat_model/__init__.py
from chat_model import config
from chat_model.model import NanoChat

__all__ = ["config", "NanoChat"]
```

This means you can now do:

```python
# Instead of:
# Instead of:
from chat_model.datasets.loader import load_split
from chat_model.model.model import NanoChat

# You can also do:
from chat_model.datasets import load_split
from chat_model.model import NanoChat
```

### Automatic Path Setup

Each main module file now has automatic path setup:

```python
# Added to every module
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_current_dir, "..", "..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
```

This allows you to run any file directly:

```bash
python src/chat_model/training/train.py
python src/chat_model/evaluation/eval.py
python src/chat_model/tokenizing/train_tokenizer.py
```

## Directory Structure

```
Project/
├── setup.sh                 # Setup script
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── src/
│   └── chat_model/         # Main package (requires installation)
│       ├── __init__.py     # Exposes public API
│       ├── config.py       # Configuration
│       ├── model/
│       ├── training/
│       ├── evaluation/
│       ├── data/
│       ├── tokenizing/
│       ├── chatbot/
│       └── utils/
├── scripts/                # Entry points
│   ├── train.py           # Adds src/ to path
│   ├── eval.py            # Adds src/ to path
│   └── run_modal.py
├── tests/                  # Test suite
└── data/                   # Data directory
```

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'chat_model'`

**Solution 1** (Recommended): Install the package
```bash
cd Project
pip install -e .
```

**Solution 2**: Use Python module syntax
```bash
python -m chat_model.training.train
```

**Solution 3**: Add src to Python path
```bash
cd Project
PYTHONPATH="${PYTHONPATH}:$(pwd)/src" python scripts/train.py
```

### Problem: Config paths are wrong

Check that `config.py` properly calculates `PROJECT_ROOT`:

```python
# Should show the Project directory
python -c "from chat_model import config; print(config.PROJECT_ROOT)"
```

If it's wrong, the issue is likely relative path calculation. Ensure you're running from the project root.

### Problem: Imports work but data files not found

Make sure you have the data downloaded:

```bash
# Download data
python -c "from chat_model.datasets import download_data; download_data()"

# Or preprocess existing data
python -c "from chat_model.datasets import preprocess; preprocess()"
```

### Problem: `ModuleNotFoundError: No module named 'rustbpe'` when running training

**Cause:** The tokenizer was pickled with nanochat's RustBPETokenizer but rustbpe is not installed.

**Solution:** Install rustbpe and nanochat dependencies:
```bash
# Quick fix - just install rustbpe
pip install rustbpe>=0.1.0

# Or install full nanochat from directory
cd ../nanochat
pip install -e .
```

## Installation Modes

### Development Mode (Recommended for development)

```bash
pip install -e .
```

This installs the package in "editable" mode, so changes to the source code are immediately reflected without reinstalling.

### Production Mode

```bash
pip install .
```

This creates a regular installation. You'd need to reinstall to see code changes.

## Dependencies

### Core Dependencies (in requirements.txt)
- torch
- transformers
- datasets
- rouge-score
- sacrebleu
- pyarrow
- pymupdf
- tqdm

### Development Dependencies (optional)
```bash
pip install -e ".[dev]"
```

Includes:
- pytest
- black
- flake8
- isort
- mypy

## Using with NanoChat

The project uses NanoChat's BPE tokenizer for medical vocabulary. NanoChat is cloned as a sibling directory:

```bash
cd ..
git clone https://github.com/karpathy/nanochat.git  # If not already cloned
cd nanochat
```

### Install NanoChat Dependencies

**Option 1: Using pip (Quick)**
```bash
# From Project root
pip install rustbpe>=0.1.0

# Or install full nanochat
cd ../nanochat
pip install -e .
```

**Option 2: Using uv (Recommended)**
```bash
cd ../nanochat
uv sync --extra cpu   # For CPU
# OR
uv sync --extra gpu   # For GPU
```

The tokenizer is pickled with references to `nanochat.tokenizer.RustBPETokenizer`. When training loads the tokenizer via `pickle.load()`, Python needs to import this module.

**Common Error:**
```
ModuleNotFoundError: No module named 'rustbpe'
```
**Solution:** Install rustbpe: `pip install rustbpe>=0.1.0`

NanoChat is imported via:
```python
RustBPETokenizer = importlib.import_module("nanochat.tokenizer").RustBPETokenizer
```

## Configuration

### Environment Variables

See `.env.example` for available settings:

```bash
cp .env.example .env
# Edit .env as needed
```

### Direct Configuration

Modify `src/chat_model/config.py` for permanent changes.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v tests/
```

## Next Steps

1. Run setup script: `bash setup.sh`
2. Activate environment: `source venv/bin/activate`
3. Train model: `python scripts/train.py`
4. Evaluate: `python scripts/eval.py`

See [README.md](README.md) for more information about the project structure and features.
