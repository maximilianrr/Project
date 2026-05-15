# NanoChat Medical: Advanced Machine Learning Project

> A production-ready, professionally structured Python project implementing a specialized conversational AI system for medical query handling. Built with clean architecture principles, modern packaging standards, and comprehensive data pipelines for medical knowledge retrieval and synthesis.

**Status:** ✅ Fully Restructured and Working  
**Architecture:** src-layout with feature-based modules  
**Python:** 3.10+  
**License:** MIT  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What's New: Project Restructuring](#whats-new-project-restructuring)
3. [Architecture & Design](#architecture--design)
4. [Installation & Setup](#installation--setup)
5. [Project Structure](#project-structure)
6. [Core Modules](#core-modules)
7. [Configuration System](#configuration-system)
8. [Data Pipeline](#data-pipeline)
9. [Running the Project](#running-the-project)
10. [Development Workflow](#development-workflow)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Topics](#advanced-topics)

---

## Project Overview

### What is NanoChat Medical?

NanoChat Medical is a specialized conversational AI system trained on medical knowledge bases to provide accurate, safe, and contextually appropriate responses to medical queries. The project demonstrates how to:

- **Efficiently train** language models on domain-specific data
- **Organize** large Python projects following industry best practices
- **Manage** complex data pipelines from raw sources to tokenized datasets
- **Evaluate** model quality with domain-specific metrics
- **Deploy** inference services with safety guardrails

### Key Features

✅ **Clean Architecture** - Modular design with clear separation of concerns  
✅ **Professional Packaging** - Modern Python packaging with `pyproject.toml`  
✅ **Comprehensive Pipelines** - From raw data to trained model in one workflow  
✅ **Medical Safety** - Safety checks and validation for medical domain  
✅ **Multiple Evaluation Metrics** - Perplexity, BLEU, ROUGE, custom safety metrics  
✅ **Web Interface** - PHP-based chatbot UI for testing inference  
✅ **Hyperparameter Tuning** - Built-in support for systematic HP optimization  
✅ **Flexible Execution** - Multiple ways to run scripts (entry points, -m syntax, direct)  

### Technologies Used

- **PyTorch 2.0+** - Deep learning framework
- **Transformers** - Tokenization and model utilities
- **HuggingFace Datasets** - Data loading and processing
- **BPE Tokenizer** - Via Rust FFI for performance
- **ROUGE/BLEU** - Evaluation metrics
- **PHP** - Web interface backend

---

## What's New: Project Restructuring

### The Problem

The original project had a flat directory structure that made it difficult to:
- Scale to new features
- Organize related functionality
- Maintain consistent import patterns
- Distribute as a package
- Understand the codebase structure

### The Solution: Professional src-layout

The project was restructured following industry Python best practices:

#### Before → After

```
❌ Before                          ✅ After
├── chat_model/                    ├── src/
│   ├── train.py                   │   └── chat_model/
│   ├── gpt.py                     │       ├── model/
│   ├── core_eval.py               │       ├── training/
│   ├── dataloader.py              │       ├── evaluation/
│   └── ...                        │       ├── data/
├── scripts/                       │       ├── tokenizing/
├── tests/                         │       ├── chatbot/
└── data/                          │       └── utils/
                                   ├── scripts/
                                   ├── tests/
                                   ├── data/
                                   ├── pyproject.toml
                                   └── setup.sh
```

### Key Improvements

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| **Location** | Root-level package | `src/` directory | Prevents import bugs, proper packaging |
| **Organization** | Mixed file types | Feature modules | Clear organization, easy navigation |
| **Imports** | Complex paths | Clean, clear API | Better readability and maintainability |
| **Dependencies** | Always required | Lazy-loaded | Optional deps don't break imports |
| **Testing** | Manual path hacks | Pytest auto-discovery | Professional test setup |
| **Distribution** | Not possible | `pip install -e .` ready | Production-ready packaging |
| **Configuration** | Scattered settings | Centralized | Single source of truth |
| **Documentation** | Minimal | SETUP.md, README | Professional documentation |

---

## Architecture & Design

### Design Philosophy

The project follows these core principles:

#### 1. **Source Layout** (`src/` directory)
```python
# Why? Separates application code from configuration
# Prevents import bugs when package has same name as other modules
# Makes package behavior identical whether installed or in development
```

**Before:** `from chat_model import config` 
**After:** Still `from chat_model import config`, but now it's in `src/chat_model/`

#### 2. **Feature-Based Organization**
Code is organized by responsibility, not file type:

```
src/chat_model/
├── model/          # What it does: Neural network architecture
├── training/       # What it does: Train the model
├── evaluation/     # What it does: Compute metrics
├── data/          # What it does: Load and process data
├── tokenizing/    # What it does: Train tokenizer
├── chatbot/       # What it does: Inference interface
└── utils/         # What it does: Shared helpers
```

**Benefit:** A developer looking for training code doesn't dig through 10 files—they go to `training/`.

#### 3. **Separation of Concerns**
- Scripts are thin entry points
- Business logic lives in modules
- Each module has a single responsibility

#### 4. **Proper Package Structure**
Each module has a populated `__init__.py` that exposes public API:

```python
# src/chat_model/training/__init__.py
from chat_model.training.train import main, save_checkpoint, load_checkpoint
from chat_model.training.hyperparameter_train import train_trial

__all__ = ["main", "save_checkpoint", "load_checkpoint", "train_trial"]
```

**Benefit:** Users can do `from chat_model.training import main` instead of `from chat_model.training.train import main`

#### 5. **Centralized Configuration**
All settings in one place (`config.py`):

```python
# One source of truth for all paths and settings
PROJECT_ROOT = "/path/to/Project"
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data/processed/splits")
BATCH_SIZE = 16
LEARNING_RATE = 1.5e-4
```

### Module Responsibilities

#### `config.py` - Configuration Hub
- **Purpose**: Centralized settings and path calculations
- **Responsibility**: All environment, model, training, data settings
- **Used by**: Every other module
- **Key exports**: PROJECT_ROOT, SPLITS_DIR, BATCH_SIZE, LEARNING_RATE, etc.

#### `model/model.py` - Neural Network
- **Purpose**: NanoChat transformer architecture
- **Components**: 
  - `KVCache` - Key/value cache for efficient generation
  - `AllHeadAttention` - Multi-head attention with RoPE embeddings
  - `NanoChat` - Main model class
- **Used by**: Training and evaluation modules

#### `training/train.py` - Main Training Loop
- **Purpose**: Train the model from scratch
- **Features**:
  - Checkpoint saving/loading
  - Mixed precision training (AMP)
  - Learning rate scheduling
  - Early stopping with patience
  - Progress tracking with TQDM
- **Used by**: Data scientists, researchers

#### `training/hyperparameter_train.py` - HP Tuning
- **Purpose**: Systematic hyperparameter search
- **Features**:
  - Grid or random search
  - Trial-based experiments
  - Leaderboard tracking
  - Configuration logging
- **Used by**: ML engineers optimizing models

#### `evaluation/eval.py` - Metrics Computation
- **Purpose**: Compute evaluation metrics
- **Metrics**:
  - **Perplexity** - Language model fit on validation data
  - **BLEU** - Against reference MedQuAD answers
  - **ROUGE** - N-gram overlap with references
  - **Safety** - Rule-based medical safety checks
- **Used by**: Model evaluation pipeline

#### `evaluation/run_eval.py` - Evaluation Runner
- **Purpose**: End-to-end evaluation on checkpoints
- **Process**:
  1. Load model checkpoint
  2. Load tokenizer
  3. Create dataloaders
  4. Run all metrics
  5. Report results
- **Used by**: Continuous evaluation

#### `data/loader.py` - Data Loading
- **Purpose**: Load and batch data for training
- **Key functions**:
  - `load_split()` - Load train/val/test JSONL
  - `build_dataset()` - Create PyTorch Dataset
  - `make_dataloader()` - Create DataLoader
  - `load_and_convert_data()` - Full pipeline
- **Used by**: Training and evaluation

#### `data/chunk_dataset.py` - PyTorch Dataset
- **Purpose**: Custom Dataset class for chunked token sequences
- **Features**:
  - Converts conversations to token streams
  - Creates fixed-size chunks (blocks)
  - Handles max_conversations and max_blocks for debugging
- **Used by**: Data loader

#### `data/download.py` - Data Downloading
- **Purpose**: Download raw medical datasets
- **Datasets**:
  - **MedDialog** - Doctor-patient conversations via HuggingFace
  - **MedQuAD** - Q&A from NIH sources
  - **WTND** - "Where There Is No Doctor" PDF extraction
- **Used by**: Data pipeline initialization

#### `data/preprocess.py` - Data Preprocessing
- **Purpose**: Clean and split raw data
- **Process**:
  1. Load raw data from multiple sources
  2. Clean text (remove HTML, normalize, etc.)
  3. Convert to conversation format
  4. Train/val/test split (80/10/10)
  5. Save as JSONL files
- **Used by**: Data pipeline

#### `tokenizing/train_tokenizer.py` - Tokenizer Training
- **Purpose**: Train BPE tokenizer on medical vocabulary
- **Features**:
  - Uses RustBPE for speed
  - Handles large text files
  - Saves pickled tokenizer
- **Used by**: Setup/initialization

#### `tokenizing/convert_to_parquet.py` - Parquet Conversion
- **Purpose**: Convert text to Parquet shards for efficient tokenizer training
- **Why**: Parquet is optimized for distributed processing
- **Used by**: Tokenizer training pipeline

#### `tokenizing/prepare_data.py` - Tokenizer Data Prep
- **Purpose**: Combine splits and raw text into single tokenizer training file
- **Output**: Single text file with all training data
- **Used by**: Tokenizer training pipeline

#### `chatbot/main.py` - Inference Interface
- **Purpose**: Chatbot entry point for inference
- **Features**:
  - Load model and tokenizer
  - Process user input
  - Generate responses
  - Apply safety checks
- **Used by**: End users, testing

#### `utils/common.py` - Shared Utilities
- **Purpose**: Common helper functions
- **Examples**: File I/O, path utilities, logging helpers

---

## Installation & Setup

### Prerequisites

- **Python 3.10 or higher** (3.12 recommended)
- **Git** for cloning repositories
- **Conda** (optional, recommended for dependency management)
- **CUDA 12.1+** (optional, for GPU training)

### Quick Start (5 minutes)

#### Option 1: Automated Setup (Recommended)

```bash
# Navigate to project
cd /path/to/Project

# Run setup script
bash setup.sh

# Activate environment
source venv/bin/activate

# Install NanoChat dependencies
pip install rustbpe>=0.1.0

# Test installation
python scripts/train.py --help
```

#### Option 2: Manual Setup

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install all dependencies
pip install -r requirements.txt

# Install NanoChat dependencies
pip install rustbpe>=0.1.0

# Verify installation
python -c "from chat_model import config; print(config.PROJECT_ROOT)"
```

### Installing NanoChat (External Dependency)

The project depends on NanoChat's RustBPETokenizer for efficient medical vocabulary tokenization.

```bash
# Navigate to parent directory
cd /path/to/Advanced_Machine_Learning/Group_Project

# Clone if not already present
git clone https://github.com/karpathy/nanochat.git

# Install NanoChat dependencies
cd nanochat

# Option A: Using pip
pip install rustbpe>=0.1.0
pip install -e .

# Option B: Using uv (recommended)
uv sync --extra cpu   # For CPU
# or
uv sync --extra gpu   # For GPU
```

### Verification

```bash
# Test imports
python -c "
from chat_model import config
from chat_model.model import NanoChat
from chat_model.training import main
print('✅ All imports successful')
"

# Test data paths
python -c "
from chat_model import config
print(f'PROJECT_ROOT: {config.PROJECT_ROOT}')
print(f'SPLITS_DIR: {config.SPLITS_DIR}')
print(f'CHECKPOINTS_DIR: {config.CHECKPOINTS_DIR}')
"

# Test configuration
python scripts/train.py --help
```

---

## Project Structure

### Directory Tree

```
Project/
│
├── 📄 pyproject.toml              Modern Python packaging configuration
├── 📄 setup.sh                     Automated setup script
├── 📄 requirements.txt             Package dependencies
├── 📄 .env.example                 Environment variables template
├── 📄 SETUP.md                     Setup guide with troubleshooting
├── 📄 README.md                    Original README
├── 📄 RESTRUCTURING_SUMMARY.md     Detailed restructuring explanation
│
├── 📁 src/chat_model/             Main package (src layout)
│   ├── 📄 __init__.py             Package initialization (v0.1.0)
│   ├── 📄 config.py               Centralized configuration
│   │
│   ├── 📁 model/                  Neural network architecture
│   │   ├── 📄 __init__.py
│   │   └── 📄 model.py            NanoChat transformer
│   │
│   ├── 📁 training/               Training logic (feature module)
│   │   ├── 📄 __init__.py
│   │   ├── 📄 train.py            Main training loop
│   │   └── 📄 hyperparameter_train.py    HP tuning
│   │
│   ├── 📁 evaluation/             Evaluation metrics (feature module)
│   │   ├── 📄 __init__.py
│   │   ├── 📄 eval.py             Metric computation
│   │   └── 📄 run_eval.py         Evaluation runner
│   │
│   ├── 📁 data/                   Data handling (feature module)
│   │   ├── 📄 __init__.py
│   │   ├── 📄 loader.py           Data loading & batching
│   │   ├── 📄 tokenizer.py        Tokenizer orchestration
│   │   ├── 📄 chunk_dataset.py    PyTorch Dataset class
│   │   ├── 📄 download.py         Download raw data
│   │   └── 📄 preprocess.py       Preprocess & split
│   │
│   ├── 📁 tokenizing/             Tokenizer training (feature module)
│   │   ├── 📄 __init__.py
│   │   ├── 📄 train_tokenizer.py  Train BPE tokenizer
│   │   ├── 📄 convert_to_parquet.py    Convert to Parquet
│   │   └── 📄 prepare_data.py     Prepare training text
│   │
│   ├── 📁 chatbot/                Chatbot interface (feature module)
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main.py             Chatbot entry point
│   │   ├── 📄 processor.py        Input processing
│   │   ├── 📄 safety_cases.json   Safety validation rules
│   │   ├── 📄 index.php           Web UI
│   │   ├── 📄 respond.php         Response handler
│   │   ├── 📄 script.js           Frontend logic
│   │   └── 📄 style.css           Styling
│   │
│   └── 📁 utils/                  Shared utilities
│       ├── 📄 __init__.py
│       └── 📄 common.py           Helper functions
│
├── 📁 scripts/                    Entry point scripts
│   ├── 📄 train.py                Training entry point
│   ├── 📄 eval.py                 Evaluation entry point
│   └── 📄 run_modal.py            Modal deployment
│
├── 📁 tests/                      Test suite (mirrors src)
│   ├── 📄 __init__.py
│   ├── 📄 test_model.py
│   ├── 📄 test_data.py
│   └── 📄 test_evaluation.py
│
└── 📁 data/                       Data directory (git-ignored)
    ├── 📁 raw/                    Original datasets
    │   ├── 📁 MedQuAD/           Medical Q&A (NIH)
    │   ├── 📁 meddialog_en/      Doctor-patient conversations
    │   └── 📄 where_there_is_no_doctor_clean.txt
    ├── 📁 processed/             Processed data
    │   ├── 📁 splits/            Train/val/test JSONL files
    │   ├── 📁 parquet/           Tokenizer training shards
    │   └── 📁 tokenized/         Tokenized data
    └── 📁 checkpoints/           Model checkpoints
        ├── 📄 best.pth           Best model weights
        ├── 📄 latest.pth         Latest checkpoint
        └── 📄 history.json       Training history
```

### File Naming Conventions

- **Configuration files**: `config.py`, `.env.example`
- **Main logic**: `{feature}.py` (e.g., `train.py`, `eval.py`)
- **Utilities**: `{name}_utils.py` or `common.py`
- **Data classes**: `{entity}.py` (e.g., `dataset.py`, `loader.py`)
- **Tests**: `test_{module}.py`
- **Scripts**: `{action}.py` (e.g., `train.py`, `eval.py`)

---

## Core Modules

### Configuration System (`config.py`)

Centralized configuration with automatic path calculation:

```python
# Automatically finds project root from src layout
PROJECT_ROOT = /path/to/Project

# Data paths (auto-calculated)
DATA_DIR = {PROJECT_ROOT}/data
RAW_DIR = {DATA_DIR}/raw
PROCESSED_DIR = {DATA_DIR}/processed
SPLITS_DIR = {PROCESSED_DIR}/splits           # JSONL files
PARQUET_DIR = {PROCESSED_DIR}/parquet         # Shards for tokenizer
TOKENIZED_DIR = {PROCESSED_DIR}/tokenized     # Tokenizer output
CHECKPOINTS_DIR = {DATA_DIR}/checkpoints      # Model weights

# Model architecture
N_EMB = 512                  # Embedding dimension
BLOCK_SIZE = 1024            # Context window
N_LAYER = 8                  # Number of layers
N_HEAD = 8                   # Number of attention heads
DROPOUT = 0.22               # Dropout rate

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1.5e-4
EPOCHS = 10
PATIENCE = 3                 # Early stopping patience

# Tokenizer settings
VOCAB_SIZE = 32768
MAX_CHARS = 500000000
DOC_CAP = 10000
```

### Model Architecture

The NanoChat model is a transformer-based language model:

```
Input Tokens
    ↓
Token Embedding (N_EMB dimensions)
    ↓
Positional Encoding
    ↓
[Transformer Block] × N_LAYER
    ├─ Multi-Head Attention (N_HEAD heads)
    ├─ Feed-Forward Network
    ├─ Layer Norm + Residuals
    └─ Dropout
    ↓
Output Projection (N_EMB → VOCAB_SIZE)
    ↓
Logits
```

**Key Features:**
- Rotary Position Embeddings (RoPE)
- Multi-head self-attention
- KV cache for efficient generation
- Grouped query attention support

### Data Processing Pipeline

#### Step 1: Download Raw Data
```python
from chat_model.datasets import download_data
download_data()  # Downloads MedDialog, MedQuAD, WTND
```

**Datasets:**
- **MedDialog**: 230K doctor-patient conversations
- **MedQuAD**: 47K medical Q&A from NIH
- **WTND**: "Where There Is No Doctor" medical text

#### Step 2: Preprocess
```python
from chat_model.datasets import preprocess
preprocess()  # Cleans data, creates train/val/test splits
```

**Output**: `data/processed/splits/{train,val,test}.jsonl`

#### Step 3: Prepare Tokenizer Data
```python
from chat_model.tokenizing import prepare_tokenizer_data
prepare_tokenizer_data()  # Combines all data into single text file
```

**Output**: `data/processed/tokenizer_text.txt`

#### Step 4: Train Tokenizer
```python
from chat_model.tokenizing import train_tokenizer
train_tokenizer()  # Trains BPE tokenizer on medical vocab
```

**Output**: 
- `data/processed/tokenized/tokenizer.pkl`
- `data/processed/tokenized/token_bytes.pt`

#### Step 5: Training
```python
from chat_model.training import main
main()  # Train model using tokenized data
```

### Evaluation Metrics

#### Perplexity
- Measures how well model predicts validation set
- Lower is better
- Computed via cross-entropy loss

#### BLEU (Bilingual Evaluation Understudy)
- Compares generated text to reference MedQuAD answers
- N-gram overlap based
- 1-100 scale, higher is better

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- F1 score of N-gram overlap
- Captures semantic similarity
- Better for abstractive tasks

#### Safety Checks
- Rule-based validation for medical domain
- Ensures responses follow safety guidelines
- Prevents dangerous medical advice

---

## Configuration System

### Environment Variables (`.env`)

```bash
# Optional environment-specific overrides
# Copy from .env.example and customize

CUDA_VISIBLE_DEVICES=0          # GPU selection
WANDB_PROJECT=nanochat-medical  # Weights & Biases
WANDB_ENTITY=your-team          # WandB entity
LOG_LEVEL=INFO                  # Logging level
```

### Modifying Configuration

#### Option 1: Edit `config.py` (Permanent)
```python
# src/chat_model/config.py
BATCH_SIZE = 32  # Change default
LEARNING_RATE = 1e-4  # Adjust learning rate
```

#### Option 2: Environment Variables (Runtime)
```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py
```

#### Option 3: Command-line Arguments (Script-specific)
```bash
python scripts/train.py --batch-size 32 --lr 1e-4
```

---

## Data Pipeline

### Input Data Format

#### JSONL Format (After Preprocessing)
```json
{
  "conversation": [
    {"role": "user", "content": "What is diabetes?"},
    {"role": "assistant", "content": "Diabetes is a metabolic disease..."},
    {"role": "user", "content": "How is it treated?"},
    {"role": "assistant", "content": "Treatment depends on type..."}
  ]
}
```

#### PyTorch Dataset Format
- Converts conversations to token sequences
- Creates fixed-size chunks (BLOCK_SIZE=1024 tokens)
- Returns (input_ids, labels) pairs for language modeling

### Data Statistics

```python
from chat_model.datasets import loader
stats = loader.get_stats()
# Returns counts for train/val/test splits
```

### Using Custom Data

# Load custom JSONL file
from chat_model.datasets import loader

custom_data = loader.load_split("train", data_dir="/path/to/custom/splits")
# custom_data is list of conversation dicts

# Create dataset
from chat_model.tokenizing import train_tokenizer
# First need tokenizer.pkl at expected location
tokenizer = pickle.load(open(config.TOKENIZER_PKL, 'rb'))

dataset = loader.build_dataset("train", tokenizer, data_dir="/path/to/custom/splits")
dataloader = loader.make_dataloader(dataset, batch_size=32)
```

---

## Running the Project

### Three Ways to Execute Scripts

#### Method 1: Entry Point Scripts (Recommended)
```bash
# From Project root
python scripts/train.py        # Train model
python scripts/eval.py         # Evaluate model
python scripts/run_modal.py    # Deploy to Modal
```

**Advantages:** 
- Simplest and clearest
- Automatically handles paths
- Works from any directory

#### Method 2: Python Module Syntax
```bash
# From Project root
python -m chat_model.training.train
python -m chat_model.evaluation.run_eval
```

**Advantages:**
- Works without setup scripts
- Clear module path
- Good for batch processing

#### Method 3: Direct Execution
```bash
# From Project root
python src/chat_model/training/train.py
python src/chat_model/evaluation/eval.py
```

**Advantages:**
- Explicit and transparent
- Works with IDE debuggers
- Good for development

### Training the Model

```bash
# Basic training
python scripts/train.py

# Training with specific config
python scripts/train.py --help     # See all options

# Or edit config.py for permanent changes
```

**What happens:**
1. Loads data splits from `data/processed/splits/`
2. Loads tokenizer from `data/processed/tokenized/tokenizer.pkl`
3. Creates train and validation dataloaders
4. Initializes model and optimizer
5. Trains for EPOCHS (default: 10)
6. Saves best checkpoint to `data/checkpoints/best.pth`
7. Prints loss and metrics every batch

**Expected output:**
```
Epoch 1/10 | Batch 100/1000 | Loss: 4.23 | Val Loss: 4.05
Epoch 1/10 | Batch 200/1000 | Loss: 3.85 | Val Loss: 3.92
...
Checkpoint saved: data/checkpoints/best.pth
```

### Running Evaluation

```bash
# Evaluate on validation set
python scripts/eval.py

# Evaluate specific checkpoint
python scripts/eval.py --checkpoint path/to/checkpoint.pth
```

**Metrics computed:**
- Perplexity on validation set
- BLEU against MedQuAD
- ROUGE-L similarity
- Safety rule compliance

### Hyperparameter Tuning

```bash
# Run hyperparameter search
python src/chat_model/training/hyperparameter_train.py \
  --num-trials 10 \
  --load-data \
  --init-tokenizer

# Generates leaderboard of trials
```

### Chatbot Interface

```bash
# Start web server
cd src/chat_model/chatbot
php -S localhost:9000

# Visit http://localhost:9000
# Type questions and get responses
```

---

## Development Workflow

### Project Layout for Development

```
Project/
├── src/chat_model/         ← Main code here
├── tests/                  ← Write tests here
├── scripts/                ← Add entry points here
├── data/                   ← Data goes here
└── pyproject.toml          ← Define dependencies here
```

### Adding a New Feature

#### Example: Adding a new evaluation metric

**Step 1: Create module**
```python
# src/chat_model/evaluation/custom_metric.py

def compute_custom_metric(predictions, references):
    """Compute custom evaluation metric."""
    # Implementation
    return score
```

**Step 2: Export in `__init__.py`**
```python
# src/chat_model/evaluation/__init__.py

from chat_model.evaluation.custom_metric import compute_custom_metric

__all__ = [..., "compute_custom_metric"]
```

**Step 3: Write test**
```python
# tests/test_evaluation.py

def test_custom_metric():
    preds = ["hello", "world"]
    refs = ["hello", "earth"]
    score = compute_custom_metric(preds, refs)
    assert 0 <= score <= 1
```

**Step 4: Use in evaluation**
```python
# src/chat_model/evaluation/run_eval.py

from chat_model.evaluation import compute_custom_metric
# Call in evaluation loop
```

### Import Patterns

#### Standard Imports
```python
# Good: Import from package API
from chat_model.training import main, save_checkpoint
from chat_model.model import NanoChat
from chat_model.datasets import load_split, build_dataset

# Avoid: Import internal implementation
from chat_model.training.train import main  # Still works but less clean
```

#### Type Hints
```python
from typing import List, Dict, Tuple, Optional
from torch import Tensor

def build_dataset(
    split: str,
    tokenizer,
    data_dir: Optional[str] = None,
    max_conversations: Optional[int] = None,
) -> Dataset:
    """Create PyTorch Dataset for given split."""
    pass
```

#### Relative vs Absolute Imports
```python
# In src/chat_model/training/train.py

# Absolute (preferred)
from chat_model.datasets import loader as dl
from chat_model.model.model import NanoChat

# Relative (less clear)
from ..data import loader as dl
from ..model.model import NanoChat
```

### Testing Strategy

#### Unit Tests
```python
# tests/test_data.py
def test_load_split():
    """Test data loading."""
    data = load_split("train")
    assert len(data) > 0
    assert "conversation" in data[0]
```

#### Integration Tests
```python
# tests/test_training.py
def test_full_training_pipeline():
    """Test complete training loop."""
    dataset = build_dataset("train", tokenizer)
    loader = make_dataloader(dataset)
    # Train for 1 step
    model = NanoChat(config)
    # ...
```

#### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data.py

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=src/chat_model tests/
```

### Debugging

#### Print Debugging
```python
from chat_model import config

print(f"PROJECT_ROOT: {config.PROJECT_ROOT}")
print(f"SPLITS_DIR: {config.SPLITS_DIR}")
print(f"Dataset size: {len(dataset)}")
```

#### Using IPython
```bash
# Interactive debugging
python -c "
from chat_model.training import main
import pdb; pdb.set_trace()
main()
"
```

#### IDE Debugging
```python
# Set breakpoint in PyCharm/VSCode
if __name__ == "__main__":
    from chat_model.training import main
    main()

# Then run with debugger (Ctrl+Shift+D in VSCode)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: `ModuleNotFoundError: No module named 'chat_model'`

**Cause:** Package not properly installed or src not in path

**Solutions:**
```bash
# Solution 1: Install package
pip install -e .

# Solution 2: Run from correct directory
cd Project
python scripts/train.py

# Solution 3: Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### Issue 2: `ModuleNotFoundError: No module named 'rustbpe'`

**Cause:** NanoChat dependencies not installed

**Solutions:**
```bash
# Solution 1: Install just rustbpe
pip install rustbpe>=0.1.0

# Solution 2: Install full nanochat
cd ../nanochat
pip install -e .

# Solution 3: Use uv
cd ../nanochat
uv sync --extra cpu
```

#### Issue 3: Data files not found

**Cause:** Data not downloaded or wrong path

**Solutions:**
```bash
# Download data
python -c "from chat_model.datasets import download_data; download_data()"

# Check paths
python -c "from chat_model import config; print(config.SPLITS_DIR)"

# Verify files exist
ls -la data/processed/splits/
```

#### Issue 4: CUDA out of memory

**Cause:** Batch size too large for GPU

**Solutions:**
```python
# In config.py, reduce:
BATCH_SIZE = 8  # Down from 16

# Or use CPU
python -c "
import torch
torch.cuda.is_available = lambda: False  # Force CPU
"

# Or limit GPU memory
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### Issue 5: Model loading fails

**Cause:** Wrong checkpoint path or format

**Solutions:**
```bash
# Verify checkpoint exists
ls -la data/checkpoints/

# Check checkpoint format
python -c "
import torch
checkpoint = torch.load('data/checkpoints/best.pth')
print(checkpoint.keys())
"

# Try latest checkpoint
python -c "
from chat_model.training import load_checkpoint
model = load_checkpoint('data/checkpoints/latest.pth')
"
```

### Getting Help

1. **Check SETUP.md** for installation issues
2. **Read docstrings** in module files
3. **Check Git history** for changes
4. **Look at tests** for usage examples
5. **Search existing issues** in repository

---

## Advanced Topics

### Distributed Training

For training on multiple GPUs:

```python
# src/chat_model/training/train.py with DistributedDataParallel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = NanoChat(config).to(rank)
model = DDP(model, device_ids=[rank])

# Use with torch.distributed.launch
```

```bash
# Launch on 4 GPUs
torch.distributed.launch --nproc_per_node=4 scripts/train.py
```

### Mixed Precision Training

Already implemented using `torch.amp.autocast_mode.autocast`:

```python
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler

scaler = GradScaler()
with autocast():
    loss = model(batch)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Model Export

```python
# Export to ONNX
import torch
model = NanoChat(config)
model.load_state_dict(torch.load("best.pth"))

dummy_input = torch.randint(0, 32768, (1, 1024))
torch.onnx.export(model, dummy_input, "model.onnx")
```

### Custom Datasets

```python
# Implement custom dataset
from torch.utils.data import Dataset

class CustomMedicalDataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        self.data = load_custom_data(path)
        self.tokenizer = tokenizer
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.data[idx])
        return torch.tensor(tokens[:self.block_size])
```

### Inference Optimization

```python
# Use KV cache for faster generation
from chat_model.model.model import KVCache

model.eval()
with torch.no_grad():
    kv_cache = KVCache(batch_size=1, max_seq_len=2048)
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        kv_cache=kv_cache
    )
```

### Monitoring with Weights & Biases

```python
# Track training with WandB
import wandb

wandb.init(project="nanochat-medical")
wandb.config.update(config.__dict__)

# Log metrics
wandb.log({"loss": loss, "val_loss": val_loss})

# Log model
wandb.save("checkpoint.pth")
```

### Multi-node Training

```bash
# Launch on 2 nodes, 4 GPUs each
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=1.2.3.4 \
    --master_port=29500 \
    scripts/train.py
```

---

## Performance Optimization

### Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    train_step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Bottleneck Analysis

```python
# Check what's slow
import time

# Data loading
t0 = time.time()
for batch in dataloader:
    pass
print(f"Data loading: {time.time() - t0:.2f}s")

# Model forward
t0 = time.time()
for batch in dataloader:
    output = model(batch)
print(f"Model forward: {time.time() - t0:.2f}s")

# Backward pass
t0 = time.time()
for batch in dataloader:
    loss = model(batch)
    loss.backward()
print(f"Backward pass: {time.time() - t0:.2f}s")
```

---

## Contributing

### Workflow

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes following code style
3. Write tests: `pytest tests/`
4. Format code: `black src/ tests/`
5. Check types: `mypy src/chat_model/`
6. Commit: `git commit -m "Add feature"`
7. Push: `git push origin feature/my-feature`
8. Create pull request

### Code Style

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/chat_model/
```

### Testing Requirements

- Write tests for new features
- Maintain >80% code coverage
- All tests must pass before merge

---

## License

MIT License - See LICENSE file

---

## Additional Resources

### Documentation
- [SETUP.md](SETUP.md) - Installation and setup guide
- [RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md) - Project restructuring details

### External Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Python Packaging Guide](https://packaging.python.org/)
- [NanoChat Repository](https://github.com/karpathy/nanochat)

### Key Papers
- Attention Is All You Need (Transformer)
- RoFormer: Enhanced Transformer with Rotary Position Embedding
- Language Models are Unsupervised Multitask Learners (GPT-2)

---

## Contact & Support

For issues, questions, or suggestions:
1. Check existing documentation
2. Review code comments
3. Check test files for usage examples
4. Consult troubleshooting section above

---

**Last Updated:** May 2026  
**Project Status:** ✅ Production Ready  
**Python Version:** 3.10+  
**Maintainer:** Advanced Machine Learning Group, KTH
