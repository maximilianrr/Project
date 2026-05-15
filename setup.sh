#!/bin/bash
# Installation script for the Chat Model project

set -e

echo "================================================"
echo "Chat Model Project Setup"
echo "================================================"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "1. Creating Python virtual environment..."
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    python3.12 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

echo ""
echo "2. Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "3. Installing project in development mode..."
pip install -e .

echo ""
echo "4. Installing development dependencies..."
pip install -e ".[dev]"

echo ""
echo "5. Installing base requirements..."
pip install -r requirements.txt

echo ""
echo "================================================"
echo "✓ Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run training:"
echo "     python scripts/train.py"
echo ""
echo "  3. Run evaluation:"
echo "     python scripts/eval.py"
echo ""
echo "  4. Or use Python module syntax:"
echo "     python -m chat_model.training.train"
echo "     python -m chat_model.evaluation.run_eval"
echo ""
