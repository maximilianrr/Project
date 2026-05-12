#!/usr/bin/env python3
"""
Evaluation script for the Chat Model.

Usage:
    python scripts/eval.py [--model-path MODEL_PATH]
    
Or from project root:
    python -m chat_model.evaluation.run_eval
"""

import sys
import os

# Add src to path so chat_model can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from chat_model.evaluation.run_eval import main

if __name__ == "__main__":
    main()
