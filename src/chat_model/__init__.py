"""Chat Model Package - Multi-modal conversational AI system.

A production-ready Python package for training and evaluating medical conversational AI models.
"""

__version__ = "0.1.0"
__author__ = "Advanced ML Group"

# Import main components for easier access
from chat_model import config
from chat_model.model import NanoChat

__all__ = [
    "config",
    "NanoChat",
]
