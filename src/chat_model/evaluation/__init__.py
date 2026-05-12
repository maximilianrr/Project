"""Evaluation module for model metrics computation."""

from chat_model.evaluation.eval import (
    ESCALATION_PHRASES,
    GenerationMetrics,
    HIGH_RISK_SYMPTOMS,
    SafetyReport,
    UNSAFE_PATTERNS,
    compute_bleu_rouge,
    compute_perplexity,
    load_test_cases,
    run_full_eval,
    safety_eval,
)

__all__ = [
    "ESCALATION_PHRASES",
    "GenerationMetrics",
    "HIGH_RISK_SYMPTOMS",
    "SafetyReport",
    "UNSAFE_PATTERNS",
    "compute_bleu_rouge",
    "compute_perplexity",
    "load_test_cases",
    "run_full_eval",
    "safety_eval",
]