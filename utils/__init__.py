"""Utility package for scoring, explanations, and validation."""
from .scoring import calculate_dynamic_weights, calculate_fabric_scores
from .explanations import generate_fabric_explanation, generate_tradeoff_analysis
from .validation import validate_recommendation_inputs

__all__ = [
    "calculate_dynamic_weights",
    "calculate_fabric_scores",
    "generate_fabric_explanation",
    "generate_tradeoff_analysis",
    "validate_recommendation_inputs"
]
