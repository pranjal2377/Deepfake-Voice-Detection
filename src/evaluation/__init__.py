"""Evaluation utilities — model evaluation, reporting, cross-dataset testing."""

from .evaluator import ModelEvaluator, format_report, save_report, load_model_for_evaluation

__all__ = [
    "ModelEvaluator",
    "format_report",
    "save_report",
    "load_model_for_evaluation",
]
