"""Training utilities — trainer, metrics, checkpointing."""

from .trainer import Trainer
from .metrics import compute_metrics, compute_eer, compute_confusion, find_optimal_threshold

__all__ = [
    "Trainer",
    "compute_metrics",
    "compute_eer",
    "compute_confusion",
    "find_optimal_threshold",
]
