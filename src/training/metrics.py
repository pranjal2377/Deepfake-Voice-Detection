"""
Training Metrics — Computes classification metrics for deepfake detection.

Provides per-epoch metrics (accuracy, precision, recall, F1, AUC) and
the Equal Error Rate (EER) which is the standard metric for voice
anti-spoofing systems.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (0=real, 1=fake)
        y_pred_prob: Predicted probabilities of being fake
        threshold: Decision threshold

    Returns:
        Dict with accuracy, precision, recall, f1, auc, eer
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC (requires both classes present)
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_pred_prob)
        metrics["eer"] = compute_eer(y_true, y_pred_prob)
    else:
        metrics["auc"] = 0.0
        metrics["eer"] = 1.0

    return metrics


def compute_eer(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
) -> float:
    """
    Compute Equal Error Rate (EER).

    EER is the point where False Acceptance Rate = False Rejection Rate.
    It's the standard metric for speaker verification / anti-spoofing.

    Lower EER = better model.

    Args:
        y_true: Ground truth labels (0=real, 1=fake)
        y_pred_prob: Predicted probabilities of being fake

    Returns:
        EER as a float (0 to 1)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    fnr = 1 - tpr

    # Find the point where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2

    return float(eer)


def compute_confusion(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Compute confusion matrix values.

    Returns:
        Dict with TP, TN, FP, FN counts
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "TN": int(cm[0, 0]),  # correctly identified real
        "FP": int(cm[0, 1]),  # real misclassified as fake
        "FN": int(cm[1, 0]),  # fake misclassified as real
        "TP": int(cm[1, 1]),  # correctly identified fake
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes F1 score.

    Returns:
        (optimal_threshold, best_f1_score)
    """
    best_f1 = 0.0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 0.95, 0.01):
        y_pred = (y_pred_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh), float(best_f1)
