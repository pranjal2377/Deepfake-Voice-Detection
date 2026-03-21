"""
Model Evaluator — Comprehensive evaluation of the deepfake detection model.

Provides:
  - Per-sample inference with probabilities
  - Aggregate metrics (accuracy, precision, recall, F1, AUC, EER)
  - Confusion matrix computation
  - Per-class analysis (real vs fake breakdown)
  - Threshold calibration
  - Text and JSON report generation
  - Support for evaluating on arbitrary directories (cross-dataset)
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.audio.preprocessor import load_audio, normalize_audio, trim_silence
from src.features.extractor import features_to_model_input
from src.data.augmentation import random_crop
from src.training.metrics import (
    compute_metrics,
    compute_eer,
    compute_confusion,
    find_optimal_threshold,
)
from src.utils.config import SAMPLE_RATE, FRAME_DURATION, N_MELS, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates a trained DeepfakeCNN model on a dataset.

    Supports evaluation on:
      1. A manifest DataFrame (from the train/val/test split)
      2. An arbitrary directory of audio files (cross-dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            model: Trained DeepfakeCNN model
            device: 'cpu' or 'cuda' (auto-detect if None)
            threshold: Decision threshold for binary classification
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.sr = SAMPLE_RATE
        self.target_length = int(SAMPLE_RATE * FRAME_DURATION)

    @torch.no_grad()
    def predict_file(self, file_path: str) -> Dict:
        """
        Run inference on a single audio file.
        Now includes NLP scam detection.

        Returns:
            Dict with probability, predicted_label, predicted_name
        """
        from src.detection.detector import RealtimeDetector
        try:
            # We can use the Detector to get both NLP and Audio scores
            # But creating a new Detector for each file is slow.
            # However, evaluating full audio with transcription is exactly what the detector does
            
            audio, sr = load_audio(file_path, sr=self.sr)
            audio = normalize_audio(audio)
            audio = trim_silence(audio, sr)
            audio = random_crop(audio, self.target_length)
            
            # 1. Voice probability
            mel = features_to_model_input(audio, sr)
            tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)

            prob = self.model(tensor).squeeze().item()
            
            # 2. NLP probability (simulate by transcribing the audio crop)
            # Actually, to integrate NLP efficiently without breaking existing structure,
            # we just do a simple pass if Transcriber is available
            try:
                from src.nlp.transcriber import AudioTranscriber
                from src.nlp.scam_detector import ScamDetector
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    transcriber = AudioTranscriber()
                    scam_detector = ScamDetector()
                    
                    text = transcriber.transcribe(audio, sr)
                    nlp_res = scam_detector.analyze_transcript(text)
                    nlp_prob = nlp_res["nlp_probability"]
            except Exception as e:
                logger.debug(f"NLP eval fail: {e}")
                nlp_prob = 0.0
            
            predicted_label = 1 if prob >= self.threshold else 0
            nlp_label = 1 if nlp_prob >= self.threshold else 0
            combined_prob = max(prob, nlp_prob)
            combined_label = 1 if combined_prob >= self.threshold else 0

            return {
                "probability": prob,
                "nlp_probability": nlp_prob,
                "combined_probability": combined_prob,
                "predicted_label": predicted_label,
                "nlp_label": nlp_label,
                "combined_label": combined_label,
                "predicted_name": "fake" if predicted_label == 1 else "real",
                "error": None,
            }
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            return {
                "probability": 0.5,
                "nlp_probability": 0.0,
                "combined_probability": 0.5,
                "predicted_label": -1,
                "nlp_label": -1,
                "combined_label": -1,
                "predicted_name": "error",
                "error": str(e),
            }

    def evaluate_manifest(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate the model on a manifest DataFrame.

        Args:
            df: DataFrame with 'path' and 'label' columns
            verbose: Show progress bar

        Returns:
            Dict with metrics, per_sample results, confusion, report text
        """
        start_time = time.time()

        all_probs = []
        all_nlp_probs = []
        all_combined_probs = []
        all_labels = []
        per_sample = []
        errors = 0

        iterator = tqdm(df.iterrows(), total=len(df), desc="Evaluating") if verbose else df.iterrows()
        for _, row in iterator:
            result = self.predict_file(row["path"])
            true_label = int(row["label"])

            if result["error"] is not None:
                errors += 1
                continue

            all_probs.append(result["probability"])
            all_nlp_probs.append(result.get("nlp_probability", 0.0))
            all_combined_probs.append(result.get("combined_probability", result["probability"]))
            all_labels.append(true_label)
            per_sample.append({
                "path": row["path"],
                "filename": os.path.basename(row["path"]),
                "true_label": true_label,
                "true_name": row.get("label_name", "unknown"),
                "predicted_label": result["predicted_label"],
                "predicted_name": result["predicted_name"],
                "probability": result["probability"],
                "nlp_probability": result.get("nlp_probability", 0.0),
                "combined_probability": result.get("combined_probability", result["probability"]),
                # Combined correct check
                "correct": result.get("combined_label", result["predicted_label"]) == true_label,
            })

        elapsed = time.time() - start_time
        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_nlp_prob = np.array(all_nlp_probs)
        y_comb_prob = np.array(all_combined_probs)

        # Compute metrics
        metrics = compute_metrics(y_true, y_prob, threshold=self.threshold)
        nlp_metrics = compute_metrics(y_true, y_nlp_prob, threshold=self.threshold)
        comb_metrics = compute_metrics(y_true, y_comb_prob, threshold=self.threshold)
        
        confusion = compute_confusion(y_true, y_prob, threshold=self.threshold)
        nlp_confusion = compute_confusion(y_true, y_nlp_prob, threshold=self.threshold)
        comb_confusion = compute_confusion(y_true, y_comb_prob, threshold=self.threshold)
        
        opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_prob)

        # Per-class breakdown
        per_class = self._per_class_analysis(y_true, y_prob)
        nlp_per_class = self._per_class_analysis(y_true, y_nlp_prob)
        comb_per_class = self._per_class_analysis(y_true, y_comb_prob)

        # Misclassified samples
        misclassified = [s for s in per_sample if not s["correct"]]

        report = {
            "metrics": metrics,
            "nlp_metrics": nlp_metrics,
            "comb_metrics": comb_metrics,
            "confusion": confusion,
            "nlp_confusion": nlp_confusion,
            "comb_confusion": comb_confusion,
            "optimal_threshold": opt_thresh,
            "optimal_f1": opt_f1,
            "per_class": per_class,
            "nlp_per_class": nlp_per_class,
            "comb_per_class": comb_per_class,
            "per_sample": per_sample,
            "misclassified": misclassified,
            "total_samples": len(y_true),
            "errors": errors,
            "threshold_used": self.threshold,
            "evaluation_time_seconds": elapsed,
        }

        return report

    def evaluate_directory(
        self,
        directory: str,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate on an arbitrary directory of audio files (cross-dataset).
        Uses directory structure / filenames to infer labels.

        Args:
            directory: Path to directory containing audio files

        Returns:
            Same format as evaluate_manifest
        """
        from src.data.dataset_loader import discover_files

        entries = discover_files(directory)
        if not entries:
            logger.warning(f"No audio files found in {directory}")
            return {"metrics": {}, "total_samples": 0, "errors": 0}

        df = pd.DataFrame(entries)
        logger.info(f"Found {len(df)} files in {directory}")

        return self.evaluate_manifest(df, verbose=verbose)

    def _per_class_analysis(
        self, y_true: np.ndarray, y_prob: np.ndarray,
    ) -> Dict:
        """Compute metrics separately for each class."""
        result = {}

        for label, name in [(0, "real"), (1, "fake")]:
            mask = y_true == label
            if mask.sum() == 0:
                continue

            class_probs = y_prob[mask]
            class_labels = y_true[mask]

            # For real samples: correct = predicted real (prob < threshold)
            # For fake samples: correct = predicted fake (prob >= threshold)
            if label == 0:
                correct = (class_probs < self.threshold).sum()
            else:
                correct = (class_probs >= self.threshold).sum()

            result[name] = {
                "total": int(mask.sum()),
                "correct": int(correct),
                "accuracy": float(correct / mask.sum()),
                "avg_probability": float(class_probs.mean()),
                "std_probability": float(class_probs.std()),
                "min_probability": float(class_probs.min()),
                "max_probability": float(class_probs.max()),
            }

        return result


def format_report(report: Dict) -> str:
    """Format an evaluation report as a readable text string."""
    lines = []
    lines.append("=" * 60)
    lines.append("  DEEPFAKE VOICE DETECTION — EVALUATION REPORT")
    lines.append("=" * 60)

    m = report.get("metrics", {})
    lines.append(f"\n  Total Samples:      {report.get('total_samples', 0)}")
    lines.append(f"  Errors:             {report.get('errors', 0)}")
    lines.append(f"  Threshold Used:     {report.get('threshold_used', 0.5):.2f}")
    lines.append(f"  Evaluation Time:    {report.get('evaluation_time_seconds', 0):.1f}s")

    lines.append(f"\n  ── Metrics ──")
    lines.append(f"  Accuracy:           {m.get('accuracy', 0):.4f}")
    lines.append(f"  Precision:          {m.get('precision', 0):.4f}")
    lines.append(f"  Recall:             {m.get('recall', 0):.4f}")
    lines.append(f"  F1 Score:           {m.get('f1', 0):.4f}")
    lines.append(f"  AUC-ROC:            {m.get('auc', 0):.4f}")
    lines.append(f"  EER:                {m.get('eer', 0):.4f}")

    lines.append(f"\n  Optimal Threshold:  {report.get('optimal_threshold', 0.5):.2f}")
    lines.append(f"  Optimal F1:         {report.get('optimal_f1', 0):.4f}")

    cm = report.get("confusion", {})
    lines.append(f"\n  ── Confusion Matrix ──")
    lines.append(f"                    Predicted")
    lines.append(f"                  Real    Fake")
    lines.append(f"  Actual Real   {cm.get('TN', 0):5d}   {cm.get('FP', 0):5d}")
    lines.append(f"  Actual Fake   {cm.get('FN', 0):5d}   {cm.get('TP', 0):5d}")

    pc = report.get("per_class", {})
    if pc:
        lines.append(f"\n  ── Per-Class Breakdown ──")
        for name, stats in pc.items():
            lines.append(f"\n  {name.upper()}:")
            lines.append(f"    Samples:   {stats['total']}")
            lines.append(f"    Correct:   {stats['correct']} ({stats['accuracy']:.1%})")
            lines.append(f"    Avg prob:  {stats['avg_probability']:.4f} "
                         f"± {stats['std_probability']:.4f}")
            lines.append(f"    Range:     [{stats['min_probability']:.4f}, "
                         f"{stats['max_probability']:.4f}]")

    mis = report.get("misclassified", [])
    if mis:
        lines.append(f"\n  ── Misclassified Samples ({len(mis)}) ──")
        for s in mis[:20]:  # show at most 20
            lines.append(
                f"    {s['filename']:30s}  "
                f"true={s['true_name']:4s}  pred={s['predicted_name']:4s}  "
                f"prob={s['probability']:.4f}"
            )
        if len(mis) > 20:
            lines.append(f"    ... and {len(mis) - 20} more")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def save_report(report: Dict, output_dir: str) -> Tuple[str, str]:
    """
    Save evaluation report as both text and JSON files.

    Returns:
        (text_path, json_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Text report
    text_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(text_path, "w") as f:
        f.write(format_report(report))

    # JSON report (strip per_sample for compact version)
    json_report = {k: v for k, v in report.items() if k != "per_sample"}
    # Convert numpy types to native Python types
    json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=_json_serializer)

    # Full per-sample CSV
    if report.get("per_sample"):
        csv_path = os.path.join(output_dir, "evaluation_per_sample.csv")
        pd.DataFrame(report["per_sample"]).to_csv(csv_path, index=False)

    logger.info(f"Reports saved to {output_dir}")
    return text_path, json_path


def _json_serializer(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_model_for_evaluation(
    checkpoint_path: str = None,
) -> nn.Module:
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint (defaults to models/best_model.pt)

    Returns:
        Loaded DeepfakeCNN model in eval mode
    """
    from src.model.cnn_model import DeepfakeCNN

    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODELS_DIR, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model = DeepfakeCNN()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return model
