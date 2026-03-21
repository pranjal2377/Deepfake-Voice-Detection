#!/usr/bin/env python3
"""
Model Evaluation Script — Phase 4 CLI.

Evaluates a trained model on the test split or a custom directory,
and generates detailed reports.

Usage:
    # Evaluate on test split (default):
    python scripts/evaluate.py

    # Evaluate with a custom model checkpoint:
    python scripts/evaluate.py --model models/best_model.pt

    # Evaluate on a custom directory (cross-dataset):
    python scripts/evaluate.py --data-dir /path/to/test/audio

    # Set custom threshold:
    python scripts/evaluate.py --threshold 0.6

    # Save report to specific directory:
    python scripts/evaluate.py --output-dir reports/
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import (
    ModelEvaluator,
    format_report,
    save_report,
    load_model_for_evaluation,
)
from src.data.dataset_loader import load_manifest, get_dataset_stats
from src.data.splitter import get_subset
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument(
        "--model",
        default=os.path.join(MODELS_DIR, "best_model.pt"),
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with audio files to evaluate (cross-dataset mode). "
             "If not provided, evaluates on the test split of the manifest.",
    )
    parser.add_argument(
        "--subset",
        default="test",
        choices=["train", "val", "test"],
        help="Which manifest subset to evaluate (default: test)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save reports (default: models/evaluation/)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use: cpu or cuda (auto-detect if omitted)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(MODELS_DIR, "evaluation")

    print("=" * 60)
    print("  Deepfake Voice Detection — Model Evaluation")
    print("=" * 60)

    # Load model
    print(f"\n🧠 Loading model from: {args.model}")
    model = load_model_for_evaluation(args.model)

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=args.device,
        threshold=args.threshold,
    )

    # Evaluate
    if args.data_dir:
        # Cross-dataset evaluation
        print(f"\n📂 Evaluating on directory: {args.data_dir}")
        report = evaluator.evaluate_directory(args.data_dir)
    else:
        # Manifest-based evaluation
        print(f"\n📊 Loading manifest...")
        df = load_manifest()
        test_df = get_subset(df, args.subset)

        if len(test_df) == 0:
            print(f"\n⚠️  No samples in '{args.subset}' subset!")
            print("   Run `python scripts/prepare_dataset.py --force-resplit` first.")
            return

        print(f"   Evaluating on {len(test_df)} {args.subset} samples")
        report = evaluator.evaluate_manifest(test_df)

    # Display report
    print("\n" + format_report(report))

    # Save reports
    text_path, json_path = save_report(report, output_dir)
    print(f"\n💾 Reports saved:")
    print(f"   Text:  {text_path}")
    print(f"   JSON:  {json_path}")

    csv_path = os.path.join(output_dir, "evaluation_per_sample.csv")
    if os.path.exists(csv_path):
        print(f"   CSV:   {csv_path}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
