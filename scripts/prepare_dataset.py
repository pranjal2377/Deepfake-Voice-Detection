#!/usr/bin/env python3
"""
Dataset Preparation Script — Phase 2 CLI.

Discovers audio files, creates a manifest, splits into train/val/test,
and prints dataset statistics.

Usage:
    # From project root:
    python scripts/prepare_dataset.py

    # Custom data directory:
    python scripts/prepare_dataset.py --data-dir /path/to/audio/files

    # Force re-split:
    python scripts/prepare_dataset.py --force-resplit

    # Generate synthetic test data for smoke testing:
    python scripts/prepare_dataset.py --generate-demo
"""

import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import create_manifest, get_dataset_stats
from src.data.splitter import split_dataset
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_demo_data(output_dir: str, n_per_class: int = 10):
    """
    Generate synthetic WAV files for testing the pipeline.
    Creates simple sine waves for 'real' and noise for 'fake'.
    """
    import numpy as np
    import soundfile as sf

    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)

    for subset in ["training", "testing"]:
        for label in ["real", "fake"]:
            dirpath = os.path.join(output_dir, subset, label)
            os.makedirs(dirpath, exist_ok=True)

            n = n_per_class if subset == "training" else max(3, n_per_class // 3)

            for i in range(n):
                if label == "real":
                    # Simulate speech-like signal: sum of harmonics
                    t = np.linspace(0, duration, n_samples, dtype=np.float32)
                    freq = 150 + np.random.uniform(-30, 30)
                    audio = (
                        0.5 * np.sin(2 * np.pi * freq * t)
                        + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
                        + 0.1 * np.sin(2 * np.pi * 3 * freq * t)
                        + 0.05 * np.random.randn(n_samples)
                    ).astype(np.float32)
                else:
                    # Simulate synthetic / robotic signal
                    t = np.linspace(0, duration, n_samples, dtype=np.float32)
                    freq = 200 + np.random.uniform(-20, 20)
                    audio = (
                        0.7 * np.sin(2 * np.pi * freq * t)
                        + 0.2 * np.random.randn(n_samples)
                    ).astype(np.float32)

                # Normalize
                audio = audio / (np.max(np.abs(audio)) + 1e-10)

                filepath = os.path.join(dirpath, f"{label}_{i:04d}.wav")
                sf.write(filepath, audio, sr)

    logger.info(f"Generated demo data in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--data-dir",
        default=RAW_DATA_DIR,
        help=f"Root directory containing audio files (default: {RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROCESSED_DATA_DIR, "manifest.csv"),
        help="Output manifest CSV path",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Force re-split even if subsets already assigned",
    )
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate synthetic demo data for testing",
    )
    parser.add_argument(
        "--demo-samples",
        type=int,
        default=10,
        help="Number of samples per class for demo data (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Deepfake Voice Detection — Dataset Preparation")
    print("=" * 60)

    # Optionally generate demo data
    if args.generate_demo:
        print("\n📦 Generating demo data...")
        generate_demo_data(args.data_dir, n_per_class=args.demo_samples)

    # Step 1: Discover files and create manifest
    print(f"\n🔍 Scanning: {args.data_dir}")
    df = create_manifest(root_dir=args.data_dir, output_path=args.output)

    if len(df) == 0:
        print("\n⚠️  No audio files found!")
        print(f"   Place your audio files in: {args.data_dir}")
        print("   Expected structure:")
        print("     data/raw/training/real/*.wav")
        print("     data/raw/training/fake/*.wav")
        print("     data/raw/testing/real/*.wav")
        print("     data/raw/testing/fake/*.wav")
        print("\n   Or run with --generate-demo to create test data.")
        return

    # Step 2: Split dataset
    print("\n✂️  Splitting dataset...")
    df = split_dataset(df, force_resplit=args.force_resplit)

    # Save updated manifest
    df.to_csv(args.output, index=False)
    print(f"💾 Manifest saved to: {args.output}")

    # Step 3: Print statistics
    stats = get_dataset_stats(df)
    print("\n📊 Dataset Statistics:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Labels: {stats['label_distribution']}")
    print(f"   Subsets: {stats['subset_distribution']}")
    print("\n   Per subset:")
    for subset, dist in stats["label_by_subset"].items():
        print(f"     {subset}: {dist}")

    print("\n✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()
