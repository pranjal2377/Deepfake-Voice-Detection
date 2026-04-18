#!/usr/bin/env python3
"""
Download the DynamicSuperb DEEP-VOICE deepfake audio dataset from HuggingFace
and save it as .wav files into the project's data/raw/ directory.

Usage:
    python scripts/download_dataset.py

This script:
  1. Reads cached parquet files (downloaded by HuggingFace datasets library)
  2. Extracts audio bytes and labels (real / fake)
  3. Saves audio files into data/raw/training/{real,fake}/ and data/raw/testing/{real,fake}/
  4. Uses an 80/20 train/test split
"""

import sys
import os
import random
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

from src.utils.config import RAW_DATA_DIR

# Parquet file locations (cached by HuggingFace)
CACHE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--DynamicSuperb--DeepFakeVoiceRecognition_DEEP-VOICE"
    "/snapshots/98c0615ac69481968268550f6a556c099671517a/data"
)


def ensure_dataset_downloaded():
    """Download the dataset if not already cached."""
    parquet_files = []
    if os.path.isdir(CACHE_DIR):
        parquet_files = sorted(
            [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]
        )

    if not parquet_files:
        print("📥 Dataset not in cache. Downloading from HuggingFace...")
        try:
            from datasets import load_dataset
            # This downloads parquet files to the HF cache
            load_dataset(
                "DynamicSuperb/DeepFakeVoiceRecognition_DEEP-VOICE",
                split="test",
            )
            print("✅ Download complete. Re-scanning cache...")
            parquet_files = sorted(
                [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]
            )
        except Exception as e:
            print(f"❌ Failed to download: {e}")
            sys.exit(1)

    return parquet_files


def main():
    print("=" * 60)
    print("  Deepfake Voice Detection — Dataset Download")
    print("  Source: DynamicSuperb/DeepFakeVoiceRecognition_DEEP-VOICE")
    print("=" * 60)

    # Step 1: Ensure parquet files exist
    parquet_files = ensure_dataset_downloaded()
    print(f"\n📦 Found {len(parquet_files)} parquet file(s)")

    # Step 2: Read all parquet files
    print("📖 Reading audio data from parquet files...")
    all_real = []  # list of (audio_bytes, original_filename)
    all_fake = []

    for pf in parquet_files:
        table = pq.read_table(pf)
        n_rows = len(table)
        labels = table.column("label").to_pylist()
        audios = table.column("audio").to_pylist()

        for i in range(n_rows):
            label = labels[i].strip().lower()
            audio_dict = audios[i]
            audio_bytes = audio_dict["bytes"]
            orig_name = audio_dict.get("path", f"sample_{i}.wav")

            if label == "real":
                all_real.append((audio_bytes, orig_name))
            elif label == "fake":
                all_fake.append((audio_bytes, orig_name))
            else:
                print(f"   ⚠️  Unknown label '{label}' for {orig_name} — skipping")

        print(f"   Processed {pf.split('/')[-1]}: {n_rows} rows")

    print(f"\n📊 Total: {len(all_real)} real + {len(all_fake)} fake = {len(all_real)+len(all_fake)} samples")

    if len(all_real) == 0 or len(all_fake) == 0:
        print("❌ Dataset has no samples for one class. Aborting.")
        sys.exit(1)

    # Step 3: Clear old data
    print(f"\n🗑️  Clearing old data from {RAW_DATA_DIR}...")
    for subset_dir in ["training", "testing"]:
        for label_dir in ["real", "fake"]:
            dirpath = os.path.join(RAW_DATA_DIR, subset_dir, label_dir)
            if os.path.exists(dirpath):
                for fname in os.listdir(dirpath):
                    if fname.endswith(".wav"):
                        os.remove(os.path.join(dirpath, fname))
            os.makedirs(dirpath, exist_ok=True)
    print("   ✅ Old data cleared")

    # Step 4: Shuffle and split 80/20
    random.seed(42)
    random.shuffle(all_real)
    random.shuffle(all_fake)

    train_ratio = 0.8
    real_train_n = int(len(all_real) * train_ratio)
    fake_train_n = int(len(all_fake) * train_ratio)

    splits = {
        "training": {
            "real": all_real[:real_train_n],
            "fake": all_fake[:fake_train_n],
        },
        "testing": {
            "real": all_real[real_train_n:],
            "fake": all_fake[fake_train_n:],
        },
    }

    # Step 5: Save audio files
    print("\n💾 Saving audio files...")
    target_sr = 16000
    saved_count = 0

    for subset_name, labels in splits.items():
        for label_name, samples in labels.items():
            dirpath = os.path.join(RAW_DATA_DIR, subset_name, label_name)

            for j, (audio_bytes, orig_name) in enumerate(samples):
                try:
                    # Decode audio bytes
                    audio_data, sr = sf.read(io.BytesIO(audio_bytes))

                    # Convert to mono if stereo
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)

                    audio_data = audio_data.astype(np.float32)

                    # Resample to 16kHz if needed
                    if sr != target_sr:
                        import librosa
                        audio_data = librosa.resample(
                            audio_data, orig_sr=sr, target_sr=target_sr
                        )

                    # Normalize
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val

                    # Save
                    filename = f"{label_name}_{j:04d}.wav"
                    filepath = os.path.join(dirpath, filename)
                    sf.write(filepath, audio_data, target_sr)
                    saved_count += 1

                except Exception as e:
                    print(f"   ⚠️  Error processing {orig_name}: {e}")

            count = len([f for f in os.listdir(dirpath) if f.endswith(".wav")])
            print(f"   ✅ {subset_name}/{label_name}: {count} files saved")

    print(f"\n🎉 Total saved: {saved_count} audio files")

    # Step 6: Show final structure
    print(f"\n📁 Final dataset structure:")
    for subset_dir in ["training", "testing"]:
        for label_dir in ["real", "fake"]:
            dirpath = os.path.join(RAW_DATA_DIR, subset_dir, label_dir)
            count = len([f for f in os.listdir(dirpath) if f.endswith(".wav")])
            print(f"   {subset_dir}/{label_dir}: {count} files")

    print("\n✅ Dataset download complete!")
    print("   Next: run 'python scripts/prepare_dataset.py --force-resplit'")
    print("   Then: run 'python scripts/train.py'")


if __name__ == "__main__":
    main()
