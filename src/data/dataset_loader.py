"""
Dataset Loader — Discovers and catalogs audio files with labels.

Supports two dataset structures:

1. **Fake-or-Real (for-original / for-norm)** — Kaggle:
   data/raw/
     ├── training/
     │   ├── real/
     │   └── fake/
     └── testing/
         ├── real/
         └── fake/

2. **Flat directory with naming convention**:
   data/raw/
     ├── real_001.wav
     ├── fake_001.wav
     └── ...

Labels: 0 = real (genuine human voice), 1 = fake (deepfake / synthetic)
"""

import os
import glob
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import pandas as pd

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, get_config

logger = logging.getLogger(__name__)

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


def discover_files(root_dir: str) -> List[Dict]:
    """
    Recursively discover audio files under root_dir and infer labels.

    Labeling heuristics (tried in order):
      1. Parent folder named 'real' or 'fake' → label from folder name
      2. Filename starts with 'real_' or 'fake_' → label from prefix
      3. Filename contains 'real' or 'fake' → label from substring
      4. Otherwise → label = -1 (unknown, filtered out)

    Returns:
        List of dicts with keys: path, label, subset, filename
    """
    entries = []
    root = Path(root_dir)

    if not root.exists():
        logger.warning(f"Dataset directory does not exist: {root_dir}")
        return entries

    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        label = _infer_label(fpath)
        subset = _infer_subset(fpath, root)

        entries.append({
            "path": str(fpath),
            "label": label,
            "label_name": {0: "real", 1: "fake"}.get(label, "unknown"),
            "subset": subset,
            "filename": fpath.name,
        })

    # Filter out unknowns
    known = [e for e in entries if e["label"] != -1]
    unknown = len(entries) - len(known)
    if unknown > 0:
        logger.warning(f"Skipped {unknown} files with unknown labels")

    logger.info(
        f"Discovered {len(known)} audio files "
        f"({sum(1 for e in known if e['label']==0)} real, "
        f"{sum(1 for e in known if e['label']==1)} fake)"
    )
    return known


def _infer_label(fpath: Path) -> int:
    """Infer label from directory structure or filename."""
    # Check parent folder names
    parts = [p.lower() for p in fpath.parts]

    for part in reversed(parts[:-1]):  # skip filename itself
        if part in ("real", "genuine", "bonafide", "bona-fide"):
            return 0
        if part in ("fake", "spoof", "synthetic", "deepfake", "generated"):
            return 1

    # Check filename prefix / content
    fname = fpath.stem.lower()
    if fname.startswith("real") or fname.startswith("genuine"):
        return 0
    if fname.startswith("fake") or fname.startswith("spoof"):
        return 1
    if "real" in fname and "fake" not in fname:
        return 0
    if "fake" in fname and "real" not in fname:
        return 1

    return -1  # unknown


def _infer_subset(fpath: Path, root: Path) -> str:
    """Infer train/test/val subset from directory structure."""
    parts = [p.lower() for p in fpath.relative_to(root).parts]

    for part in parts:
        if part in ("train", "training"):
            return "train"
        if part in ("val", "validation", "valid", "dev"):
            return "val"
        if part in ("test", "testing", "eval", "evaluation"):
            return "test"

    return "unassigned"


def create_manifest(
    root_dir: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Discover all audio files and save a CSV manifest.

    Args:
        root_dir: Root directory to scan (defaults to RAW_DATA_DIR)
        output_path: Where to save the CSV (defaults to data/processed/manifest.csv)

    Returns:
        DataFrame with columns: path, label, label_name, subset, filename
    """
    root_dir = root_dir or RAW_DATA_DIR
    output_path = output_path or os.path.join(PROCESSED_DATA_DIR, "manifest.csv")

    entries = discover_files(root_dir)

    if len(entries) == 0:
        logger.warning("No audio files found. Creating empty manifest.")
        df = pd.DataFrame(columns=["path", "label", "label_name", "subset", "filename"])
    else:
        df = pd.DataFrame(entries)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Manifest saved to {output_path} ({len(df)} entries)")

    return df


def load_manifest(path: str = None) -> pd.DataFrame:
    """Load a previously saved manifest CSV."""
    path = path or os.path.join(PROCESSED_DATA_DIR, "manifest.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest not found at {path}. "
            "Run `python scripts/prepare_dataset.py` first."
        )

    df = pd.read_csv(path)
    logger.info(f"Loaded manifest with {len(df)} entries from {path}")
    return df


def get_dataset_stats(df: pd.DataFrame) -> Dict:
    """Compute summary statistics for a manifest DataFrame."""
    stats = {
        "total_files": len(df),
        "label_distribution": df["label_name"].value_counts().to_dict(),
        "subset_distribution": df["subset"].value_counts().to_dict(),
        "label_by_subset": {},
    }

    for subset in df["subset"].unique():
        sub = df[df["subset"] == subset]
        stats["label_by_subset"][subset] = sub["label_name"].value_counts().to_dict()

    return stats
