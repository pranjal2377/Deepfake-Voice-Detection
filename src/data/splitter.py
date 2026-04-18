"""
Data Splitting — Stratified train/val/test split.

If the dataset already has subset assignments (from directory structure),
those are preserved. Otherwise, files are split automatically with
stratification by label to maintain class balance.
"""

import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Default split ratios
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    force_resplit: bool = False,
) -> pd.DataFrame:
    """
    Split a manifest DataFrame into train/val/test subsets.

    If the DataFrame already has valid subset assignments and
    force_resplit=False, the existing splits are kept.

    Args:
        df: Manifest DataFrame with 'path', 'label', 'subset' columns
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        seed: Random seed for reproducibility
        force_resplit: Ignore existing subsets and re-split

    Returns:
        DataFrame with updated 'subset' column
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    df = df.copy()

    # Check if subsets are already assigned
    if not force_resplit:
        assigned = df[df["subset"].isin(["train", "val", "test"])]
        if len(assigned) == len(df) and len(df) > 0:
            logger.info("Dataset already has valid subset assignments. Skipping split.")
            _log_split_stats(df)
            return df

        # If dataset has train/test from directory but no val, split train into train+val
        if set(df["subset"].unique()) >= {"train", "test"} and "val" not in df["subset"].unique():
            logger.info("Found train/test splits. Creating val split from training data.")
            return _split_train_into_train_val(df, val_ratio, seed)

    # Full re-split from scratch
    logger.info(f"Splitting dataset: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    if len(df) == 0:
        return df

    # Stratified split
    labels = df["label"].values

    # First split: train+val vs test
    trainval_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: train vs val
    trainval_labels = labels[trainval_idx]
    relative_val = val_ratio / (train_ratio + val_ratio)

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val,
        stratify=trainval_labels,
        random_state=seed,
    )

    df.iloc[train_idx, df.columns.get_loc("subset")] = "train"
    df.iloc[val_idx, df.columns.get_loc("subset")] = "val"
    df.iloc[test_idx, df.columns.get_loc("subset")] = "test"

    _log_split_stats(df)
    return df


def _split_train_into_train_val(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Split existing train subset into train + val."""
    df = df.copy()
    train_mask = df["subset"] == "train"
    train_df = df[train_mask]

    if len(train_df) < 4:  # too few to split
        logger.warning("Too few training samples to create validation split")
        return df

    labels = train_df["label"].values
    train_idx, val_idx = train_test_split(
        train_df.index,
        test_size=val_ratio / (1.0 - val_ratio + val_ratio),  # relative to train
        stratify=labels,
        random_state=seed,
    )

    df.loc[val_idx, "subset"] = "val"

    _log_split_stats(df)
    return df


def _log_split_stats(df: pd.DataFrame):
    """Log split statistics."""
    for subset in ["train", "val", "test"]:
        sub = df[df["subset"] == subset]
        if len(sub) == 0:
            continue
        real = (sub["label"] == 0).sum()
        fake = (sub["label"] == 1).sum()
        logger.info(f"  {subset:5s}: {len(sub):5d} files (real={real}, fake={fake})")


def get_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Get a specific subset from the manifest."""
    result = df[df["subset"] == subset].copy()
    logger.info(f"Retrieved {subset} subset: {len(result)} samples")
    return result
