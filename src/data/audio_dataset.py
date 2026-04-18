"""
PyTorch Dataset — Loads audio files, extracts features, and provides
(mel_spectrogram, label) pairs for training.

Mel spectrograms are the primary input to the CNN model. Feature extraction
happens lazily (on __getitem__) so we can apply augmentation on-the-fly.
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.audio.preprocessor import load_audio, normalize_audio, trim_silence
from src.features.extractor import features_to_model_input
from src.data.augmentation import AudioAugmentor, random_crop
from src.utils.config import (
    SAMPLE_RATE, FRAME_DURATION, N_MELS, BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class DeepfakeAudioDataset(Dataset):
    """
    PyTorch Dataset that loads audio files and returns mel spectrogram tensors.

    Each sample:
      Input:  (1, n_mels, time_frames) — mel spectrogram as single-channel image
      Label:  int (0=real, 1=fake)

    Augmentation is applied only in training mode.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        sr: int = SAMPLE_RATE,
        duration: float = FRAME_DURATION,
        augment: bool = False,
        augment_prob: float = 0.5,
    ):
        """
        Args:
            manifest_df: DataFrame with 'path' and 'label' columns
            sr: Target sample rate
            duration: Target clip duration in seconds
            augment: Whether to apply augmentation (True for training)
            augment_prob: Probability of each augmentation being applied
        """
        self.df = manifest_df.reset_index(drop=True)
        self.sr = sr
        self.duration = duration
        self.target_length = int(sr * duration)
        self.augment = augment

        self.augmentor = AudioAugmentor(
            sr=sr, p=augment_prob, target_length=self.target_length
        ) if augment else None

        logger.info(
            f"Created dataset: {len(self)} samples, "
            f"augment={augment}, duration={duration}s"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load, preprocess, optionally augment, and extract features for one sample.

        Returns:
            (mel_tensor, label) where mel_tensor is (1, n_mels, time_frames)
        """
        row = self.df.iloc[idx]
        file_path = row["path"]
        label = int(row["label"])

        try:
            # Load and preprocess
            audio, sr = load_audio(file_path, sr=self.sr)
            audio = normalize_audio(audio)
            audio = trim_silence(audio, sr)

            # Crop or pad to target length
            audio = random_crop(audio, self.target_length)

            # Augment (training only)
            if self.augmentor is not None:
                audio = self.augmentor(audio)

            # Extract mel spectrogram → (n_mels, time_frames)
            mel = features_to_model_input(audio, sr)
            # Add channel dimension → (1, n_mels, time_frames)
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}. Returning zeros.")
            # Return zeros on error to avoid crashing the DataLoader
            time_frames = int(self.target_length / 512) + 1  # approximate
            mel_tensor = torch.zeros(1, N_MELS, time_frames, dtype=torch.float32)

        return mel_tensor, label

    def get_label_weights(self) -> torch.Tensor:
        """
        Compute inverse class frequency weights for balanced training.

        Returns:
            Tensor of shape (2,) with [weight_real, weight_fake]
        """
        labels = self.df["label"].values
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * 2  # normalize so they sum to num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        Returns:
            Tensor of shape (N,) with weight for each sample
        """
        label_weights = self.get_label_weights()
        labels = self.df["label"].values
        sample_weights = label_weights[labels]
        return sample_weights


def create_dataloader(
    manifest_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    balanced: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from a manifest DataFrame.

    Args:
        manifest_df: DataFrame with 'path' and 'label' columns
        batch_size: Batch size
        augment: Whether to apply augmentation
        shuffle: Whether to shuffle (ignored if balanced=True)
        num_workers: Number of data loading workers
        balanced: Use WeightedRandomSampler for class balance

    Returns:
        DataLoader yielding (mel_batch, label_batch)
    """
    dataset = DeepfakeAudioDataset(
        manifest_df=manifest_df,
        augment=augment,
    )

    sampler = None
    if balanced and len(dataset) > 0:
        sample_weights = dataset.get_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )
