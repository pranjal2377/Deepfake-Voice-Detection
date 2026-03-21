"""
Phase 2 smoke tests — Dataset loading, splitting, augmentation, PyTorch dataset.
"""

import sys
import os
import tempfile
import shutil

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _create_test_audio_tree(root: str, n_per_class: int = 4):
    """Create a temporary directory tree with dummy audio files."""
    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)

    for subset in ["training", "testing"]:
        for label in ["real", "fake"]:
            dirpath = os.path.join(root, subset, label)
            os.makedirs(dirpath, exist_ok=True)
            for i in range(n_per_class):
                audio = np.random.randn(n_samples).astype(np.float32) * 0.5
                sf.write(os.path.join(dirpath, f"{label}_{i:03d}.wav"), audio, sr)


def test_discover_files():
    """Test that discover_files finds and labels audio files correctly."""
    from src.data.dataset_loader import discover_files

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_test_audio_tree(tmpdir, n_per_class=3)
        entries = discover_files(tmpdir)

        assert len(entries) == 12  # 2 subsets × 2 classes × 3 files
        labels = {e["label"] for e in entries}
        assert labels == {0, 1}
        subsets = {e["subset"] for e in entries}
        assert "train" in subsets and "test" in subsets
        print("✅ discover_files works correctly")


def test_create_manifest():
    """Test manifest creation and CSV save."""
    from src.data.dataset_loader import create_manifest, load_manifest

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "audio")
        _create_test_audio_tree(data_dir, n_per_class=3)

        manifest_path = os.path.join(tmpdir, "manifest.csv")
        df = create_manifest(root_dir=data_dir, output_path=manifest_path)

        assert len(df) == 12
        assert os.path.exists(manifest_path)

        # Test loading
        df2 = load_manifest(manifest_path)
        assert len(df2) == 12
        print("✅ create_manifest and load_manifest work")


def test_split_dataset():
    """Test stratified train/val/test splitting."""
    from src.data.dataset_loader import discover_files
    from src.data.splitter import split_dataset
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_test_audio_tree(tmpdir, n_per_class=10)
        entries = discover_files(tmpdir)
        df = pd.DataFrame(entries)

        # Force resplit into 70/15/15
        df = split_dataset(df, force_resplit=True)

        assert set(df["subset"].unique()) == {"train", "val", "test"}
        # Check that every subset has both classes
        for subset in ["train", "val", "test"]:
            sub = df[df["subset"] == subset]
            assert 0 in sub["label"].values
            assert 1 in sub["label"].values
        print("✅ split_dataset works with stratification")


def test_augmentation():
    """Test that augmentation functions produce valid audio."""
    from src.data.augmentation import (
        add_gaussian_noise, pitch_shift, time_stretch,
        volume_perturbation, random_crop, AudioAugmentor,
    )

    sr = 16000
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)

    # Individual augmentations
    noisy = add_gaussian_noise(audio, snr_db=15)
    assert noisy.shape == audio.shape
    assert not np.array_equal(noisy, audio)

    shifted = pitch_shift(audio, sr=sr, n_steps=2)
    assert len(shifted) > 0

    stretched = time_stretch(audio, rate=1.2)
    assert len(stretched) > 0

    louder = volume_perturbation(audio, gain_db=6)
    assert louder.shape == audio.shape

    cropped = random_crop(audio, sr // 2)
    assert cropped.shape == (sr // 2,)

    padded = random_crop(audio[:100], sr)
    assert padded.shape == (sr,)

    # Augmentor pipeline
    aug = AudioAugmentor(sr=sr, p=1.0, target_length=sr)
    result = aug(audio)
    assert result.shape == (sr,)

    print("✅ Augmentation functions work correctly")


def test_pytorch_dataset():
    """Test DeepfakeAudioDataset and DataLoader."""
    import torch
    from src.data.audio_dataset import DeepfakeAudioDataset, create_dataloader
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a few test files
        sr = 16000
        n_samples = int(sr * 2.0)

        records = []
        for i in range(6):
            label = i % 2
            label_name = "real" if label == 0 else "fake"
            fname = f"{label_name}_{i:03d}.wav"
            fpath = os.path.join(tmpdir, fname)
            audio = np.random.randn(n_samples).astype(np.float32) * 0.3
            sf.write(fpath, audio, sr)
            records.append({"path": fpath, "label": label, "label_name": label_name})

        df = pd.DataFrame(records)

        # Test Dataset
        dataset = DeepfakeAudioDataset(df, augment=False)
        assert len(dataset) == 6

        mel, label = dataset[0]
        assert isinstance(mel, torch.Tensor)
        assert mel.dim() == 3  # (1, n_mels, time_frames)
        assert label in (0, 1)

        # Test label weights
        weights = dataset.get_label_weights()
        assert weights.shape == (2,)

        # Test DataLoader
        loader = create_dataloader(df, batch_size=2, augment=False, balanced=False)
        batch_mel, batch_labels = next(iter(loader))
        assert batch_mel.shape[0] == 2
        assert batch_mel.dim() == 4  # (batch, 1, n_mels, time_frames)

        print("✅ PyTorch Dataset and DataLoader work correctly")


if __name__ == "__main__":
    test_discover_files()
    test_create_manifest()
    test_split_dataset()
    test_augmentation()
    test_pytorch_dataset()
    print("\n🎉 All Phase 2 smoke tests passed!")
