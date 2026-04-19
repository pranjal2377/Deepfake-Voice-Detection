"""
Audio Preprocessor

Handles audio loading, normalization, resampling, and frame splitting.
This module prepares raw audio for feature extraction.
"""

import os
import logging
import numpy as np
import librosa
from typing import Tuple, Optional

from src.utils.config import SAMPLE_RATE, FRAME_DURATION, HOP_DURATION

logger = logging.getLogger(__name__)


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono, and resample.

    Args:
        file_path: Path to audio file (.wav, .flac, .mp3, etc.)
        sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If the file cannot be decoded as audio.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    except Exception as exc:
        raise ValueError(f"Cannot decode audio file '{file_path}': {exc}") from exc

    if len(audio) == 0:
        raise ValueError(f"Audio file is empty (0 samples): {file_path}")

    logger.debug("Loaded %s — %d samples @ %d Hz", file_path, len(audio), sample_rate)
    return audio, sample_rate


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Advanced RMS normalization (Finetuned for actual live mics).
    Removes recording bias and squashes dynamic web-audio distortion.
    """
    # 0. Pre-emphasis filter to boost high frequencies 
    # (sharpens speech formants against low-frequency laptop/phone mic rumble)
    if len(audio) > 1:
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # 1. Remove DC Offset (typical in cheap laptop/phone mics)
    audio = audio - np.mean(audio)
    
    # 2. RMS Energy Match (Targets 0.05 typical voice presence)
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0.001:
        audio = audio * (0.05 / rms)
        
    # 3. Soft clipping (TanH) to maintain acoustic fidelity without tearing
    # Scales audio neatly within [-1.0, 1.0] bounds
    audio = np.tanh(audio)
    
    return audio


def trim_silence(audio: np.ndarray, sr: int = SAMPLE_RATE,
                 top_db: int = 25) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.

    Args:
        audio: Audio signal
        sr: Sample rate
        top_db: Threshold below reference to consider as silence

    Returns:
        Trimmed audio array
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def split_into_frames(audio: np.ndarray, sr: int = SAMPLE_RATE,
                      frame_duration: float = FRAME_DURATION,
                      hop_duration: float = HOP_DURATION) -> list:
    """
    Split audio into overlapping frames for analysis.

    Args:
        audio: Audio signal
        sr: Sample rate
        frame_duration: Length of each frame in seconds
        hop_duration: Hop between consecutive frames in seconds

    Returns:
        List of audio frames (numpy arrays)
    """
    frame_length = int(sr * frame_duration)
    hop_length = int(sr * hop_duration)

    frames = []
    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start:start + frame_length]
        frames.append(frame)

    return frames


def preprocess_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline for a single audio file.

    Load → Normalize → Trim silence → Return

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (preprocessed_audio, sample_rate)

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file cannot be decoded or is empty after trimming.
    """
    audio, sr = load_audio(file_path)
    audio = normalize_audio(audio)
    audio = trim_silence(audio, sr)

    if len(audio) == 0:
        logger.warning("Audio is empty after silence trimming: %s", file_path)
        # Re-load without trimming so we still have something to analyse
        audio, sr = load_audio(file_path)
        audio = normalize_audio(audio)

    return audio, sr
