"""
Feature Extractor

Extracts audio features for deepfake detection:
- MFCC (Mel Frequency Cepstral Coefficients)
- Mel Spectrogram
- Pitch (F0) stability
- Spectral centroid
- Energy distribution
- Harmonic-to-Noise Ratio (HNR)
"""

import numpy as np
import librosa
from typing import Dict, Optional

from src.utils.config import SAMPLE_RATE, N_MFCC, N_MELS, N_FFT, HOP_LENGTH, FMAX


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract MFCC features.
    MFCCs capture the timbral characteristics of speech,
    and deepfake voices often show subtle differences here.
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return mfcc


def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate mel spectrogram.
    Spectrograms reveal frequency patterns over time;
    synthetic speech often has artifacts in higher frequencies.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
    )
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """
    Extract pitch (F0) and its stability.
    Deepfake voices often have unnaturally stable or unstable pitch.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), sr=sr
    )

    # Remove NaN values (unvoiced frames)
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])

    if len(f0_clean) == 0:
        f0_clean = np.array([0.0])

    return {
        "mean": float(np.mean(f0_clean)),
        "std": float(np.std(f0_clean)),
        "stability": float(1.0 - min(np.std(f0_clean) / (np.mean(f0_clean) + 1e-8), 1.0)),
    }


def extract_spectral_centroid(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """
    Extract spectral centroid — the "center of mass" of the spectrum.
    Indicates the brightness of the sound.
    """
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return {
        "mean": float(np.mean(centroid)),
        "std": float(np.std(centroid)),
    }


def extract_energy(audio: np.ndarray) -> Dict[str, float]:
    """
    Extract RMS energy distribution.
    Deepfakes may have unnatural energy patterns.
    """
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    return {
        "mean": float(np.mean(rms)),
        "std": float(np.std(rms)),
        "max": float(np.max(rms)),
    }


def extract_hnr(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Estimate Harmonic-to-Noise Ratio.
    Higher HNR = more harmonic (natural voice).
    Deepfakes may show abnormal HNR values.
    """
    harmonic, percussive = librosa.effects.hpss(audio)
    harmonic_energy = np.sum(harmonic ** 2)
    noise_energy = np.sum(percussive ** 2)

    if noise_energy < 1e-10:
        return 40.0  # Very clean signal
    return float(10 * np.log10(harmonic_energy / noise_energy))


def extract_all_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict:
    """
    Extract ALL features from an audio segment.

    Returns a dictionary with all feature sets.
    This is the main entry point for feature extraction.
    """
    return {
        "mfcc": extract_mfcc(audio, sr),
        "mel_spectrogram": extract_mel_spectrogram(audio, sr),
        "pitch": extract_pitch(audio, sr),
        "spectral_centroid": extract_spectral_centroid(audio, sr),
        "energy": extract_energy(audio),
        "hnr": extract_hnr(audio, sr),
    }


def features_to_model_input(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Convert audio to the feature representation used by the model.
    Returns a mel spectrogram as a 2D array suitable for CNN input.

    Shape: (n_mels, time_steps)
    """
    mel_spec = extract_mel_spectrogram(audio, sr)
    return mel_spec
