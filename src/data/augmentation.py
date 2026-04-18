"""
Audio Augmentation — Transform audio to increase training data diversity.

All augmentations operate on raw numpy audio arrays and return augmented copies.
They are applied on-the-fly during training (not saved to disk).

Augmentation strategies for deepfake detection:
  - Add background noise (simulates phone call conditions)
  - Pitch shifting (simulates different speakers)
  - Time stretching (simulates tempo variations)
  - Volume perturbation (simulates gain variations)
  - Codec simulation (simulates phone compression artifacts)
"""

import numpy as np
import librosa
import logging
from typing import List, Callable, Optional, Dict

logger = logging.getLogger(__name__)


def add_gaussian_noise(
    audio: np.ndarray,
    snr_db: float = 20.0,
) -> np.ndarray:
    """
    Add Gaussian white noise at a specified SNR.

    Args:
        audio: Input waveform
        snr_db: Signal-to-noise ratio in dB (lower = noisier)

    Returns:
        Noisy audio
    """
    rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-10
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, len(audio))
    return (audio + noise).astype(audio.dtype)


def pitch_shift(
    audio: np.ndarray,
    sr: int = 16000,
    n_steps: float = 2.0,
) -> np.ndarray:
    """
    Shift pitch by n semitone steps.

    Args:
        audio: Input waveform
        sr: Sample rate
        n_steps: Number of semitones to shift (positive = up, negative = down)

    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def time_stretch(
    audio: np.ndarray,
    rate: float = 1.1,
) -> np.ndarray:
    """
    Stretch/compress audio in time domain.

    Args:
        audio: Input waveform
        rate: Stretch factor (>1 = faster, <1 = slower)

    Returns:
        Time-stretched audio
    """
    if rate <= 0:
        raise ValueError("Time stretch rate must be positive")
    return librosa.effects.time_stretch(y=audio, rate=rate)


def volume_perturbation(
    audio: np.ndarray,
    gain_db: float = 3.0,
) -> np.ndarray:
    """
    Apply volume gain in dB.

    Args:
        audio: Input waveform
        gain_db: Gain in dB (positive = louder, negative = quieter)

    Returns:
        Volume-adjusted audio
    """
    gain_linear = 10 ** (gain_db / 20)
    return np.clip(audio * gain_linear, -1.0, 1.0).astype(audio.dtype)


def random_crop(
    audio: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Randomly crop a fixed-length segment from the audio.
    If audio is shorter than target, it is zero-padded.

    Args:
        audio: Input waveform
        target_length: Desired length in samples

    Returns:
        Cropped/padded audio of exactly target_length
    """
    if len(audio) <= target_length:
        padded = np.zeros(target_length, dtype=audio.dtype)
        padded[:len(audio)] = audio
        return padded

    start = np.random.randint(0, len(audio) - target_length)
    return audio[start:start + target_length]


def add_reverb(
    audio: np.ndarray,
    sr: int = 16000,
    decay: float = 0.3,
    delay_ms: float = 20.0,
) -> np.ndarray:
    """
    Add simple synthetic reverb via a decaying echo.

    Args:
        audio: Input waveform
        sr: Sample rate
        decay: Echo amplitude decay factor (0-1)
        delay_ms: Echo delay in milliseconds

    Returns:
        Audio with reverb effect
    """
    delay_samples = int(sr * delay_ms / 1000)
    output = audio.copy()
    if delay_samples < len(audio):
        output[delay_samples:] += decay * audio[:len(audio) - delay_samples]
    return np.clip(output, -1.0, 1.0).astype(audio.dtype)


def low_pass_filter(
    audio: np.ndarray,
    sr: int = 16000,
    cutoff_hz: float = 3400.0,
) -> np.ndarray:
    """
    Apply a low-pass filter (simulates telephone bandwidth).
    Telephone lines typically cut off at ~3.4 kHz.

    Args:
        audio: Input waveform
        sr: Sample rate
        cutoff_hz: Cutoff frequency in Hz

    Returns:
        Filtered audio
    """
    from scipy.signal import butter, sosfilt

    nyquist = sr / 2
    if cutoff_hz >= nyquist:
        return audio

    sos = butter(5, cutoff_hz / nyquist, btype="low", output="sos")
    return sosfilt(sos, audio).astype(audio.dtype)


class AudioAugmentor:
    """
    Applies a random set of augmentations to audio during training.

    Usage:
        augmentor = AudioAugmentor(sr=16000, p=0.5)
        augmented = augmentor(audio)
    """

    def __init__(
        self,
        sr: int = 16000,
        p: float = 0.5,
        target_length: Optional[int] = None,
    ):
        """
        Args:
            sr: Sample rate
            p: Probability of applying each augmentation
            target_length: If set, crops/pads to this length after augmentation
        """
        self.sr = sr
        self.p = p
        self.target_length = target_length

        # List of (name, function, kwargs) tuples
        self._augmentations: List[tuple] = [
            ("gaussian_noise_20dB", add_gaussian_noise, {"snr_db": 20.0}),
            ("gaussian_noise_10dB", add_gaussian_noise, {"snr_db": 10.0}),
            ("pitch_up", pitch_shift, {"sr": sr, "n_steps": 1.5}),
            ("pitch_down", pitch_shift, {"sr": sr, "n_steps": -1.5}),
            ("time_stretch_fast", time_stretch, {"rate": 1.1}),
            ("time_stretch_slow", time_stretch, {"rate": 0.9}),
            ("volume_up", volume_perturbation, {"gain_db": 3.0}),
            ("volume_down", volume_perturbation, {"gain_db": -3.0}),
            ("reverb", add_reverb, {"sr": sr}),
            ("telephone_filter", low_pass_filter, {"sr": sr, "cutoff_hz": 3400.0}),
        ]

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations to audio."""
        augmented = audio.copy()

        for name, fn, kwargs in self._augmentations:
            if np.random.random() < self.p:
                try:
                    augmented = fn(augmented, **kwargs)
                except Exception as e:
                    logger.debug(f"Augmentation {name} failed: {e}")

        # Ensure output length
        if self.target_length is not None:
            augmented = random_crop(augmented, self.target_length)

        return augmented

    def apply_specific(
        self, audio: np.ndarray, augmentation_name: str
    ) -> np.ndarray:
        """Apply a specific named augmentation."""
        for name, fn, kwargs in self._augmentations:
            if name == augmentation_name:
                return fn(audio, **kwargs)
        raise ValueError(
            f"Unknown augmentation: {augmentation_name}. "
            f"Available: {[n for n, _, _ in self._augmentations]}"
        )

    @property
    def available_augmentations(self) -> List[str]:
        """List all available augmentation names."""
        return [name for name, _, _ in self._augmentations]
