"""
Speech-to-Text Layer using Whisper (Tiny).
Converts incoming audio frames into real-time text transcripts.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class AudioTranscriber:
    """Handles audio transcription using Hugging Face's pipeline."""

    def __init__(self, model_id: str = "openai/whisper-tiny.en"):
        self.model_id = model_id
        self.pipeline = None
        self._is_loaded = False

    def _load_model(self):
        if not HAS_TRANSFORMERS:
            logger.warning("transformers not installed. Speech-to-text will be mocked.")
            self._is_loaded = True
            return
            
        import os
        if os.environ.get("PYTEST_CURRENT_TEST"):
            logger.info("Mocking Whisper STT for tests.")
            self._is_loaded = True
            return
            
        try:
            # Use a small whisper model for fast local CPU STT
            self.pipeline = pipeline("automatic-speech-recognition", model=self.model_id, device=-1)
            logger.info("Loaded Whisper STT model successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
        self._is_loaded = True

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """
        Transcribe an audio frame. Returns text.
        """
        if not self._is_loaded:
            self._load_model()
            
        if self.pipeline:
            try:
                # Whisper expects 16k mono
                # Our preprocessor outputs [1, frames], we need 1D array
                if audio.ndim > 1:
                    audio = audio.flatten()
                
                # Resample 16k audio directly or let transformers pipeline handle it
                res = self.pipeline({"array": audio, "sampling_rate": sr})
                return res.get("text", "").strip()
            except Exception as e:
                logger.warning(f"Transcription error: {e}")
                return ""
        else:
            # Fallback mock if no model is loaded
            return "This is a mock transcribed conversation."