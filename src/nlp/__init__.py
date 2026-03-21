"""
NLP module for transcribing audio and detecting scam content.
"""

from .transcriber import AudioTranscriber
from .scam_detector import ScamDetector

__all__ = ["AudioTranscriber", "ScamDetector"]