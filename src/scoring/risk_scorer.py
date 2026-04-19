"""
Risk Scoring Engine

Aggregates model predictions across multiple audio frames
and produces a final risk classification.

Risk Levels:
  Low    (0-40%)   — Human voice likely
  Medium (40-70%)  — Suspicious voice patterns
  High   (70-100%) — Likely deepfake voice
"""

from typing import List, Dict, Tuple
from collections import deque
from enum import Enum

from src.utils.config import LOW_THRESHOLD, MEDIUM_THRESHOLD, AGGREGATION_WINDOW, SMOOTHING_FACTOR


class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Colors for each risk level (used by UI)
RISK_COLORS = {
    RiskLevel.LOW: {"primary": "#10B981", "bg": "#ECFDF5", "text": "#065F46"},
    RiskLevel.MEDIUM: {"primary": "#F59E0B", "bg": "#FFFBEB", "text": "#92400E"},
    RiskLevel.HIGH: {"primary": "#EF4444", "bg": "#FEF2F2", "text": "#991B1B"},
}


class RiskScorer:
    """
    Maintains a sliding window of predictions and computes
    a smoothed risk score with classification.
    """

    def __init__(self, window_size: int = AGGREGATION_WINDOW,
                 smoothing: float = SMOOTHING_FACTOR):
        self.window_size = window_size
        self.smoothing = smoothing
        self.predictions = deque(maxlen=window_size)
        self.nlp_predictions = deque(maxlen=window_size)
        self.smoothed_score: float = 0.0
        self.smoothed_nlp_score: float = 0.0

    def add_prediction(self, deepfake_probability: float, nlp_probability: float = 0.0) -> Dict:
        """
        Add a new frame prediction and return updated risk assessment.

        Args:
            deepfake_probability: Model output probability (0.0 to 1.0)
            nlp_probability: Scam NLP detection probability (0.0 to 1.0)

        Returns:
            Dict with score, level, confidence, and history
        """
        score_pct = deepfake_probability * 100.0
        nlp_score_pct = nlp_probability * 100.0
        self.predictions.append(score_pct)
        self.nlp_predictions.append(nlp_score_pct)

        # Exponential moving average for smoothing
        if len(self.predictions) == 1:
            self.smoothed_score = score_pct
            self.smoothed_nlp_score = nlp_score_pct
        else:
            self.smoothed_score = (
                self.smoothing * score_pct +
                (1 - self.smoothing) * self.smoothed_score
            )
            # NLP updates less frequently, use smaller smoothing or direct average
            self.smoothed_nlp_score = (
                0.5 * nlp_score_pct +
                0.5 * self.smoothed_nlp_score
            )

        return self.get_assessment()

    def get_assessment(self) -> Dict:
        """Get current risk assessment."""
        # Multi-Modal Fine-Tuning Fusion (Live Mic Optimization)
        # Deepfakes trigger high acoustic risks due to cheap mics. 
        # But if the person makes normal 'human' non-scam conversation,
        # we weight down the false-positive AI audio score automatically.
        v_score = self.smoothed_score
        n_score = self.smoothed_nlp_score
        
        if n_score > 35.0:
            # Significant linguistic threat increases the total risk profile
            combined_score = max(v_score, n_score) * 0.8 + (v_score * 0.2)
        else:
            # If language is totally normal, drastically cut acoustic false-positive risk.
            # E.g. Trusting it's just a bad microphone and relying mostly on content.
            combined_score = (v_score * 0.60) + (n_score * 0.40)
            
        # Ensure percentages stay cleanly in [0, 100] bounds
        combined_score = max(0.0, min(100.0, combined_score))
        
        level = self.classify(combined_score)
        return {
            "score": round(combined_score, 1),
            "voice_score": round(self.smoothed_score, 1),
            "nlp_score": round(self.smoothed_nlp_score, 1),
            "level": level,
            "confidence": self._compute_confidence(),
            "frame_count": len(self.predictions),
            "raw_scores": list(self.predictions),
            "raw_nlp": list(self.nlp_predictions),
        }

    def classify(self, score: float) -> RiskLevel:
        """Classify score into risk level."""
        if score <= LOW_THRESHOLD:
            return RiskLevel.LOW
        elif score <= MEDIUM_THRESHOLD:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def _compute_confidence(self) -> float:
        """
        Confidence based on how many frames we've seen
        and how consistent the predictions are.
        """
        if len(self.predictions) == 0:
            return 0.0

        # More frames = higher base confidence
        frame_conf = min(len(self.predictions) / self.window_size, 1.0)

        # Lower variance = higher confidence
        if len(self.predictions) >= 2:
            import numpy as np
            std = float(np.std(list(self.predictions)))
            consistency = max(0.0, 1.0 - (std / 50.0))
        else:
            consistency = 0.5

        return round(frame_conf * 0.6 + consistency * 0.4, 2)

    def reset(self):
        """Reset scorer state for a new call/session."""
        self.predictions.clear()
        self.nlp_predictions.clear()
        self.smoothed_score = 0.0
        self.smoothed_nlp_score = 0.0
