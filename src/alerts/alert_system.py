"""
Alert & Explanation Module

Generates user-facing alerts and explanations
for why a voice was flagged as suspicious or deepfake.
"""

from typing import Dict, List, Optional
from src.scoring.risk_scorer import RiskLevel


def generate_alert(assessment: Dict, features: Optional[Dict] = None) -> Dict:
    """
    Generate a user-facing alert based on risk assessment.

    Args:
        assessment: Output from RiskScorer.get_assessment()
        features: Optional extracted features for explanation

    Returns:
        Dict with alert info: title, message, explanation, level
    """
    level = assessment["level"]
    score = assessment["score"]

    alert = {
        "level": level,
        "score": score,
        "title": _get_title(level),
        "message": _get_message(level, score),
        "explanation": _get_explanation(level, score, features),
        "action": _get_action(level),
    }

    return alert


def _get_title(level: RiskLevel) -> str:
    """Get alert title based on risk level."""
    titles = {
        RiskLevel.LOW: "Voice Appears Authentic",
        RiskLevel.MEDIUM: "Suspicious Voice Patterns Detected",
        RiskLevel.HIGH: "⚠ Possible Deepfake Voice Detected",
    }
    return titles[level]


def _get_message(level: RiskLevel, score: float) -> str:
    """Get alert message."""
    if level == RiskLevel.LOW:
        return (
            f"The voice analysis indicates a {score:.0f}% deepfake probability. "
            f"The voice appears to be from a real human speaker."
        )
    elif level == RiskLevel.MEDIUM:
        return (
            f"The voice analysis shows a {score:.0f}% deepfake probability. "
            f"Some patterns are unusual. Proceed with caution."
        )
    else:
        return (
            f"WARNING: The voice analysis shows a {score:.0f}% deepfake probability. "
            f"This voice shows strong indicators of AI generation."
        )


def _get_explanation(level: RiskLevel, score: float,
                     features: Optional[Dict] = None) -> List[str]:
    """
    Generate explanation points for why the voice was flagged.
    Uses feature data when available, otherwise provides general explanations.
    """
    explanations = []

    if features:
        # Pitch analysis
        pitch = features.get("pitch", {})
        stability = pitch.get("stability", 0.5)
        if stability > 0.95:
            explanations.append(
                "Unnaturally stable pitch detected — real voices have more variation"
            )
        elif stability < 0.1:
            explanations.append(
                "Extremely unstable pitch patterns — may indicate synthesis artifacts"
            )

        # Energy analysis
        energy = features.get("energy", {})
        energy_std = energy.get("std", 0)
        if energy_std < 0.001:
            explanations.append(
                "Very uniform energy levels — natural speech varies in loudness"
            )

        # HNR analysis
        hnr = features.get("hnr", 15.0)
        if isinstance(hnr, (int, float)):
            if hnr > 35:
                explanations.append(
                    "Abnormally high harmonic-to-noise ratio — too clean for natural speech"
                )
            elif hnr < 5:
                explanations.append(
                    "Very low harmonic content — unusual noise characteristics"
                )

        # Spectral centroid
        centroid = features.get("spectral_centroid", {})
        centroid_std = centroid.get("std", 0)
        if centroid_std < 50:
            explanations.append(
                "Spectral brightness barely changes — synthetic voices lack timbral variation"
            )

    # General explanations based on score
    if not explanations:
        if level == RiskLevel.HIGH:
            explanations = [
                "Voice patterns match known deepfake characteristics",
                "Spectral analysis reveals synthetic artifacts",
                "Multiple detection indicators triggered simultaneously",
            ]
        elif level == RiskLevel.MEDIUM:
            explanations = [
                "Some voice characteristics appear slightly unnatural",
                "Additional frames needed for higher confidence",
            ]
        else:
            explanations = [
                "Voice characteristics match natural human speech patterns",
            ]

    return explanations


def _get_action(level: RiskLevel) -> str:
    """Recommended action for the user."""
    actions = {
        RiskLevel.LOW: "No action needed. The call appears safe.",
        RiskLevel.MEDIUM: "Stay alert. Verify the caller's identity through other means.",
        RiskLevel.HIGH: "End the call immediately. Do NOT share personal information.",
    }
    return actions[level]
