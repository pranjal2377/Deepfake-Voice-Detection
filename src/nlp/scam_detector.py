"""
NLP Scam Detection Module.
Analyzes transcripts to detect scam or social engineering patterns.
"""

from typing import Dict, List

# Rule-based keywords commonly used in fraud calls
SCAM_KEYWORDS = [
    "send otp",
    "urgent payment required",
    "verify your bank account",
    "transfer money immediately",
    "your account will be blocked",
    "gift card",
    "anydesk",
    "teamviewer",
    "remote access",
    "social security",
    "arrest warrant",
    "pay immediately",
]

class ScamDetector:
    """
    Detects scam language in text transcripts and classifies the intent.
    Uses rule-based detection for immediate real-time execution.
    """

    def analyze_transcript(self, text: str) -> Dict:
        """
        Analyze a transcript segment.

        Returns:
            Dict containing: scam_probability, detected_phrases, intent
        """
        text_lower = text.lower()
        detected_phrases = []

        for kw in SCAM_KEYWORDS:
            if kw in text_lower:
                detected_phrases.append(kw)

        # Basic probability: each detected phrase increases risk
        scam_prob = len(detected_phrases) * 0.35
        scam_prob = min(scam_prob, 1.0) # Cap at 1.0

        # Intent Classification
        intent = "Normal conversation"
        if scam_prob > 0:
            if any(w in text_lower for w in ["bank", "transfer", "payment", "gift card", "pay"]):
                intent = "Financial scam attempt"
            elif any(w in text_lower for w in ["urgent", "blocked", "arrest", "immediately"]):
                intent = "Emergency impersonation scam"
            else:
                intent = "Suspicious request"

        return {
            "transcript": text,
            "nlp_probability": scam_prob,
            "detected_phrases": detected_phrases,
            "intent": intent
        }