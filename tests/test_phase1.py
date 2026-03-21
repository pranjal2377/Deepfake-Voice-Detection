"""
Phase 1 smoke tests — verify all modules import and basic classes work.
"""

import sys
import os

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def test_config_loads():
    from src.utils.config import (
        SAMPLE_RATE, FRAME_DURATION, N_MFCC, N_MELS,
        LOW_THRESHOLD, MEDIUM_THRESHOLD,
    )
    assert SAMPLE_RATE == 16000
    assert FRAME_DURATION == 2.0
    assert N_MFCC == 40
    assert N_MELS == 128
    assert LOW_THRESHOLD == 40
    assert MEDIUM_THRESHOLD == 70
    print("✅ config loads correctly")


def test_risk_scorer_basic():
    from src.scoring.risk_scorer import RiskScorer, RiskLevel

    scorer = RiskScorer()
    scorer.add_prediction(0.2)
    scorer.add_prediction(0.3)
    result = scorer.get_assessment()

    assert result["level"] == RiskLevel.LOW
    assert 0 <= result["score"] <= 100
    assert 0 <= result["confidence"] <= 100
    print("✅ RiskScorer works")


def test_alert_system():
    from src.scoring.risk_scorer import RiskScorer
    from src.alerts.alert_system import generate_alert

    scorer = RiskScorer()
    scorer.add_prediction(0.9)
    scorer.add_prediction(0.85)
    assessment = scorer.get_assessment()
    alert = generate_alert(assessment)

    assert alert["level"].name == "HIGH"
    assert "deepfake" in alert["message"].lower() or "warning" in alert["message"].lower()
    assert len(alert["explanation"]) > 0
    print("✅ Alert system works")


def test_detector_stub():
    from src.detection.detector import RealtimeDetector

    det = RealtimeDetector(model=None)
    assert det._predict(None) == 0.5  # stub returns 0.5
    print("✅ Detector stub works")


if __name__ == "__main__":
    test_config_loads()
    test_risk_scorer_basic()
    test_alert_system()
    test_detector_stub()
    print("\n🎉 All Phase 1 smoke tests passed!")
