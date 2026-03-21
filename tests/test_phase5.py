"""
Phase 5 smoke tests — Real-time detection, simulation, callbacks.
"""

import sys
import os
import time
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def test_analyze_file():
    """Test file-based analysis still works after Phase 5 rewrite."""
    from src.detection.detector import RealtimeDetector

    with tempfile.TemporaryDirectory() as tmpdir:
        sr = 16000
        audio = np.random.randn(sr * 3).astype(np.float32) * 0.3
        fpath = os.path.join(tmpdir, "test.wav")
        sf.write(fpath, audio, sr)

        detector = RealtimeDetector(model=None)
        result = detector.analyze_file(fpath)

        assert "assessment" in result
        assert "alert" in result
        assert "frame_results" in result
        assert result["num_frames"] >= 1

        # With stub model, probability should be 0.5
        for fr in result["frame_results"]:
            assert fr["probability"] == 0.5

    print("✅ File analysis works correctly")


def test_simulation_mode():
    """Test simulation mode with callback."""
    from src.detection.detector import RealtimeDetector

    sr = 16000
    duration = 3.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.3

    results_received = []

    def my_callback(result):
        results_received.append(result)

    detector = RealtimeDetector(model=None)
    detector.start_simulation(audio, callback=my_callback, speed=100.0)

    # Wait for simulation to complete
    timeout = time.time() + 10
    while detector.is_running and time.time() < timeout:
        time.sleep(0.05)

    assert not detector.is_running, "Simulation should have stopped"
    assert len(results_received) > 0, "Should have received at least one result"

    # Check result structure
    r = results_received[0]
    assert "frame_index" in r
    assert "probability" in r
    assert "assessment" in r
    assert "alert" in r
    assert "features" in r
    assert "timestamp" in r

    print(f"✅ Simulation mode works ({len(results_received)} frames processed)")


def test_get_summary():
    """Test session summary after simulation."""
    from src.detection.detector import RealtimeDetector

    sr = 16000
    audio = np.random.randn(sr * 4).astype(np.float32) * 0.3

    detector = RealtimeDetector(model=None)
    detector.start_simulation(audio, speed=100.0)

    timeout = time.time() + 10
    while detector.is_running and time.time() < timeout:
        time.sleep(0.05)

    summary = detector.get_summary()

    assert summary["frames_analyzed"] > 0
    assert "avg_probability" in summary
    assert "max_probability" in summary
    assert "min_probability" in summary
    assert "latest_assessment" in summary
    assert "latest_alert" in summary

    # Stub model → probability = 0.5 for all frames
    assert abs(summary["avg_probability"] - 0.5) < 0.01

    print(f"✅ Session summary works ({summary['frames_analyzed']} frames)")


def test_simulation_with_model():
    """Test simulation with actual trained model (if available)."""
    from src.detection.detector import RealtimeDetector
    from src.utils.config import MODELS_DIR
    import torch

    model_path = os.path.join(MODELS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print("⏭️  Skipping model simulation test (no trained model)")
        return

    from src.evaluation.evaluator import load_model_for_evaluation

    model = load_model_for_evaluation(model_path)

    sr = 16000
    # Create a simple "real-like" signal
    t = np.linspace(0, 3.0, sr * 3, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 150 * t)
             + 0.3 * np.sin(2 * np.pi * 300 * t)
             + 0.05 * np.random.randn(len(t))).astype(np.float32)

    results = []
    detector = RealtimeDetector(model=model)
    detector.start_simulation(audio, callback=lambda r: results.append(r), speed=100.0)

    timeout = time.time() + 15
    while detector.is_running and time.time() < timeout:
        time.sleep(0.05)

    assert len(results) > 0
    # Model should produce varied probabilities (not all 0.5)
    probs = [r["probability"] for r in results]
    assert not all(p == 0.5 for p in probs), "Model should produce non-stub predictions"

    print(f"✅ Simulation with trained model works "
          f"(avg prob: {np.mean(probs):.3f}, {len(results)} frames)")


def test_multiple_callbacks():
    """Test that multiple callbacks all receive results."""
    from src.detection.detector import RealtimeDetector

    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32) * 0.3

    results_a = []
    results_b = []

    detector = RealtimeDetector(model=None)
    detector.start_simulation(audio, callback=lambda r: results_a.append(r), speed=100.0)

    # Can't add second callback after start in simulation mode easily,
    # but let's test add_callback before start
    detector2 = RealtimeDetector(model=None)
    detector2.add_callback(lambda r: results_a.append(r))
    detector2.add_callback(lambda r: results_b.append(r))
    detector2.start_simulation(audio, speed=100.0)

    timeout = time.time() + 10
    while detector.is_running or detector2.is_running:
        time.sleep(0.05)
        if time.time() > timeout:
            break

    assert len(results_a) > 0
    assert len(results_b) > 0

    print("✅ Multiple callbacks work correctly")


if __name__ == "__main__":
    test_analyze_file()
    test_simulation_mode()
    test_get_summary()
    test_simulation_with_model()
    test_multiple_callbacks()
    print("\n🎉 All Phase 5 smoke tests passed!")
