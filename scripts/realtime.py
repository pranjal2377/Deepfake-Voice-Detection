#!/usr/bin/env python3
"""
Real-Time Detection Script — Phase 5 CLI.

Runs deepfake voice detection on live microphone input or
simulates real-time analysis on an audio file.

Usage:
    # Live microphone (requires a microphone):
    python scripts/realtime.py --mode mic

    # Simulate real-time on a file:
    python scripts/realtime.py --mode simulate --file path/to/audio.wav

    # Simulate on demo data (auto-picks a file):
    python scripts/realtime.py --mode demo

    # With trained model:
    python scripts/realtime.py --mode demo --model models/best_model.pt

    # Set analysis duration (seconds):
    python scripts/realtime.py --mode mic --duration 30
"""

import sys
import os
import argparse
import logging
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import RealtimeDetector
from src.evaluation.evaluator import load_model_for_evaluation
from src.scoring.risk_scorer import RiskLevel
from src.utils.config import MODELS_DIR, RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ANSI colors for terminal output
COLORS = {
    RiskLevel.LOW: "\033[92m",      # green
    RiskLevel.MEDIUM: "\033[93m",   # yellow
    RiskLevel.HIGH: "\033[91m",     # red
}
RESET = "\033[0m"
BOLD = "\033[1m"


def frame_callback(result: dict):
    """Print each frame's analysis result to the terminal."""
    assessment = result["assessment"]
    alert = result["alert"]
    level = assessment["level"]
    color = COLORS.get(level, "")

    prob = result["probability"]
    score = assessment["score"]
    confidence = assessment["confidence"]

    idx = result["frame_index"]

    print(
        f"  Frame {idx:3d} │ "
        f"Prob: {prob:.3f} │ "
        f"Score: {score:5.1f}% │ "
        f"Confidence: {confidence:5.1f}% │ "
        f"{color}{BOLD}{level.name:6s}{RESET} │ "
        f"{alert['title']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Real-time deepfake voice detection")
    parser.add_argument(
        "--mode",
        choices=["mic", "simulate", "demo"],
        default="demo",
        help="Detection mode: mic (live), simulate (file), demo (auto)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Audio file path for simulate mode",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(MODELS_DIR, "best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Max duration in seconds for mic mode (default: 30)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Simulation speed multiplier (default: 4x)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Run without a model (uses 0.5 probability stub)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Deepfake Voice Detection — Real-Time Analysis")
    print("=" * 70)

    # Load model
    model = None
    if not args.no_model:
        try:
            model = load_model_for_evaluation(args.model)
            print(f"\n🧠 Model loaded: {args.model}")
        except FileNotFoundError:
            print(f"\n⚠️  No model found at {args.model}. Using stub (prob=0.5).")
            print("   Train a model first: python scripts/train.py")

    detector = RealtimeDetector(model=model)

    print(f"\n{'─' * 70}")
    print(f"  {'Frame':>8s} │ {'Prob':>5s} │ {'Score':>7s} │ {'Conf':>11s} │ {'Level':>6s} │ Alert")
    print(f"{'─' * 70}")

    if args.mode == "mic":
        # Live microphone
        print("\n🎤 Listening on microphone... (Ctrl+C to stop)\n")

        # Handle Ctrl+C gracefully
        def _signal_handler(sig, frame):
            detector.stop_realtime()

        signal.signal(signal.SIGINT, _signal_handler)

        detector.start_realtime(callback=frame_callback)

        # Wait for duration or until stopped
        start = time.time()
        while detector.is_running and (time.time() - start) < args.duration:
            time.sleep(0.2)

        detector.stop_realtime()

    elif args.mode == "simulate":
        # Simulate on a specific file
        if not args.file:
            print("❌ --file is required for simulate mode")
            return

        print(f"\n📂 Simulating on: {args.file} (speed={args.speed}x)\n")

        import librosa
        audio, sr = librosa.load(args.file, sr=16000)

        detector.start_simulation(audio, callback=frame_callback, speed=args.speed)

        while detector.is_running:
            time.sleep(0.1)

    elif args.mode == "demo":
        # Auto-pick a file from demo data
        demo_files = []
        for root, dirs, files in os.walk(RAW_DATA_DIR):
            for f in files:
                if f.endswith(".wav"):
                    demo_files.append(os.path.join(root, f))

        if not demo_files:
            print("❌ No demo data found. Run: python scripts/prepare_dataset.py --generate-demo")
            return

        # Pick one real and one fake
        real_files = [f for f in demo_files if "real" in f]
        fake_files = [f for f in demo_files if "fake" in f]

        import librosa

        for label, files in [("REAL", real_files[:1]), ("FAKE", fake_files[:1])]:
            for fpath in files:
                print(f"\n🔊 Analyzing {label} sample: {os.path.basename(fpath)} "
                      f"(speed={args.speed}x)\n")

                audio, sr = librosa.load(fpath, sr=16000)
                detector.scorer.reset()
                detector.frame_history.clear()

                detector.start_simulation(audio, callback=frame_callback, speed=args.speed)
                while detector.is_running:
                    time.sleep(0.1)

                # Summary
                summary = detector.get_summary()
                level = summary["latest_assessment"]["level"]
                color = COLORS.get(level, "")
                print(f"\n  📊 Result: {color}{BOLD}{level.name}{RESET} "
                      f"(avg prob: {summary['avg_probability']:.3f}, "
                      f"frames: {summary['frames_analyzed']})")
                print(f"{'─' * 70}")

    # Final summary
    summary = detector.get_summary()
    if summary["frames_analyzed"] > 0:
        print(f"\n{'=' * 70}")
        print(f"  Session Summary")
        print(f"{'=' * 70}")
        print(f"  Frames analyzed:  {summary['frames_analyzed']}")
        print(f"  Avg probability:  {summary['avg_probability']:.4f}")
        print(f"  Max probability:  {summary['max_probability']:.4f}")
        print(f"  Final verdict:    {summary['latest_assessment']['level'].name}")

    print("\n✅ Real-time analysis complete!")


if __name__ == "__main__":
    main()
