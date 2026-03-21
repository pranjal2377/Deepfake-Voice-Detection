"""
Phase 4 smoke tests — Evaluation, reporting, cross-dataset testing.
"""

import sys
import os
import tempfile

import numpy as np
import soundfile as sf
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _create_test_files(tmpdir: str, n: int = 6):
    """Create dummy audio files with real/fake labels."""
    import pandas as pd

    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)
    records = []

    for i in range(n):
        label = i % 2
        name = "real" if label == 0 else "fake"
        fname = f"{name}_{i:03d}.wav"
        fpath = os.path.join(tmpdir, fname)
        audio = np.random.randn(n_samples).astype(np.float32) * 0.3
        sf.write(fpath, audio, sr)
        records.append({
            "path": fpath,
            "label": label,
            "label_name": name,
            "filename": fname,
        })

    return pd.DataFrame(records)


def test_evaluator_predict_file():
    """Test single-file prediction."""
    from src.model.cnn_model import DeepfakeCNN
    from src.evaluation.evaluator import ModelEvaluator

    model = DeepfakeCNN()
    evaluator = ModelEvaluator(model, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create one test file
        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fpath = os.path.join(tmpdir, "test.wav")
        sf.write(fpath, audio, sr)

        result = evaluator.predict_file(fpath)

        assert "probability" in result
        assert 0 <= result["probability"] <= 1
        assert result["predicted_label"] in (0, 1)
        assert result["predicted_name"] in ("real", "fake")
        assert result["error"] is None

    print("✅ Single file prediction works")


def test_evaluator_manifest():
    """Test evaluation on a manifest DataFrame."""
    from src.model.cnn_model import DeepfakeCNN
    from src.evaluation.evaluator import ModelEvaluator

    model = DeepfakeCNN()
    evaluator = ModelEvaluator(model, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        df = _create_test_files(tmpdir, n=6)
        report = evaluator.evaluate_manifest(df, verbose=False)

        assert report["total_samples"] == 6
        assert "metrics" in report
        assert "confusion" in report
        assert "per_class" in report
        assert "misclassified" in report
        assert len(report["per_sample"]) == 6
        assert report["errors"] == 0

        # Check metrics keys
        m = report["metrics"]
        for key in ["accuracy", "precision", "recall", "f1", "auc", "eer"]:
            assert key in m, f"Missing metric: {key}"

        # Check confusion keys
        cm = report["confusion"]
        for key in ["TP", "TN", "FP", "FN"]:
            assert key in cm
        assert cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"] == 6

    print("✅ Manifest evaluation works")


def test_format_and_save_report():
    """Test report formatting and saving."""
    from src.model.cnn_model import DeepfakeCNN
    from src.evaluation.evaluator import (
        ModelEvaluator, format_report, save_report,
    )

    model = DeepfakeCNN()
    evaluator = ModelEvaluator(model, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        df = _create_test_files(tmpdir, n=4)
        report = evaluator.evaluate_manifest(df, verbose=False)

        # Format report
        text = format_report(report)
        assert "EVALUATION REPORT" in text
        assert "Accuracy" in text
        assert "Confusion Matrix" in text

        # Save report
        report_dir = os.path.join(tmpdir, "reports")
        text_path, json_path = save_report(report, report_dir)

        assert os.path.exists(text_path)
        assert os.path.exists(json_path)
        assert os.path.exists(os.path.join(report_dir, "evaluation_per_sample.csv"))

        # Verify JSON is valid
        import json
        with open(json_path) as f:
            loaded = json.load(f)
        assert "metrics" in loaded

    print("✅ Report formatting and saving works")


def test_evaluate_directory():
    """Test cross-dataset evaluation from a directory."""
    from src.model.cnn_model import DeepfakeCNN
    from src.evaluation.evaluator import ModelEvaluator

    model = DeepfakeCNN()
    evaluator = ModelEvaluator(model, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure: real/ and fake/ subfolders
        sr = 16000
        for label_dir in ["real", "fake"]:
            dirpath = os.path.join(tmpdir, label_dir)
            os.makedirs(dirpath)
            for i in range(3):
                audio = np.random.randn(sr * 2).astype(np.float32) * 0.3
                sf.write(os.path.join(dirpath, f"{label_dir}_{i}.wav"), audio, sr)

        report = evaluator.evaluate_directory(tmpdir, verbose=False)

        assert report["total_samples"] == 6
        assert "metrics" in report

    print("✅ Cross-dataset directory evaluation works")


def test_load_model():
    """Test loading a model from checkpoint."""
    from src.model.cnn_model import DeepfakeCNN
    from src.evaluation.evaluator import load_model_for_evaluation

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save a dummy checkpoint
        model = DeepfakeCNN()
        path = os.path.join(tmpdir, "test_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 10,
        }, path)

        # Load it
        loaded = load_model_for_evaluation(path)
        assert isinstance(loaded, DeepfakeCNN)

        # Verify it produces output
        x = torch.randn(1, 1, 128, 63)
        with torch.no_grad():
            out = loaded(x)
        assert out.shape == (1, 1)

    print("✅ Model loading from checkpoint works")


if __name__ == "__main__":
    test_evaluator_predict_file()
    test_evaluator_manifest()
    test_format_and_save_report()
    test_evaluate_directory()
    test_load_model()
    print("\n🎉 All Phase 4 smoke tests passed!")
