"""
Phase 3 smoke tests — Metrics, Trainer, and model training pipeline.
"""

import sys
import os
import tempfile

import numpy as np
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def test_metrics():
    """Test that metrics computation is correct."""
    from src.training.metrics import (
        compute_metrics, compute_eer, compute_confusion, find_optimal_threshold,
    )

    # Perfect predictions
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95])

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc"] > 0.99

    # EER should be very low for perfect separation
    eer = compute_eer(y_true, y_pred)
    assert eer < 0.1

    # Confusion matrix
    cm = compute_confusion(y_true, y_pred)
    assert cm["TP"] == 3
    assert cm["TN"] == 3
    assert cm["FP"] == 0
    assert cm["FN"] == 0

    # Optimal threshold
    thresh, f1 = find_optimal_threshold(y_true, y_pred)
    assert 0.1 < thresh < 0.9
    assert f1 == 1.0

    print("✅ Metrics computation works correctly")


def test_trainer_single_epoch():
    """Test that Trainer can run one epoch without errors."""
    from src.model.cnn_model import DeepfakeCNN
    from src.training.trainer import Trainer

    # Create a small dummy dataset
    n_samples = 8
    n_mels = 128
    time_frames = 63  # ~2s at 16kHz with hop=512

    # Create fake DataLoaders
    train_data = torch.utils.data.TensorDataset(
        torch.randn(n_samples, 1, n_mels, time_frames),
        torch.randint(0, 2, (n_samples,)),
    )
    val_data = torch.utils.data.TensorDataset(
        torch.randn(n_samples, 1, n_mels, time_frames),
        torch.randint(0, 2, (n_samples,)),
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=4)

    model = DeepfakeCNN()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=0.001,
            epochs=2,
            patience=5,
            device="cpu",
            save_dir=tmpdir,
        )

        results = trainer.train()

        assert "best_epoch" in results
        assert "final_metrics" in results
        assert len(results["history"]) == 2
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))

        # Test history export
        history_path = trainer.export_history(os.path.join(tmpdir, "history.csv"))
        assert os.path.exists(history_path)

    print("✅ Trainer runs correctly for 2 epochs")


def test_model_forward_pass():
    """Test that CNN model produces correct output shape."""
    from src.model.cnn_model import DeepfakeCNN

    model = DeepfakeCNN()
    model.eval()

    # Batch of 4, 1 channel, 128 mel bins, 63 time frames
    x = torch.randn(4, 1, 128, 63)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0, 1]"

    print("✅ CNN model forward pass produces correct output")


def test_checkpoint_save_load():
    """Test model checkpoint save and load."""
    from src.model.cnn_model import DeepfakeCNN

    model = DeepfakeCNN()
    x = torch.randn(1, 1, 128, 63)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        path = os.path.join(tmpdir, "test_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 5,
        }, path)

        # Load into a new model
        model2 = DeepfakeCNN()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model2.load_state_dict(checkpoint["model_state_dict"])

        # Verify same output
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)

        assert torch.allclose(out1, out2), "Loaded model should produce same output"

    print("✅ Checkpoint save/load works correctly")


if __name__ == "__main__":
    test_metrics()
    test_model_forward_pass()
    test_trainer_single_epoch()
    test_checkpoint_save_load()
    print("\n🎉 All Phase 3 smoke tests passed!")
