"""
Model Trainer — Full training loop with validation, early stopping,
checkpointing, and metrics logging.

Handles:
  - Training loop with BCELoss (binary classification)
  - Validation after each epoch
  - Early stopping based on validation loss
  - Model checkpoint saving (best + periodic)
  - Metrics tracking & history export
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_metrics, find_optimal_threshold
from src.utils.config import (
    MODELS_DIR, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE, BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trains a DeepfakeCNN model with full pipeline:
    train → validate → checkpoint → early stop.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = LEARNING_RATE,
        epochs: int = EPOCHS,
        patience: int = EARLY_STOPPING_PATIENCE,
        device: Optional[str] = None,
        save_dir: str = MODELS_DIR,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            model: The CNN model to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            lr: Learning rate
            epochs: Maximum number of epochs
            patience: Early stopping patience (epochs without improvement)
            device: 'cpu', 'cuda', or None (auto-detect)
            save_dir: Directory to save model checkpoints
            class_weights: Optional class weights for imbalanced data
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.save_dir = save_dir

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        logger.info(f"Training on: {self.device}")

        # Loss function (with optional class weighting)
        if class_weights is not None:
            # For BCELoss, we use pos_weight (ratio of neg/pos)
            pos_weight = class_weights[1] / class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]).to(self.device)
            )
            self._use_logits = True
        else:
            self.criterion = nn.BCELoss()
            self._use_logits = False

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3,
        )

        # Tracking
        self.history: List[Dict] = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        os.makedirs(save_dir, exist_ok=True)

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Dict with final metrics and training history
        """
        logger.info(f"Starting training: {self.epochs} epochs, "
                     f"patience={self.patience}")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # Train one epoch
            train_loss, train_preds, train_labels = self._train_epoch(epoch)

            # Validate
            val_loss, val_preds, val_labels = self._validate_epoch(epoch)

            # Compute metrics
            train_metrics = compute_metrics(train_labels, train_preds)
            val_metrics = compute_metrics(val_labels, val_preds)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Log epoch results
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_metrics["accuracy"],
                "val_acc": val_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "val_eer": val_metrics["eer"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_record)

            logger.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, "
                f"Val EER: {val_metrics['eer']:.4f}"
            )

            # Checkpoint: save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"  ★ New best model (val_loss={val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        elapsed = time.time() - start_time
        logger.info(
            f"Training complete in {elapsed:.1f}s. "
            f"Best epoch: {self.best_epoch}, "
            f"Best val_loss: {self.best_val_loss:.4f}"
        )

        # Final metrics on best model
        self._load_best_model()
        _, final_preds, final_labels = self._validate_epoch(0)
        final_metrics = compute_metrics(final_labels, final_preds)
        opt_thresh, opt_f1 = find_optimal_threshold(final_labels, final_preds)

        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "final_metrics": final_metrics,
            "optimal_threshold": opt_thresh,
            "optimal_f1": opt_f1,
            "training_time_seconds": elapsed,
            "history": self.history,
        }

    def _train_epoch(
        self, epoch: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Train for one epoch. Returns (loss, predictions, labels)."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )

        for batch_mel, batch_labels in pbar:
            batch_mel = batch_mel.to(self.device)
            batch_labels = batch_labels.float().to(self.device)

            self.optimizer.zero_grad()

            if self._use_logits:
                # BCEWithLogitsLoss expects raw logits
                # But our model has sigmoid — so we need to adjust
                outputs = self.model(batch_mel).squeeze(-1)
                # Convert probabilities back to logits for BCEWithLogitsLoss
                outputs_logit = torch.log(outputs / (1 - outputs + 1e-7) + 1e-7)
                loss = self.criterion(outputs_logit, batch_labels)
                preds = outputs.detach().cpu().numpy()
            else:
                outputs = self.model(batch_mel).squeeze(-1)
                loss = self.criterion(outputs, batch_labels)
                preds = outputs.detach().cpu().numpy()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch_mel.size(0)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss, np.array(all_preds), np.array(all_labels)

    @torch.no_grad()
    def _validate_epoch(
        self, epoch: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Validate for one epoch. Returns (loss, predictions, labels)."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_mel, batch_labels in self.val_loader:
            batch_mel = batch_mel.to(self.device)
            batch_labels = batch_labels.float().to(self.device)

            outputs = self.model(batch_mel).squeeze(-1)

            if self._use_logits:
                outputs_logit = torch.log(outputs / (1 - outputs + 1e-7) + 1e-7)
                loss = self.criterion(outputs_logit, batch_labels)
            else:
                loss = self.criterion(outputs, batch_labels)

            total_loss += loss.item() * batch_mel.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        avg_loss = total_loss / max(len(self.val_loader.dataset), 1)
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": self.best_val_loss,
            "history": self.history,
        }

        if is_best:
            path = os.path.join(self.save_dir, "best_model.pt")
        else:
            path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def _load_best_model(self):
        """Load the best model checkpoint."""
        path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            logger.warning("No best model checkpoint found")

    def export_history(self, path: Optional[str] = None) -> str:
        """Export training history to CSV."""
        import pandas as pd

        if path is None:
            path = os.path.join(self.save_dir, "training_history.csv")

        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)
        logger.info(f"Training history saved to {path}")
        return path
