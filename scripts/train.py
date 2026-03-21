#!/usr/bin/env python3
"""
Training Script — Launches model training with the prepared dataset.

Usage:
    # Train with defaults (uses config.yaml settings):
    python scripts/train.py

    # Train with custom settings:
    python scripts/train.py --epochs 30 --batch-size 16 --lr 0.0005

    # Train with demo data (auto-generates if needed):
    python scripts/train.py --demo

    # Resume from checkpoint:
    python scripts/train.py --resume models/checkpoint_epoch_10.pt
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd

from src.model.cnn_model import DeepfakeCNN
from src.data.dataset_loader import load_manifest, create_manifest, get_dataset_stats
from src.data.splitter import split_dataset, get_subset
from src.data.audio_dataset import create_dataloader, DeepfakeAudioDataset
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, compute_confusion
from src.utils.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE,
    MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, N_MELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train deepfake voice detection model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--demo", action="store_true", help="Use demo data (auto-generate)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Deepfake Voice Detection — Model Training")
    print("=" * 60)

    # ── Step 1: Load dataset manifest ──
    manifest_path = os.path.join(PROCESSED_DATA_DIR, "manifest.csv")

    if args.demo and not os.path.exists(manifest_path):
        print("\n📦 Generating demo data first...")
        from scripts.prepare_dataset import generate_demo_data
        generate_demo_data(RAW_DATA_DIR, n_per_class=15)
        df = create_manifest()
        df = split_dataset(df, force_resplit=True)
        df.to_csv(manifest_path, index=False)
    elif not os.path.exists(manifest_path):
        print(f"\n❌ Manifest not found at {manifest_path}")
        print("   Run `python scripts/prepare_dataset.py` first.")
        sys.exit(1)

    df = load_manifest(manifest_path)

    # Ensure val split exists
    if "val" not in df["subset"].unique():
        df = split_dataset(df, force_resplit=True)
        df.to_csv(manifest_path, index=False)

    stats = get_dataset_stats(df)
    print(f"\n📊 Dataset: {stats['total_files']} files")
    for subset, dist in stats["label_by_subset"].items():
        print(f"   {subset}: {dist}")

    # ── Step 2: Create DataLoaders ──
    train_df = get_subset(df, "train")
    val_df = get_subset(df, "val")

    if len(train_df) == 0 or len(val_df) == 0:
        print("❌ Empty train or val set. Check your dataset.")
        sys.exit(1)

    train_loader = create_dataloader(
        train_df,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        balanced=True,
    )
    val_loader = create_dataloader(
        val_df,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        balanced=False,
    )

    print(f"\n🔧 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Step 3: Create model ──
    model = DeepfakeCNN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model: DeepfakeCNN ({trainable_params:,} trainable params)")

    # Resume from checkpoint if requested
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"📂 Resumed from: {args.resume}")

    # ── Step 4: Get class weights for balanced training ──
    train_dataset = DeepfakeAudioDataset(train_df, augment=False)
    class_weights = train_dataset.get_label_weights()
    print(f"⚖️  Class weights: real={class_weights[0]:.3f}, fake={class_weights[1]:.3f}")

    # ── Step 5: Train ──
    print(f"\n🚀 Training for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"   Device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"   LR: {args.lr}, Batch: {args.batch_size}\n")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        save_dir=MODELS_DIR,
        class_weights=class_weights,
    )

    results = trainer.train()

    # ── Step 6: Print results ──
    print("\n" + "=" * 60)
    print("  Training Results")
    print("=" * 60)
    print(f"  Best Epoch:          {results['best_epoch']}")
    print(f"  Best Val Loss:       {results['best_val_loss']:.4f}")
    print(f"  Training Time:       {results['training_time_seconds']:.1f}s")
    print(f"  Optimal Threshold:   {results['optimal_threshold']:.2f}")

    fm = results["final_metrics"]
    print(f"\n  Final Metrics (on validation set):")
    print(f"    Accuracy:   {fm['accuracy']:.4f}")
    print(f"    Precision:  {fm['precision']:.4f}")
    print(f"    Recall:     {fm['recall']:.4f}")
    print(f"    F1 Score:   {fm['f1']:.4f}")
    print(f"    AUC:        {fm['auc']:.4f}")
    print(f"    EER:        {fm['eer']:.4f}")

    # Save history
    history_path = trainer.export_history()
    print(f"\n📈 History saved to: {history_path}")
    print(f"💾 Best model saved to: {os.path.join(MODELS_DIR, 'best_model.pt')}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
