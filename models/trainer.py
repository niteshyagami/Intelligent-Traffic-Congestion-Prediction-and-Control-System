"""
Training Script for the Bi-LSTM + Attention Prediction Model.
Handles training loop, evaluation, early stopping, and model saving.
Optimized for CPU training with data subsampling.
"""

import os
import sys
import io
import json
import numpy as np

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.data_processor import load_and_engineer, prepare_datasets, FEATURE_COLS, CONGESTION_CLASSES
from models.predictor import create_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")


def train_model(
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.001,
    seq_len: int = 12,
    patience: int = 6,
    device: str = None,
    max_train_samples: int = 50000,
):
    """Train the Bi-LSTM + Attention model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Trainer] Device: {device}", flush=True)

    # Load data :
    df = load_and_engineer()

    # Subsample for faster CPU training while preserving class distribution
    if len(df) > max_train_samples * 1.25:
        print(f"[Trainer] Subsampling from {len(df):,} to ~{max_train_samples:,} rows for faster training", flush=True)
        df = df.groupby("congestion_level", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1000, int(max_train_samples * len(x) / len(df)))),
                               random_state=42)
        ).reset_index(drop=True)
        print(f"[Trainer] After sampling: {len(df):,} rows", flush=True)

    train_loader, test_loader, scaler, label_encoder, info = prepare_datasets(
        df, seq_len=seq_len, batch_size=batch_size
    )

    # Create model
    model = create_model(
        input_dim=info["num_features"],
        num_classes=info["num_classes"],
        device=device,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*60}", flush=True)
    print(f"  Training Bi-LSTM + Attention Model", flush=True)
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Seq: {seq_len}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # --- Evaluate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            improved = " *BEST*"
            _save_model(model, scaler, label_encoder, info, history)
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{improved}",
            flush=True,
        )

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience: {patience})", flush=True)
            break

    # Final report
    print(f"\n{'='*60}", flush=True)
    print(f"  Training Complete! Best Val Accuracy: {best_val_acc:.4f}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nClassification Report:", flush=True)
    try:
        labels = list(range(len(CONGESTION_CLASSES)))
        print(classification_report(all_labels, all_preds, target_names=CONGESTION_CLASSES, labels=labels, zero_division=0), flush=True)
    except Exception as e:
        print(f"  (Report skipped: {e})", flush=True)

    return model, history


def _save_model(model, scaler, label_encoder, info, history):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "predictor.pth"))

    import joblib
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    metadata = {
        "model_type": "BiLSTM_MultiHeadAttention",
        "input_dim": info["num_features"],
        "num_classes": info["num_classes"],
        "seq_len": info["seq_len"],
        "feature_names": info["feature_names"],
        "class_names": info["class_names"],
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [Saved] Model, scaler, encoder -> {MODELS_DIR}", flush=True)


if __name__ == "__main__":
    train_model(epochs=30, batch_size=128, lr=0.001, seq_len=12, max_train_samples=50000)
