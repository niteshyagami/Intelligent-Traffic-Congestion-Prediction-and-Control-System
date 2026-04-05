"""Convenience training script — run from project root."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
from trainer import train_model

if __name__ == "__main__":
    train_model(epochs=50, batch_size=64, lr=0.001, seq_len=12)
