"""
Data Processor — Load CSV, feature engineering, and PyTorch dataset creation.
Handles both the Indian Smart Traffic Dataset and generated sample data.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

FEATURE_COLS = [
    "vehicle_count", "avg_speed", "hour", "minute",
    "day_of_week", "is_weekend", "is_rush_hour",
    "time_sin", "time_cos", "day_sin", "day_cos"
]

CONGESTION_CLASSES = ["Low", "Medium", "High"]


def find_dataset() -> str:
    candidates = [
        os.path.join(DATA_DIR, "indian_smart_traffic.csv"),
        os.path.join(DATA_DIR, "smart_traffic.csv"),
        os.path.join(DATA_DIR, "traffic_data.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    gen = os.path.join(DATA_DIR, "sample_generator.py")
    if os.path.exists(gen):
        print("[DataProcessor] No dataset found. Generating sample data...")
        import subprocess, sys
        subprocess.run([sys.executable, gen], check=True)
        default = os.path.join(DATA_DIR, "indian_smart_traffic.csv")
        if os.path.exists(default):
            return default

    raise FileNotFoundError(f"No dataset in {DATA_DIR}. Run: python data/sample_generator.py")


def load_and_engineer(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = find_dataset()

    df = pd.read_csv(csv_path)
    print(f"[DataProcessor] Loaded {len(df):,} rows from {os.path.basename(csv_path)}")

    # Standardize column names
    renames = {}
    for col in df.columns:
        lc = col.strip().lower().replace(" ", "_")
        if "intersect" in lc and "id" in lc:
            renames[col] = "intersection_id"
        elif "intersect" in lc and "name" in lc:
            renames[col] = "intersection_name"
        elif "lane" in lc:
            renames[col] = "lane_id"
        elif "time" in lc or "date" in lc:
            renames[col] = "timestamp"
        elif "vehicle" in lc and "count" in lc:
            renames[col] = "vehicle_count"
        elif "speed" in lc:
            renames[col] = "avg_speed"
        elif "congestion" in lc or "level" in lc:
            renames[col] = "congestion_level"
        elif "city" in lc:
            renames[col] = "city"
    if renames:
        df = df.rename(columns=renames)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Feature engineering
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = ((df["hour"].between(8, 10)) | (df["hour"].between(17, 20))).astype(int)
    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Add lane_id if missing (aggregate data)
    if "lane_id" not in df.columns:
        df["lane_id"] = "Lane_N"

    # Generate congestion labels if missing :
    if "congestion_level" not in df.columns:
        conditions = [
            (df["avg_speed"] < 10) | (df["vehicle_count"] > 45),
            (df["avg_speed"] < 18) | (df["vehicle_count"] > 25),
        ]
        df["congestion_level"] = np.select(conditions, ["High", "Medium"], default="Low")

    df["vehicle_count"] = pd.to_numeric(df["vehicle_count"], errors="coerce").fillna(0).astype(int)
    df["avg_speed"] = pd.to_numeric(df["avg_speed"], errors="coerce").fillna(0).astype(float)

    print(f"[DataProcessor] Congestion distribution:\n{df['congestion_level'].value_counts().to_string()}")
    return df


class TrafficDataset(Dataset):
    """PyTorch dataset for traffic congestion prediction with sequence windows."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = 12):
        self.seq_len = seq_len
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return x, y


def prepare_datasets(
    df: pd.DataFrame = None,
    seq_len: int = 12,
    test_size: float = 0.2,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, StandardScaler, LabelEncoder, Dict]:
    """Prepare train/test DataLoaders from the traffic DataFrame."""
    if df is None:
        df = load_and_engineer()

    label_encoder = LabelEncoder()
    label_encoder.fit(CONGESTION_CLASSES)
    labels = label_encoder.transform(df["congestion_level"].values)

    scaler = StandardScaler()
    features = scaler.fit_transform(df[FEATURE_COLS].values)

    split_idx = int(len(features) * (1 - test_size))
    train_features, test_features = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    train_dataset = TrafficDataset(train_features, train_labels, seq_len)
    test_dataset = TrafficDataset(test_features, test_labels, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    info = {
        "num_features": len(FEATURE_COLS),
        "num_classes": len(CONGESTION_CLASSES),
        "seq_len": seq_len,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "feature_names": FEATURE_COLS,
        "class_names": CONGESTION_CLASSES,
    }

    print(f"[DataProcessor] Train: {info['train_size']:,} | Test: {info['test_size']:,} | Seq: {seq_len}")
    return train_loader, test_loader, scaler, label_encoder, info
