"""
PyTorch Deep Sequence Model Trainer (LSTM with Temporal Attention)
-------------------------------------------------------------------
Trains an academic-grade Deep Recurrent Neural Network (LSTM + Self-Attention)
for high-frequency Bitcoin price direction forecasting:
- Architecture: 2-layer Stacked LSTM + Temporal Self-Attention + MLP Head
- Enforces strict continuous segmentation (no sequence lookahead across gaps > 60s)
- Chronological train/validation holdout split (80/20)
- Computes Train/Val Loss, Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC
- Evaluates Quant Deadband Confidence Filtering (Threshold >= 0.55 / <= 0.45)
- Saves models/dl_lstm_model.pth and models/dl_model_metadata.json
"""

# IMPORTANT FOR WINDOWS: import torch first to avoid MKL/OpenMP DLL initialization conflicts
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import sys
import json
import sqlite3
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Workspace Root
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = os.path.join(WORKSPACE_DIR, "historical_features.db")
MODEL_DIR = os.path.join(WORKSPACE_DIR, "models")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "dl_lstm_model.pth")
DL_METADATA_PATH = os.path.join(MODEL_DIR, "dl_model_metadata.json")

START_TIMESTAMP = 1790605091.587
TARGET_HORIZON_SECONDS = 60
MIN_FUTURE_GAP_SECONDS = 45
MAX_FUTURE_GAP_SECONDS = 75
MAX_CONTINUOUS_GAP_SECONDS = 60

# Deep Learning Sequence Hyperparameters
SEQUENCE_LENGTH = 10      # 10 consecutive ticks sequence window
HIDDEN_DIM = 64          # LSTM hidden state size
NUM_LAYERS = 2           # 2-layer stacked LSTM
DROPOUT = 0.25           # Dropout rate
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 15
TRAIN_RATIO = 0.80

# 53 Microstructure & Price Dynamics Features
FEATURE_COLUMNS = [
    "price_return_5s", "price_return_10s", "price_return_30s",
    "price_momentum_30s", "price_momentum_60s", "momentum_acceleration",
    "rolling_std_price_5s", "rolling_std_price_10s", "rolling_std_price_30s",
    "rolling_std_price_60s", "price_range_10s", "price_range_30s",
    "price_range_60s", "volatility_10s", "volatility_30s",
    "rolling_volume_5s", "rolling_volume_10s", "rolling_volume_30s",
    "rolling_volume_60s", "volume_acceleration", "trade_frequency_5s",
    "trade_frequency_10s", "trade_frequency_30s", "trade_frequency_60s",
    "trade_frequency_change",
    # Microstructure 5s
    "buy_volume_5s", "sell_volume_5s", "buy_trade_count_5s",
    "sell_trade_count_5s", "volume_imbalance_5s", "trade_imbalance_5s",
    "buy_sell_ratio_5s",
    # Microstructure 10s
    "buy_volume_10s", "sell_volume_10s", "buy_trade_count_10s",
    "sell_trade_count_10s", "volume_imbalance_10s", "trade_imbalance_10s",
    "buy_sell_ratio_10s",
    # Microstructure 30s
    "buy_volume_30s", "sell_volume_30s", "buy_trade_count_30s",
    "sell_trade_count_30s", "volume_imbalance_30s", "trade_imbalance_30s",
    "buy_sell_ratio_30s",
    # Microstructure 60s
    "buy_volume_60s", "sell_volume_60s", "buy_trade_count_60s",
    "sell_trade_count_60s", "volume_imbalance_60s", "trade_imbalance_60s",
    "buy_sell_ratio_60s"
]

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# PYTORCH DEEP LEARNING ARCHITECTURE
# ============================================================

class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism over LSTM hidden states.
    Allows the model to dynamically weight which time steps in the
    sequence window matter most for price direction.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_output):
        # rnn_output shape: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_output)              # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)      # (batch_size, seq_len, 1)
        context = torch.sum(weights * rnn_output, dim=1) # (batch_size, hidden_dim)
        return context, weights


class CryptoLSTMWithAttention(nn.Module):
    """
    Stacked 2-Layer LSTM with Temporal Attention and MLP Classifier Head.
    """
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(CryptoLSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention layer
        self.attention = TemporalAttention(hidden_dim)

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)                  # (batch_size, seq_len, hidden_dim)
        context, attn_weights = self.attention(lstm_out) # (batch_size, hidden_dim)
        logits = self.classifier(context).squeeze(-1)    # (batch_size,)
        return logits, attn_weights


# ============================================================
# DATASET & CONTINUOUS SEQUENCE CREATION
# ============================================================

class SequentialCryptoDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_and_preprocess_data():
    """Loads feature store from SQLite and enforces continuous temporal segmentation."""
    print("======================================================================", flush=True)
    print("DEEP LEARNING: PYTORCH LSTM TRAINING PIPELINE", flush=True)
    print("======================================================================", flush=True)
    print(f"Connecting to feature store: {DB_PATH}", flush=True)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    query = """
    SELECT timestamp, price, volume, {cols}
    FROM crypto_features
    WHERE timestamp >= ?
    ORDER BY timestamp ASC
    """.format(cols=", ".join(FEATURE_COLUMNS))

    df = pd.read_sql_query(query, conn, params=(START_TIMESTAMP,))
    conn.close()
    print(f"Loaded {len(df):,} raw tick records.", flush=True)

    # 1. Segment continuous trading intervals
    df = df.sort_values("timestamp").reset_index(drop=True)
    time_gaps = df["timestamp"].diff()
    df["segment_id"] = (time_gaps > MAX_CONTINUOUS_GAP_SECONDS).cumsum()

    # 2. Generate T+60s binary labels strictly within the same continuous segment
    labeled_segments = []
    for seg_id, segment in df.groupby("segment_id"):
        segment = segment.sort_values("timestamp").reset_index(drop=True)
        timestamps = segment["timestamp"].to_numpy()
        prices = segment["price"].to_numpy()

        target_times = timestamps + TARGET_HORIZON_SECONDS
        future_indices = np.searchsorted(timestamps, target_times, side="left")

        valid_idx, labels = [], []
        for i, j in enumerate(future_indices):
            if j >= len(segment):
                continue
            gap = timestamps[j] - timestamps[i]
            if not (MIN_FUTURE_GAP_SECONDS <= gap <= MAX_FUTURE_GAP_SECONDS):
                continue

            future_ret = (prices[j] - prices[i]) / prices[i]
            label = 1 if future_ret > 0.0 else 0
            valid_idx.append(i)
            labels.append(label)

        if not valid_idx:
            continue

        part = segment.iloc[valid_idx].copy()
        part["target"] = labels
        labeled_segments.append(part)

    clean_df = pd.concat(labeled_segments, ignore_index=True)
    clean_df = clean_df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    print(f"Generated {len(clean_df):,} clean labeled samples across continuous segments.", flush=True)

    return clean_df


def build_sequences(df, seq_len=SEQUENCE_LENGTH):
    """
    Creates temporal sequence windows of length seq_len strictly within continuous segments.
    Input shape per sample: (seq_len, num_features). Target: label of the last step.
    """
    # Chronological Split
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Fit StandardScaler only on training set
    scaler = StandardScaler()
    train_scaled_features = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_scaled_features = scaler.transform(test_df[FEATURE_COLUMNS])

    train_df["scaled_features"] = list(train_scaled_features)
    test_df["scaled_features"] = list(test_scaled_features)

    def extract_sequences(dataframe):
        X_seqs = []
        y_labels = []
        for _, segment in dataframe.groupby("segment_id"):
            if len(segment) < seq_len:
                continue
            feats = np.stack(segment["scaled_features"].values)
            targets = segment["target"].values
            for i in range(len(segment) - seq_len + 1):
                X_seqs.append(feats[i : i + seq_len])
                y_labels.append(targets[i + seq_len - 1])
        return np.array(X_seqs, dtype=np.float32), np.array(y_labels, dtype=np.int64)

    X_train, y_train = extract_sequences(train_df)
    X_test, y_test = extract_sequences(test_df)

    print("\n----------------------------------------------------------------------", flush=True)
    print(f"SEQUENCE DATASET SUMMARY (Window Length = {seq_len} Ticks)", flush=True)
    print("----------------------------------------------------------------------", flush=True)
    print(f"Train Sequences : {X_train.shape} | Labels: {y_train.shape}", flush=True)
    print(f"Test Sequences  : {X_test.shape}  | Labels: {y_test.shape}", flush=True)
    print(f"Train Class Dist: 0: {(y_train == 0).mean()*100:.2f}% | 1: {(y_train == 1).mean()*100:.2f}%", flush=True)
    print(f"Test Class Dist : 0: {(y_test == 0).mean()*100:.2f}% | 1: {(y_test == 1).mean()*100:.2f}%", flush=True)

    return X_train, y_train, X_test, y_test, scaler


# ============================================================
# TRAINING & EVALUATION LOOP
# ============================================================

def train_pytorch_model(X_train, y_train, X_test, y_test, scaler):
    train_dataset = SequentialCryptoDataset(X_train, y_train)
    test_dataset = SequentialCryptoDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}", flush=True)

    input_dim = len(FEATURE_COLUMNS)
    model = CryptoLSTMWithAttention(input_dim=input_dim).to(device)

    # Class weight to penalize minority class mistakes
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print("\n----------------------------------------------------------------------", flush=True)
    print("TRAINING NEURAL NETWORK (EPOCHS & LOSS TRACKING)", flush=True)
    print("----------------------------------------------------------------------", flush=True)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_model_state = None

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * len(batch_y)

        avg_train_loss = total_train_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item() * len(batch_y)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.50).astype(int)
                all_val_preds.extend(preds)
                all_val_targets.extend(batch_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_dataset)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        scheduler.step(avg_val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            mark = "* (Saved Best)"
        else:
            mark = ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc*100:5.2f}% {mark}",
            flush=True
        )

    train_time = time.time() - t0
    print(f"\n[OK] Deep Learning training finished in {train_time:.2f}s.", flush=True)

    # Load best model weights for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Full Out-of-Sample Evaluation
    model.eval()
    test_probs = []
    test_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(batch_y.numpy())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    test_preds = (test_probs >= 0.50).astype(int)

    acc = accuracy_score(test_targets, test_preds)
    bal_acc = balanced_accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, zero_division=0)
    rec = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)
    roc_auc = roc_auc_score(test_targets, test_probs)
    cm = confusion_matrix(test_targets, test_preds)

    print("\n======================================================================", flush=True)
    print("FINAL OUT-OF-SAMPLE TEST EVALUATION (PyTorch LSTM + Attention)", flush=True)
    print("======================================================================", flush=True)
    print(f"Accuracy         : {acc:.4f}", flush=True)
    print(f"Balanced Accuracy: {bal_acc:.4f}", flush=True)
    print(f"ROC-AUC Score    : {roc_auc:.4f}", flush=True)
    print(f"Precision (UP)   : {prec:.4f}", flush=True)
    print(f"Recall (UP)      : {rec:.4f}", flush=True)
    print(f"F1-Score (UP)    : {f1:.4f}", flush=True)
    print(f"\nConfusion Matrix:\n{cm}", flush=True)

    # Deadband Confidence Evaluation
    print("\n----------------------------------------------------------------------", flush=True)
    print("QUANT DEAD-BAND EVALUATION (High Conviction Signals)", flush=True)
    print("----------------------------------------------------------------------", flush=True)
    deadband_results = {}
    for threshold in [0.55, 0.60, 0.65]:
        up_mask = test_probs >= threshold
        down_mask = test_probs <= (1.0 - threshold)
        active_mask = up_mask | down_mask

        if active_mask.sum() > 0:
            filtered_preds = np.where(up_mask[active_mask], 1, 0)
            filtered_actual = test_targets[active_mask]
            f_acc = accuracy_score(filtered_actual, filtered_preds)
            f_prec = precision_score(filtered_actual, filtered_preds, zero_division=0)
            coverage = active_mask.mean() * 100
            print(f"Threshold >= {threshold:.2f} / <= {1-threshold:.2f} | Coverage: {coverage:5.1f}% | Accuracy: {f_acc:.4f} | Precision: {f_prec:.4f}", flush=True)
            deadband_results[str(threshold)] = {
                "coverage_pct": round(coverage, 2),
                "accuracy": round(float(f_acc), 4),
                "precision": round(float(f_prec), 4)
            }

    # Save Model Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), DL_MODEL_PATH)
    print(f"\n[OK] Saved PyTorch weights to: {DL_MODEL_PATH}", flush=True)

    # Metadata for UI & academic reporting
    meta = {
        "model_type": "PyTorch_LSTM_With_Temporal_Attention",
        "framework": f"PyTorch {torch.__version__}",
        "architecture": {
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "sequence_length": SEQUENCE_LENGTH,
            "attention": "Temporal_Self_Attention",
            "classifier_head": "Linear(64->32) + BatchNorm1d + ReLU + Dropout(0.25) + Linear(32->1)"
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss_weighted",
            "train_time_seconds": round(train_time, 2),
            "history": history
        },
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4)
        },
        "deadband_filters": deadband_results,
        "features": FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }

    with open(DL_METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"[OK] Saved Deep Learning metadata to: {DL_METADATA_PATH}", flush=True)

    return model, meta


def main():
    df = load_and_preprocess_data()
    X_train, y_train, X_test, y_test, scaler = build_sequences(df)
    train_pytorch_model(X_train, y_train, X_test, y_test, scaler)

    print("\n======================================================================", flush=True)
    print("PYTORCH DEEP LEARNING MODEL UPGRADE COMPLETE!", flush=True)
    print("======================================================================", flush=True)


if __name__ == "__main__":
    main()
