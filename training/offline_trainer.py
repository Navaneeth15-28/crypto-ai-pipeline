"""
Offline XGBoost Model Trainer
-----------------------------
Trains a high-frequency binary price-direction classifier (T + 60s horizon):
- Label: 1 for UP (future_return > 0), 0 for DOWN/FLAT (future_return <= 0)
- Enforces strict continuous segmentation (rejects session gaps > 60s)
- Chronological train/validation holdout split (no lookahead bias)
- Evaluates Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC
- Evaluates Deadband Confidence Filtering (e.g. Prob >= 0.60 or <= 0.40)
- Computes baseline comparisons (Majority, Momentum, Random)
- Exports models/model.pkl and models/model_metadata.json
"""

import os
import json
import sqlite3
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "historical_features.db"
MODEL_DIR = "models"
MODEL_PKL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Start timestamp where microstructure features are fully populated
START_TIMESTAMP = 1790605091.587

# Target horizon settings
TARGET_HORIZON_SECONDS = 60
MIN_FUTURE_GAP_SECONDS = 45
MAX_FUTURE_GAP_SECONDS = 75
MAX_CONTINUOUS_GAP_SECONDS = 60

# Train / Holdout split ratio
TRAIN_RATIO = 0.80

# Features to use (Stationary price dynamics + all order flow microstructure)
FEATURE_COLUMNS = [
    # Price returns & momentums
    "price_return_5s",
    "price_return_10s",
    "price_return_30s",
    "price_momentum_30s",
    "price_momentum_60s",
    "momentum_acceleration",

    # Volatility & ranges
    "rolling_std_price_5s",
    "rolling_std_price_10s",
    "rolling_std_price_30s",
    "rolling_std_price_60s",
    "price_range_10s",
    "price_range_30s",
    "price_range_60s",
    "volatility_10s",
    "volatility_30s",

    # Volume & frequency dynamics
    "rolling_volume_5s",
    "rolling_volume_10s",
    "rolling_volume_30s",
    "rolling_volume_60s",
    "volume_acceleration",
    "trade_frequency_5s",
    "trade_frequency_10s",
    "trade_frequency_30s",
    "trade_frequency_60s",
    "trade_frequency_change",

    # 5s Order Flow Microstructure
    "buy_volume_5s",
    "sell_volume_5s",
    "buy_trade_count_5s",
    "sell_trade_count_5s",
    "volume_imbalance_5s",
    "trade_imbalance_5s",
    "buy_sell_ratio_5s",

    # 10s Order Flow Microstructure
    "buy_volume_10s",
    "sell_volume_10s",
    "buy_trade_count_10s",
    "sell_trade_count_10s",
    "volume_imbalance_10s",
    "trade_imbalance_10s",
    "buy_sell_ratio_10s",

    # 30s Order Flow Microstructure
    "buy_volume_30s",
    "sell_volume_30s",
    "buy_trade_count_30s",
    "sell_trade_count_30s",
    "volume_imbalance_30s",
    "trade_imbalance_30s",
    "buy_sell_ratio_30s",

    # 60s Order Flow Microstructure
    "buy_volume_60s",
    "sell_volume_60s",
    "buy_trade_count_60s",
    "sell_trade_count_60s",
    "volume_imbalance_60s",
    "trade_imbalance_60s",
    "buy_sell_ratio_60s",
]


def load_raw_data(db_path, start_ts=START_TIMESTAMP):
    """Loads historical features from SQLite in read-only mode."""
    print("======================================================================")
    print("CRYPTO-AI MLOps: OFFLINE XGBOOST TRAINER")
    print("======================================================================")
    print(f"Connecting to database: {db_path} (Read-Only)...")
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    
    query = """
    SELECT timestamp, price, volume, {cols}
    FROM crypto_features
    WHERE timestamp >= ?
    ORDER BY timestamp ASC
    """.format(cols=", ".join(FEATURE_COLUMNS))
    
    df = pd.read_sql_query(query, conn, params=(start_ts,))
    conn.close()
    
    print(f"Loaded {len(df):,} raw ticks with timestamp >= {start_ts:.3f}")
    return df


def create_continuous_segments(df, max_gap_seconds=MAX_CONTINUOUS_GAP_SECONDS):
    """Assigns segment IDs based on continuous trading intervals."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    time_gaps = df["timestamp"].diff()
    segment_ids = (time_gaps > max_gap_seconds).cumsum()
    df["segment_id"] = segment_ids
    num_segments = segment_ids.nunique()
    print(f"Identified {num_segments} continuous trading segments (max gap: {max_gap_seconds}s)")
    return df


def build_binary_labels(df, horizon_seconds=TARGET_HORIZON_SECONDS):
    """
    Labels each tick based on short-term price direction:
    1 = UP (future_return > 0)
    0 = DOWN / FLAT (future_return <= 0)
    Ensures target calculation stays strictly within the same continuous segment.
    """
    print(f"Generating binary direction labels (T + {horizon_seconds}s horizon)...")
    labeled_segments = []

    for seg_id, segment in df.groupby("segment_id"):
        segment = segment.sort_values("timestamp").reset_index(drop=True)
        timestamps = segment["timestamp"].to_numpy()
        prices = segment["price"].to_numpy()

        target_times = timestamps + horizon_seconds
        future_indices = np.searchsorted(timestamps, target_times, side="left")

        valid_indices = []
        labels = []
        future_returns = []

        for i, j in enumerate(future_indices):
            if j >= len(segment):
                continue

            gap = timestamps[j] - timestamps[i]
            if not (MIN_FUTURE_GAP_SECONDS <= gap <= MAX_FUTURE_GAP_SECONDS):
                continue

            future_ret = (prices[j] - prices[i]) / prices[i]
            
            # Binary target: 1 for UP, 0 for DOWN/FLAT
            label = 1 if future_ret > 0.0 else 0
            
            valid_indices.append(i)
            labels.append(label)
            future_returns.append(future_ret)

        if not valid_indices:
            continue

        labeled_part = segment.iloc[valid_indices].copy()
        labeled_part["target"] = labels
        labeled_part["future_return"] = future_returns
        labeled_segments.append(labeled_part)

    if not labeled_segments:
        raise ValueError("No valid labeled rows generated!")

    clean_df = pd.concat(labeled_segments, ignore_index=True)
    # Drop rows with nulls in any feature column or target
    clean_df = clean_df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    
    print(f"Successfully generated {len(clean_df):,} clean labeled samples.")
    return clean_df


def train_and_evaluate(df):
    """Trains XGBoost on chronological split and computes full metrics."""
    # Chronological Split
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"].astype(int)

    print("\n----------------------------------------------------------------------")
    print("DATA SPLIT SUMMARY (CHRONOLOGICAL)")
    print("----------------------------------------------------------------------")
    print(f"Training samples   : {len(X_train):,} ({TRAIN_RATIO*100:.0f}%)")
    print(f"Test samples (OOS) : {len(X_test):,} ({(1-TRAIN_RATIO)*100:.0f}%)")

    train_dist = y_train.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()
    print(f"Train Class Dist   : 0 (DOWN/FLAT): {train_dist.get(0, 0)*100:.2f}% | 1 (UP): {train_dist.get(1, 0)*100:.2f}%")
    print(f"Test Class Dist    : 0 (DOWN/FLAT): {test_dist.get(0, 0)*100:.2f}% | 1 (UP): {test_dist.get(1, 0)*100:.2f}%")

    # Baseline 1: Majority Class
    majority_class = y_train.mode()[0]
    majority_preds = np.full_like(y_test, majority_class)
    majority_acc = accuracy_score(y_test, majority_preds)

    # Baseline 2: Momentum (Predict UP if 60s momentum > 0, else 0)
    momentum_preds = (test_df["price_momentum_60s"] > 0).astype(int).to_numpy()
    momentum_acc = accuracy_score(y_test, momentum_preds)

    # Train XGBoost
    print("\n----------------------------------------------------------------------")
    print("TRAINING XGBOOST CLASSIFIER")
    print("----------------------------------------------------------------------")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # Predict Probabilities and Classes
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.50).astype(int)

    acc = accuracy_score(y_test, test_preds)
    bal_acc = balanced_accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)

    print("\n======================================================================")
    print("OUT-OF-SAMPLE TEST RESULTS (Standard 0.50 Threshold)")
    print("======================================================================")
    print(f"Model Accuracy     : {acc:.4f}  (Majority Baseline: {majority_acc:.4f}, Momentum: {momentum_acc:.4f})")
    print(f"Balanced Accuracy  : {bal_acc:.4f}")
    print(f"ROC-AUC Score      : {roc_auc:.4f}")
    print(f"Precision (Class 1): {prec:.4f}")
    print(f"Recall (Class 1)   : {rec:.4f}")
    print(f"F1-Score (Class 1) : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"              Predicted 0    Predicted 1")
    print(f"Actual 0      {cm[0,0]:11d}    {cm[0,1]:11d}")
    print(f"Actual 1      {cm[1,0]:11d}    {cm[1,1]:11d}")

    # Deadband Confidence Evaluation (Quant Upgrade)
    print("\n----------------------------------------------------------------------")
    print("QUANT DEAD-BAND EVALUATION (High Confidence Filters)")
    print("----------------------------------------------------------------------")
    
    deadband_results = {}
    for threshold in [0.55, 0.60, 0.65]:
        up_mask = test_probs >= threshold
        down_mask = test_probs <= (1.0 - threshold)
        active_mask = up_mask | down_mask
        
        if active_mask.sum() > 0:
            filtered_preds = np.where(up_mask[active_mask], 1, 0)
            filtered_actual = y_test.to_numpy()[active_mask]
            f_acc = accuracy_score(filtered_actual, filtered_preds)
            f_prec = precision_score(filtered_actual, filtered_preds, zero_division=0)
            coverage = active_mask.mean() * 100
            print(f"Threshold >= {threshold:.2f} / <= {1-threshold:.2f} | Coverage: {coverage:5.1f}% | Accuracy: {f_acc:.4f} | Precision: {f_prec:.4f}")
            deadband_results[str(threshold)] = {
                "coverage_pct": round(coverage, 2),
                "accuracy": round(float(f_acc), 4),
                "precision": round(float(f_prec), 4)
            }

    # Feature Importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n----------------------------------------------------------------------")
    print("TOP 15 MOST INFLUENTIAL FEATURES")
    print("----------------------------------------------------------------------")
    for i, row in importance_df.head(15).iterrows():
        print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

    # Compute baseline feature statistics for drift monitoring in UI
    feature_stats = {}
    for col in FEATURE_COLUMNS:
        feature_stats[col] = {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "p25": float(X_train[col].quantile(0.25)),
            "p50": float(X_train[col].median()),
            "p75": float(X_train[col].quantile(0.75)),
        }

    metrics = {
        "model_type": "XGBClassifier",
        "task": "binary_price_direction_60s",
        "horizon_seconds": TARGET_HORIZON_SECONDS,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "majority_baseline_acc": round(float(majority_acc), 4),
            "momentum_baseline_acc": round(float(momentum_acc), 4)
        },
        "deadband_filters": deadband_results,
        "features": FEATURE_COLUMNS,
        "top_features": importance_df.head(15)["feature"].tolist(),
        "feature_stats": feature_stats
    }

    return model, metrics


def save_artifacts(model, metrics):
    """Saves model.pkl and model_metadata.json into models/ and workspace root."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Save in models/
    with open(MODEL_PKL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[OK] Saved model artifact to: {MODEL_PKL_PATH}")

    # 2. Also save model.pkl in root directory for seamless inference script compatibility
    root_pkl_path = "model.pkl"
    with open(root_pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[OK] Saved model artifact to root: {root_pkl_path}")

    # 3. Save metadata JSON
    with open(METADATA_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved metadata & drift baseline to: {METADATA_PATH}")


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database {DB_PATH} not found!")

    df = load_raw_data(DB_PATH)
    df = create_continuous_segments(df)
    labeled_df = build_binary_labels(df)
    
    model, metrics = train_and_evaluate(labeled_df)
    save_artifacts(model, metrics)
    
    print("\n======================================================================")
    print("TRAINING COMPLETE - READY FOR STREAMING INFERENCE")
    print("======================================================================")


if __name__ == "__main__":
    main()
