import sqlite3
import os
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "historical_features.db"

# A gap larger than this means the data is considered
# disconnected.
MAX_CONTINUOUS_GAP_SECONDS = 60

# Future target horizon
TARGET_HORIZON_SECONDS = 30

# Allowed future timestamp difference
MIN_FUTURE_GAP_SECONDS = 20
MAX_FUTURE_GAP_SECONDS = 40


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [

    "rolling_mean_price_5s",
    "rolling_std_price_5s",
    "rolling_volume_5s",
    "trade_frequency_5s",
    "price_return_5s",

    "rolling_mean_price_10s",
    "rolling_std_price_10s",
    "rolling_volume_10s",
    "trade_frequency_10s",
    "price_return_10s",
    "price_range_10s",
    "volatility_10s",

    "rolling_mean_price_30s",
    "rolling_std_price_30s",
    "rolling_volume_30s",
    "trade_frequency_30s",
    "price_return_30s",
    "price_momentum_30s",
    "price_range_30s",
    "volatility_30s",

    "rolling_mean_price_60s",
    "rolling_std_price_60s",
    "rolling_volume_60s",
    "trade_frequency_60s",
    "price_momentum_60s",
    "price_range_60s",

    "momentum_acceleration",
    "volume_acceleration",
    "trade_frequency_change",
    "volume_imbalance_60s",

    "buy_volume_5s",
    "sell_volume_5s",
    "buy_trade_count_5s",
    "sell_trade_count_5s",
    "volume_imbalance_5s",
    "trade_imbalance_5s",
    "buy_sell_ratio_5s",

    "buy_volume_10s",
    "sell_volume_10s",
    "buy_trade_count_10s",
    "sell_trade_count_10s",
    "volume_imbalance_10s",
    "trade_imbalance_10s",
    "buy_sell_ratio_10s",

    "buy_volume_30s",
    "sell_volume_30s",
    "buy_trade_count_30s",
    "sell_trade_count_30s",
    "volume_imbalance_30s",
    "trade_imbalance_30s",
    "buy_sell_ratio_30s",

    "buy_volume_60s",
    "sell_volume_60s",
    "buy_trade_count_60s",
    "sell_trade_count_60s",
    "volume_imbalance_60s",
    "trade_imbalance_60s",
    "buy_sell_ratio_60s",
]


# ============================================================
# DATA LOGGER
# ============================================================

class DataLogger:

    def __init__(self, db_path="historical_features.db"):

        self.db_path = db_path

        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )

        self.create_table()
        self.add_new_columns()

    def create_table(self):

        columns = [

            "timestamp REAL PRIMARY KEY",
            "price REAL",
            "volume REAL",

            "rolling_mean_price_5s REAL",
            "rolling_std_price_5s REAL",
            "rolling_volume_5s REAL",
            "trade_frequency_5s REAL",
            "price_return_5s REAL",

            "rolling_mean_price_10s REAL",
            "rolling_std_price_10s REAL",
            "rolling_volume_10s REAL",
            "trade_frequency_10s REAL",
            "price_return_10s REAL",
            "price_range_10s REAL",
            "volatility_10s REAL",

            "rolling_mean_price_30s REAL",
            "rolling_std_price_30s REAL",
            "rolling_volume_30s REAL",
            "trade_frequency_30s REAL",
            "price_return_30s REAL",
            "price_momentum_30s REAL",
            "price_range_30s REAL",
            "volatility_30s REAL",

            "rolling_mean_price_60s REAL",
            "rolling_std_price_60s REAL",
            "rolling_volume_60s REAL",
            "trade_frequency_60s REAL",
            "price_momentum_60s REAL",
            "price_range_60s REAL",

            "momentum_acceleration REAL",
            "volume_acceleration REAL",
            "trade_frequency_change REAL",
            "volume_imbalance_60s REAL",
        ]

        query = f"""
        CREATE TABLE IF NOT EXISTS crypto_features (
            {", ".join(columns)}
        )
        """

        self.conn.execute(query)
        self.conn.commit()

    def add_new_columns(self):

        existing = {
            row[1]
            for row in self.conn.execute(
                "PRAGMA table_info(crypto_features)"
            ).fetchall()
        }

        required = {
            "price_range_60s": "REAL",
            "momentum_acceleration": "REAL",
            "volume_acceleration": "REAL",
            "trade_frequency_change": "REAL",
            "volume_imbalance_60s": "REAL",

            "buy_volume_5s": "REAL",
            "sell_volume_5s": "REAL",
            "buy_trade_count_5s": "REAL",
            "sell_trade_count_5s": "REAL",
            "volume_imbalance_5s": "REAL",
            "trade_imbalance_5s": "REAL",
            "buy_sell_ratio_5s": "REAL",

            "buy_volume_10s": "REAL",
            "sell_volume_10s": "REAL",
            "buy_trade_count_10s": "REAL",
            "sell_trade_count_10s": "REAL",
            "volume_imbalance_10s": "REAL",
            "trade_imbalance_10s": "REAL",
            "buy_sell_ratio_10s": "REAL",

            "buy_volume_30s": "REAL",
            "sell_volume_30s": "REAL",
            "buy_trade_count_30s": "REAL",
            "sell_trade_count_30s": "REAL",
            "volume_imbalance_30s": "REAL",
            "trade_imbalance_30s": "REAL",
            "buy_sell_ratio_30s": "REAL",

            "buy_volume_60s": "REAL",
            "sell_volume_60s": "REAL",
            "buy_trade_count_60s": "REAL",
            "sell_trade_count_60s": "REAL",
            "volume_imbalance_60s": "REAL",
            "trade_imbalance_60s": "REAL",
            "buy_sell_ratio_60s": "REAL",
        }

        for column, dtype in required.items():

            if column not in existing:

                self.conn.execute(
                    f"""
                    ALTER TABLE crypto_features
                    ADD COLUMN {column} {dtype}
                    """
                )

        self.conn.commit()

    def log_tick(
        self,
        timestamp,
        price,
        volume,
        features
    ):

        columns = [
            "timestamp",
            "price",
            "volume"
        ] + FEATURE_COLUMNS

        values = [
            timestamp,
            price,
            volume
        ]

        for column in FEATURE_COLUMNS:

            values.append(
                features.get(
                    column,
                    None
                )
            )

        placeholders = ",".join(
            ["?"] * len(columns)
        )

        query = f"""
        INSERT OR REPLACE INTO crypto_features
        ({",".join(columns)})
        VALUES ({placeholders})
        """

        self.conn.execute(
            query,
            values
        )

        self.conn.commit()

    def close(self):

        if self.conn:
            self.conn.close()


# ============================================================
# GAP ANALYSIS
# ============================================================

def analyze_time_gaps(df):

    print()
    print("==========================================")
    print("TIME GAP ANALYSIS")
    print("==========================================")

    timestamps = (
        df["timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )

    gaps = timestamps.diff()

    valid_gaps = gaps.dropna()

    if len(valid_gaps) == 0:

        print("Not enough timestamps for gap analysis.")

        return

    print(
        f"Rows analyzed        : {len(timestamps):,}"
    )

    print(
        f"Largest gap          : "
        f"{valid_gaps.max():,.2f} seconds"
    )

    print(
        f"Gaps > 60 seconds    : "
        f"{(valid_gaps > 60).sum():,}"
    )

    print(
        f"Gaps > 5 minutes     : "
        f"{(valid_gaps > 300).sum():,}"
    )

    print(
        f"Gaps > 1 hour        : "
        f"{(valid_gaps > 3600).sum():,}"
    )


# ============================================================
# CREATE CONTINUOUS SEGMENTS
# ============================================================

def create_continuous_segments(df):

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    df["time_gap"] = (
        df["timestamp"].diff()
    )

    # A new segment starts whenever the gap
    # exceeds the maximum allowed continuous gap.
    df["segment_id"] = (
        df["time_gap"]
        .gt(MAX_CONTINUOUS_GAP_SECONDS)
        .cumsum()
    )

    return df


# ============================================================
# CREATE CLEAN DATASET
# ============================================================

def create_clean_dataset(db_path):

    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT
        timestamp,
        price,
        {",".join(FEATURE_COLUMNS)}
    FROM crypto_features
    ORDER BY timestamp
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    print()
    print(
        f"Rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Remove rows with missing features
    # --------------------------------------------------------

    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    print(
        f"Rows with complete features: "
        f"{len(df):,}"
    )

    if len(df) < 1000:

        raise ValueError(
            "Not enough complete feature rows."
        )

    # --------------------------------------------------------
    # Analyze gaps
    # --------------------------------------------------------

    analyze_time_gaps(df)

    # --------------------------------------------------------
    # Create continuous segments
    # --------------------------------------------------------

    df = create_continuous_segments(
        df
    )

    number_of_segments = (
        df["segment_id"]
        .nunique()
    )

    print()
    print(
        f"Continuous data segments: "
        f"{number_of_segments:,}"
    )

    # --------------------------------------------------------
    # Remove very small segments
    #
    # A segment needs at least two rows to produce
    # a future label.
    # --------------------------------------------------------

    segment_sizes = (
        df.groupby("segment_id")
        .size()
    )

    valid_segments = (
        segment_sizes[
            segment_sizes >= 2
        ]
        .index
    )

    df = df[
        df["segment_id"].isin(
            valid_segments
        )
    ].copy()

    print(
        f"Rows inside usable segments: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Create future dataset
    # --------------------------------------------------------

    future = df[
        [
            "segment_id",
            "timestamp",
            "price"
        ]
    ].copy()

    future = future.rename(
        columns={
            "timestamp": "future_timestamp",
            "price": "future_price"
        }
    )

    # --------------------------------------------------------
    # Current rows
    # --------------------------------------------------------

    current = df.copy()

    current["target_timestamp"] = (
        current["timestamp"]
        + TARGET_HORIZON_SECONDS
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Match only inside the SAME continuous segment.
    #
    # This prevents a label from jumping across
    # a collection outage.
    # --------------------------------------------------------

    labeled = pd.merge_asof(
        current.sort_values(
            [
                "segment_id",
                "target_timestamp"
            ]
        ),
        future.sort_values(
            [
                "segment_id",
                "future_timestamp"
            ]
        ),
        left_on="target_timestamp",
        right_on="future_timestamp",
        by="segment_id",
        direction="forward"
    )

    # --------------------------------------------------------
    # Calculate actual future gap
    # --------------------------------------------------------

    labeled["future_gap"] = (
        labeled["future_timestamp"]
        - labeled["timestamp"]
    )

    # --------------------------------------------------------
    # Keep only realistic ~30 second targets
    # --------------------------------------------------------

    labeled = labeled[
        (labeled["future_gap"] >= MIN_FUTURE_GAP_SECONDS)
        &
        (labeled["future_gap"] <= MAX_FUTURE_GAP_SECONDS)
    ].copy()

    labeled = labeled.dropna(
        subset=[
            "future_price"
        ]
    )

    # --------------------------------------------------------
    # Calculate future return
    # --------------------------------------------------------

    labeled["future_return"] = (
        (
            labeled["future_price"]
            - labeled["price"]
        )
        / labeled["price"]
    )

    # --------------------------------------------------------
    # Create target
    #
    # 0 = DOWN
    # 1 = FLAT
    # 2 = UP
    # --------------------------------------------------------

    labeled["target"] = np.select(
        [
            labeled["future_return"] < -0.0002,
            labeled["future_return"] > 0.0002
        ],
        [
            0,
            2
        ],
        default=1
    )

    print()
    print(
        "=========================================="
    )

    print(
        f"Rows after gap-aware 30-second labeling: "
        f"{len(labeled):,}"
    )

    print(
        f"Minimum future gap: "
        f"{labeled['future_gap'].min():.2f}s"
    )

    print(
        f"Maximum future gap: "
        f"{labeled['future_gap'].max():.2f}s"
    )

    print(
        "=========================================="
    )

    if len(labeled) < 1000:

        raise ValueError(
            "Not enough labeled rows after gap filtering."
        )

    return labeled


# ============================================================
# DISTRIBUTION
# ============================================================

def print_distribution(
    y,
    title
):

    counts = (
        y.value_counts()
        .sort_index()
    )

    print()
    print(title)

    print(
        f"DOWN : {counts.get(0, 0):,}"
    )

    print(
        f"FLAT : {counts.get(1, 0):,}"
    )

    print(
        f"UP   : {counts.get(2, 0):,}"
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            )
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    name,
    y_true,
    predictions
):

    metrics = calculate_metrics(
        y_true,
        predictions
    )

    print()
    print(
        f"========== {name} =========="
    )

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : {metrics['f1']:.4f}"
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1, 2]
    )

    print()
    print("Confusion Matrix")
    print(
        "              Predicted"
    )
    print(
        "              DOWN FLAT UP"
    )

    print(
        f"Actual DOWN    "
        f"{cm[0,0]:3d}  "
        f"{cm[0,1]:3d}  "
        f"{cm[0,2]:3d}"
    )

    print(
        f"Actual FLAT    "
        f"{cm[1,0]:3d}  "
        f"{cm[1,1]:3d}  "
        f"{cm[1,2]:3d}"
    )

    print(
        f"Actual UP      "
        f"{cm[2,0]:3d}  "
        f"{cm[2,1]:3d}  "
        f"{cm[2,2]:3d}"
    )

    return metrics


# ============================================================
# BASELINE 1
# MAJORITY CLASS
# ============================================================

def majority_baseline(
    train_df,
    test_df
):

    majority_class = (
        train_df["target"]
        .value_counts()
        .idxmax()
    )

    predictions = np.full(
        len(test_df),
        majority_class
    )

    print(
        f"\nMajority class from training: "
        f"{majority_class}"
    )

    return predictions


# ============================================================
# BASELINE 2
# MOMENTUM
# ============================================================

def momentum_baseline(
    test_df
):

    predictions = []

    for value in test_df[
        "price_momentum_30s"
    ]:

        if value > 0.02:

            predictions.append(2)

        elif value < -0.02:

            predictions.append(0)

        else:

            predictions.append(1)

    return np.array(
        predictions
    )


# ============================================================
# BASELINE 3
# RANDOM
# ============================================================

def random_baseline(
    train_df,
    test_df
):

    distribution = (
        train_df["target"]
        .value_counts(
            normalize=True
        )
        .reindex(
            [0, 1, 2],
            fill_value=0
        )
    )

    rng = np.random.default_rng(
        42
    )

    predictions = rng.choice(
        [0, 1, 2],
        size=len(test_df),
        p=distribution.values
    )

    return predictions


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def train_logistic(
    train_df,
    test_df
):

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "target"
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    return (
        model,
        scaler,
        predictions
    )


# ============================================================
# XGBOOST
# ============================================================

def train_xgboost(
    train_df,
    test_df
):

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "target"
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return (
        model,
        predictions
    )


# ============================================================
# ONE WALK
# ============================================================

def run_one_walk(
    walk_number,
    train_df,
    test_df
):

    print()
    print("==========================================")
    print(
        f"WALK {walk_number}"
    )
    print("==========================================")

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Testing rows : {len(test_df):,}"
    )

    print_distribution(
        train_df["target"],
        "Training distribution:"
    )

    print_distribution(
        test_df["target"],
        "Testing distribution:"
    )

    results = {}

    # --------------------------------------------------------
    # Majority
    # --------------------------------------------------------

    predictions = majority_baseline(
        train_df,
        test_df
    )

    results["majority"] = print_results(
        "MAJORITY BASELINE",
        test_df["target"],
        predictions
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    predictions = momentum_baseline(
        test_df
    )

    results["momentum"] = print_results(
        "MOMENTUM BASELINE",
        test_df["target"],
        predictions
    )

    # --------------------------------------------------------
    # Random
    # --------------------------------------------------------

    predictions = random_baseline(
        train_df,
        test_df
    )

    results["random"] = print_results(
        "RANDOM BASELINE",
        test_df["target"],
        predictions
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    print()
    print(
        "Training Logistic Regression..."
    )

    (
        logistic_model,
        scaler,
        predictions
    ) = train_logistic(
        train_df,
        test_df
    )

    results["logistic"] = print_results(
        "LOGISTIC REGRESSION",
        test_df["target"],
        predictions
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print()
    print(
        "Training XGBoost..."
    )

    (
        xgb_model,
        predictions
    ) = train_xgboost(
        train_df,
        test_df
    )

    results["xgboost"] = print_results(
        "XGBOOST",
        test_df["target"],
        predictions
    )

    return (
        results,
        logistic_model,
        scaler,
        xgb_model
    )


# ============================================================
# WALK-FORWARD EVALUATION
# ============================================================

def walk_forward_evaluation(
    labeled
):

    print()
    print("==========================================")
    print("GAP-AWARE WALK-FORWARD EXPERIMENT")
    print("==========================================")

    # Sort strictly by time.
    labeled = labeled.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    # Four chronological blocks.
    indices = np.array_split(
        np.arange(len(labeled)),
        4
    )

    blocks = [
        labeled.iloc[index].copy()
        for index in indices
    ]

    all_results = []

    final_logistic = None
    final_scaler = None
    final_xgb = None

    # --------------------------------------------------------
    # Walk 1
    # Train block 1 -> Test block 2
    #
    # Walk 2
    # Train blocks 1-2 -> Test block 3
    #
    # Walk 3
    # Train blocks 1-3 -> Test block 4
    # --------------------------------------------------------

    for i in range(1, 4):

        train_df = pd.concat(
            blocks[:i],
            ignore_index=True
        )

        test_df = blocks[i].copy()

        (
            results,
            logistic_model,
            scaler,
            xgb_model
        ) = run_one_walk(
            i,
            train_df,
            test_df
        )

        all_results.append(
            {
                "walk": i,
                "results": results
            }
        )

        if i == 3:

            final_logistic = (
                logistic_model
            )

            final_scaler = scaler

            final_xgb = xgb_model

    return (
        all_results,
        final_logistic,
        final_scaler,
        final_xgb
    )


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(
    logistic_model,
    scaler,
    xgb_model,
    results,
    labeled
):

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        {
            "model": logistic_model,
            "scaler": scaler,
            "features": FEATURE_COLUMNS
        },
        "models/logistic_btc_direction.joblib"
    )

    joblib.dump(
        {
            "model": xgb_model,
            "features": FEATURE_COLUMNS
        },
        "models/xgb_btc_direction.joblib"
    )

    metadata = {

        "experiment":
            "gap_aware_baseline_vs_ml_walk_forward",

        "target":
            "30_second_future_direction",

        "label_definition": {

            "DOWN":
                "< -0.02%",

            "FLAT":
                "-0.02% to +0.02%",

            "UP":
                "> +0.02%"
        },

        "future_gap_rule": {

            "target_horizon_seconds":
                TARGET_HORIZON_SECONDS,

            "minimum_seconds":
                MIN_FUTURE_GAP_SECONDS,

            "maximum_seconds":
                MAX_FUTURE_GAP_SECONDS
        },

        "continuity_rule": {

            "maximum_continuous_gap_seconds":
                MAX_CONTINUOUS_GAP_SECONDS
        },

        "total_labeled_rows":
            int(len(labeled)),

        "features":
            FEATURE_COLUMNS,

        "results":
            results
    }

    with open(
        "models/model_metadata.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print()
    print("==========================================")
    print("MODELS SAVED")
    print("==========================================")

    print(
        "models/logistic_btc_direction.joblib"
    )

    print(
        "models/xgb_btc_direction.joblib"
    )

    print(
        "models/model_metadata.json"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    all_results
):

    print()
    print("==========================================")
    print("FINAL GAP-AWARE SUMMARY")
    print("==========================================")

    for item in all_results:

        walk = item["walk"]

        results = item["results"]

        print()
        print(
            f"WALK {walk}"
        )

        print(
            f"Majority : "
            f"{results['majority']['f1']:.4f}"
        )

        print(
            f"Momentum : "
            f"{results['momentum']['f1']:.4f}"
        )

        print(
            f"Random   : "
            f"{results['random']['f1']:.4f}"
        )

        print(
            f"Logistic : "
            f"{results['logistic']['f1']:.4f}"
        )

        print(
            f"XGBoost  : "
            f"{results['xgboost']['f1']:.4f}"
        )


# ============================================================
# MAIN TRAINING
# ============================================================

def train_models(
    db_path="historical_features.db"
):

    print()
    print("==========================================")
    print("BTC DIRECTION GAP-AWARE EXPERIMENT")
    print("==========================================")

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The database will NOT be modified."
    )

    print(
        "Large time gaps will be excluded from labeling."
    )

    labeled = create_clean_dataset(
        db_path
    )

    print_distribution(
        labeled["target"],
        "Overall class distribution:"
    )

    (
        all_results,
        final_logistic,
        final_scaler,
        final_xgb
    ) = walk_forward_evaluation(
        labeled
    )

    save_models(
        final_logistic,
        final_scaler,
        final_xgb,
        all_results,
        labeled
    )

    print_final_summary(
        all_results
    )

    print()
    print("==========================================")
    print("EXPERIMENT COMPLETE")
    print("==========================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_models(
        DB_PATH
    )