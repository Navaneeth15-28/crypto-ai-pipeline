import sqlite3
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier


# =========================================================
# CONFIGURATION
# =========================================================

DB_PATH = "historical_features.db"

START = 1790605091.587
END = 1791210807.566

TARGET_HORIZON_SECONDS = 30
MIN_FUTURE_GAP_SECONDS = 20
MAX_FUTURE_GAP_SECONDS = 40

DOWN_THRESHOLD = -0.0002
UP_THRESHOLD = 0.0002


# =========================================================
# OLD FEATURES - 30
# =========================================================

OLD_FEATURES = [
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
    "volume_imbalance_60s"
]


# =========================================================
# MICROSTRUCTURE FEATURES - 27
# =========================================================

MICRO_FEATURES = [
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
    "trade_imbalance_60s",
    "buy_sell_ratio_60s"
]


# =========================================================
# FEATURE SETS
# =========================================================

FEATURE_SETS = {

    "OLD_30": OLD_FEATURES,

    "MICRO_27": MICRO_FEATURES,

    "ALL_57": list(dict.fromkeys(
        OLD_FEATURES + MICRO_FEATURES
    )),

    "TOP_20": [
        "sell_volume_60s",
        "price_range_60s",
        "rolling_mean_price_60s",
        "sell_trade_count_30s",
        "rolling_mean_price_30s",
        "rolling_volume_30s",
        "rolling_volume_60s",
        "buy_volume_30s",
        "buy_volume_60s",
        "rolling_mean_price_10s",
        "buy_sell_ratio_30s",
        "sell_volume_30s",
        "sell_trade_count_60s",
        "buy_trade_count_60s",
        "rolling_mean_price_5s",
        "momentum_acceleration",
        "buy_sell_ratio_60s",
        "volume_imbalance_30s",
        "price_range_30s",
        "trade_frequency_30s"
    ],

    "TOP_10": [
        "sell_volume_60s",
        "price_range_60s",
        "rolling_mean_price_60s",
        "sell_trade_count_30s",
        "rolling_mean_price_30s",
        "rolling_volume_30s",
        "rolling_volume_60s",
        "buy_volume_30s",
        "buy_volume_60s",
        "rolling_mean_price_10s"
    ],

    "TOP_MICRO_10": [
        "sell_volume_60s",
        "buy_volume_60s",
        "buy_volume_30s",
        "buy_sell_ratio_30s",
        "sell_trade_count_30s",
        "buy_sell_ratio_60s",
        "sell_volume_30s",
        "buy_trade_count_60s",
        "sell_trade_count_60s",
        "volume_imbalance_30s"
    ]
}


# =========================================================
# LOAD DATABASE
# =========================================================

def load_data():

    print("==========================================")
    print("ABLATION EXPERIMENT")
    print("READ-ONLY DATABASE")
    print("==========================================")

    all_features = list(
        dict.fromkeys(
            OLD_FEATURES + MICRO_FEATURES
        )
    )

    columns = [
        "timestamp",
        "price"
    ] + all_features

    columns = list(
        dict.fromkeys(columns)
    )

    connection = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    query = f"""
        SELECT {", ".join(columns)}
        FROM crypto_features
        WHERE timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """

    df = pd.read_sql_query(
        query,
        connection,
        params=(START, END)
    )

    connection.close()

    print(f"RAW ROWS: {len(df):,}")

    complete_df = df.dropna(
        subset=all_features
    ).copy()

    complete_df = complete_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"COMPLETE ROWS: {len(complete_df):,}"
    )

    return complete_df


# =========================================================
# FIND CONTINUOUS SEGMENTS
# =========================================================

def find_segments(df):

    timestamps = df["timestamp"].to_numpy()

    if len(timestamps) == 0:
        return []

    gaps = np.diff(timestamps)

    segment_ids = np.zeros(
        len(df),
        dtype=int
    )

    segment_ids[1:] = np.cumsum(
        gaps > 60
    )

    df = df.copy()

    df["segment"] = segment_ids

    segments = []

    for segment_id, segment in df.groupby(
        "segment",
        sort=True
    ):

        segment = segment.reset_index(
            drop=True
        )

        if len(segment) > 1:
            segments.append(segment)

    return segments


# =========================================================
# CREATE TARGET
# =========================================================

def create_labels(df):

    segments = find_segments(df)

    print(
        f"CONTINUOUS SEGMENTS: {len(segments)}"
    )

    labeled_parts = []

    for segment in segments:

        timestamps = segment[
            "timestamp"
        ].to_numpy()

        prices = segment[
            "price"
        ].to_numpy()

        targets = np.full(
            len(segment),
            np.nan
        )

        for i in range(len(segment)):

            target_time = (
                timestamps[i]
                + TARGET_HORIZON_SECONDS
            )

            future_index = np.searchsorted(
                timestamps,
                target_time,
                side="left"
            )

            if future_index >= len(segment):
                continue

            future_gap = (
                timestamps[future_index]
                - timestamps[i]
            )

            if (
                future_gap < MIN_FUTURE_GAP_SECONDS
                or
                future_gap > MAX_FUTURE_GAP_SECONDS
            ):
                continue

            current_price = prices[i]
            future_price = prices[future_index]

            future_return = (
                future_price - current_price
            ) / current_price

            if future_return < DOWN_THRESHOLD:
                targets[i] = 0

            elif future_return > UP_THRESHOLD:
                targets[i] = 2

            else:
                targets[i] = 1

        segment = segment.copy()

        segment["target"] = targets

        labeled_parts.append(
            segment.dropna(
                subset=["target"]
            )
        )

    if not labeled_parts:
        return pd.DataFrame()

    labeled = pd.concat(
        labeled_parts,
        ignore_index=True
    )

    labeled["target"] = (
        labeled["target"]
        .astype(int)
    )

    return labeled


# =========================================================
# TRAIN XGBOOST
# =========================================================

def train_xgb(
    X_train,
    y_train
):

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

    return model


# =========================================================
# TRAIN LOGISTIC REGRESSION
# =========================================================

def train_logistic(
    X_train,
    y_train
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
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

    return model, scaler


# =========================================================
# EVALUATE ONE FEATURE SET
# =========================================================

def evaluate_feature_set(
    df,
    feature_name,
    features
):

    print()
    print("=" * 60)
    print(
        f"FEATURE SET: {feature_name}"
    )
    print(
        f"FEATURE COUNT: {len(features)}"
    )
    print("=" * 60)

    data = df.dropna(
        subset=features + ["target"]
    ).copy()

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    blocks = np.array_split(
        data,
        4
    )

    xgb_scores = []
    logistic_scores = []

    for walk in range(3):

        train_parts = blocks[
            :walk + 1
        ]

        test_part = blocks[
            walk + 1
        ]

        train = pd.concat(
            train_parts,
            ignore_index=True
        )

        X_train = train[
            features
        ]

        y_train = train[
            "target"
        ]

        X_test = test_part[
            features
        ]

        y_test = test_part[
            "target"
        ]

        # -------------------------------------------------
        # XGBOOST
        # -------------------------------------------------

        xgb_model = train_xgb(
            X_train,
            y_train
        )

        xgb_predictions = (
            xgb_model.predict(
                X_test
            )
        )

        xgb_accuracy = accuracy_score(
            y_test,
            xgb_predictions
        )

        xgb_f1 = f1_score(
            y_test,
            xgb_predictions,
            average="macro"
        )

        xgb_scores.append(
            xgb_f1
        )

        # -------------------------------------------------
        # LOGISTIC REGRESSION
        # -------------------------------------------------

        logistic_model, scaler = (
            train_logistic(
                X_train,
                y_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

        logistic_predictions = (
            logistic_model.predict(
                X_test_scaled
            )
        )

        logistic_accuracy = (
            accuracy_score(
                y_test,
                logistic_predictions
            )
        )

        logistic_f1 = f1_score(
            y_test,
            logistic_predictions,
            average="macro"
        )

        logistic_scores.append(
            logistic_f1
        )

        print(
            f"Walk {walk + 1}: "
            f"XGB Accuracy={xgb_accuracy:.4f} "
            f"XGB F1={xgb_f1:.4f} | "
            f"Logistic Accuracy={logistic_accuracy:.4f} "
            f"Logistic F1={logistic_f1:.4f}"
        )

    xgb_average = np.mean(
        xgb_scores
    )

    logistic_average = np.mean(
        logistic_scores
    )

    print(
        f"XGB AVERAGE MACRO F1: "
        f"{xgb_average:.4f}"
    )

    print(
        f"LOGISTIC AVERAGE MACRO F1: "
        f"{logistic_average:.4f}"
    )

    return {
        "feature_set": feature_name,
        "features": len(features),

        "xgb_walk_1": xgb_scores[0],
        "xgb_walk_2": xgb_scores[1],
        "xgb_walk_3": xgb_scores[2],
        "xgb_average": xgb_average,

        "logistic_walk_1": logistic_scores[0],
        "logistic_walk_2": logistic_scores[1],
        "logistic_walk_3": logistic_scores[2],
        "logistic_average": logistic_average
    }


# =========================================================
# MAIN
# =========================================================

def main():

    df = load_data()

    labeled = create_labels(
        df
    )

    print(
        f"LABELED ROWS: {len(labeled):,}"
    )

    print()
    print("TARGET DISTRIBUTION")

    counts = (
        labeled["target"]
        .value_counts()
        .sort_index()
    )

    total = len(labeled)

    print(
        f"DOWN : {counts.get(0, 0):,} "
        f"({counts.get(0, 0) / total * 100:.2f}%)"
    )

    print(
        f"FLAT : {counts.get(1, 0):,} "
        f"({counts.get(1, 0) / total * 100:.2f}%)"
    )

    print(
        f"UP   : {counts.get(2, 0):,} "
        f"({counts.get(2, 0) / total * 100:.2f}%)"
    )

    results = []

    for feature_name, features in FEATURE_SETS.items():

        result = evaluate_feature_set(
            labeled,
            feature_name,
            features
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "xgb_average",
        ascending=False
    )

    print()
    print("=" * 90)
    print("FINAL ABLATION RESULTS")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()
    print(
        "DATABASE WAS USED IN READ-ONLY MODE."
    )

    print(
        "NO DATABASE CHANGES WERE MADE."
    )


if __name__ == "__main__":
    main()