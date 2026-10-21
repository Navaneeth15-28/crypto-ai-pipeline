import sqlite3
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

DB_PATH = "historical_features.db"

START = 1790605091.587
END = 1791210807.566

MAX_CONTINUOUS_GAP_SECONDS = 60
MIN_FUTURE_GAP_SECONDS = 20
MAX_FUTURE_GAP_SECONDS = 40


# ============================================================
# OLD FEATURES
# ============================================================

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
    "volume_imbalance_60s",
]


# ============================================================
# MICROSTRUCTURE FEATURES
# ============================================================

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
    "buy_sell_ratio_60s",
]


# ============================================================
# LOAD FRESH MICROSTRUCTURE DATA ONLY
# ============================================================

def load_data():

    columns = [
        "timestamp",
        "price"
    ] + OLD_FEATURES + MICRO_FEATURES

    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    df = pd.read_sql_query(
        f"""
        SELECT {", ".join(columns)}
        FROM crypto_features
        WHERE timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
        """,
        conn,
        params=(START, END)
    )

    conn.close()

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    required = OLD_FEATURES + MICRO_FEATURES

    df = df.dropna(
        subset=required
    ).copy()

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    print()
    print("COMPLETE ROWS:", f"{len(df):,}")

    # --------------------------------------------------------
    # Continuous segments
    # --------------------------------------------------------

    gaps = df["timestamp"].diff()

    df["segment_id"] = (
        (gaps > MAX_CONTINUOUS_GAP_SECONDS)
        .cumsum()
    )

    print(
        "CONTINUOUS SEGMENTS:",
        df["segment_id"].nunique()
    )

    # --------------------------------------------------------
    # Current rows
    # --------------------------------------------------------

    current = df[
        [
            "timestamp",
            "price",
            "segment_id"
        ] + OLD_FEATURES + MICRO_FEATURES
    ].copy()

    current["target_timestamp"] = (
        current["timestamp"] + 30
    )

    # --------------------------------------------------------
    # Future rows
    # --------------------------------------------------------

    future = df[
        [
            "timestamp",
            "price",
            "segment_id"
        ]
    ].copy()

    future = future.rename(
        columns={
            "timestamp": "future_timestamp",
            "price": "future_price"
        }
    )

    # --------------------------------------------------------
    # Same-segment future matching
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
    # Future gap
    # --------------------------------------------------------

    labeled["future_gap"] = (
        labeled["future_timestamp"]
        - labeled["timestamp"]
    )

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
    # Future return
    # --------------------------------------------------------

    labeled["future_return"] = (
        (
            labeled["future_price"]
            - labeled["price"]
        )
        / labeled["price"]
    )

    # --------------------------------------------------------
    # Target
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

    print(
        "LABELED ROWS:",
        f"{len(labeled):,}"
    )

    print()
    print("TARGET DISTRIBUTION:")

    counts = labeled["target"].value_counts().sort_index()

    names = {
        0: "DOWN",
        1: "FLAT",
        2: "UP"
    }

    for target, count in counts.items():

        print(
            f"{names[target]:5s}: {count:,} "
            f"({count / len(labeled):.2%})"
        )

    return labeled


# ============================================================
# MODELS
# ============================================================

def train_logistic(train_df, test_df, features):

    X_train = train_df[features]
    y_train = train_df["target"]

    X_test = test_df[features]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model.predict(
        X_test
    )


def train_xgboost(train_df, test_df, features):

    X_train = train_df[features]
    y_train = train_df["target"]

    X_test = test_df[features]

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

    return model.predict(
        X_test
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    train_df,
    test_df,
    features,
    model_name
):

    predictions = train_logistic(
        train_df,
        test_df,
        features
    )

    f1 = f1_score(
        test_df["target"],
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(
        test_df["target"],
        predictions
    )

    return accuracy, f1


def evaluate_xgb(
    train_df,
    test_df,
    features
):

    predictions = train_xgboost(
        train_df,
        test_df,
        features
    )

    f1 = f1_score(
        test_df["target"],
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(
        test_df["target"],
        predictions
    )

    return accuracy, f1


# ============================================================
# WALK-FORWARD
# ============================================================

def run_experiment(labeled, features, feature_name):

    print()
    print("==========================================")
    print(feature_name)
    print("FEATURES:", len(features))
    print("==========================================")

    indices = np.array_split(
        np.arange(len(labeled)),
        4
    )

    blocks = [
        labeled.iloc[index].copy()
        for index in indices
    ]

    logistic_scores = []
    xgb_scores = []

    for i in range(1, 4):

        train_df = pd.concat(
            blocks[:i],
            ignore_index=True
        )

        test_df = blocks[i].copy()

        print()
        print(
            f"WALK {i}: "
            f"TRAIN {len(train_df):,} | "
            f"TEST {len(test_df):,}"
        )

        acc, f1 = evaluate_model(
            train_df,
            test_df,
            features,
            "logistic"
        )

        print(
            f"Logistic | Accuracy: {acc:.4f} | "
            f"Macro F1: {f1:.4f}"
        )

        logistic_scores.append(f1)

        acc, f1 = evaluate_xgb(
            train_df,
            test_df,
            features
        )

        print(
            f"XGBoost  | Accuracy: {acc:.4f} | "
            f"Macro F1: {f1:.4f}"
        )

        xgb_scores.append(f1)

    print()
    print("AVERAGE RESULTS")
    print("------------------------------------------")

    print(
        f"Logistic Average Macro F1: "
        f"{np.mean(logistic_scores):.4f}"
    )

    print(
        f"XGBoost Average Macro F1: "
        f"{np.mean(xgb_scores):.4f}"
    )

    return {
        "logistic": float(np.mean(logistic_scores)),
        "xgboost": float(np.mean(xgb_scores))
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("MICROSTRUCTURE TRAINING EXPERIMENT")
    print("READ-ONLY DATABASE")
    print("==========================================")

    df = load_data()

    print(
        "RAW MICROSTRUCTURE ROWS:",
        f"{len(df):,}"
    )

    labeled = prepare_data(
        df
    )

    results = {}

    results["OLD"] = run_experiment(
        labeled,
        OLD_FEATURES,
        "OLD FEATURES ONLY"
    )

    results["MICROSTRUCTURE"] = run_experiment(
        labeled,
        MICRO_FEATURES,
        "MICROSTRUCTURE FEATURES ONLY"
    )

    results["ALL"] = run_experiment(
        labeled,
        OLD_FEATURES + [x for x in MICRO_FEATURES if x not in OLD_FEATURES],
        "ALL FEATURES"
    )

    print()
    print("==========================================")
    print("FINAL COMPARISON")
    print("==========================================")

    for name, result in results.items():

        print()
        print(name)

        print(
            f"  Logistic Macro F1: "
            f"{result['logistic']:.4f}"
        )

        print(
            f"  XGBoost Macro F1 : "
            f"{result['xgboost']:.4f}"
        )


if __name__ == "__main__":
    main()


