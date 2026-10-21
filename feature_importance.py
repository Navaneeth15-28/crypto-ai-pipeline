import sqlite3
import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

DB_PATH = "historical_features.db"

START = 1790605091.587
END = 1791210807.566

MAX_CONTINUOUS_GAP_SECONDS = 60

TARGET_HORIZON_SECONDS = 30
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
    "volume_imbalance_60s"
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
    "buy_sell_ratio_60s"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    columns = list(
        dict.fromkeys(
            ["timestamp", "price"]
            + OLD_FEATURES
            + MICRO_FEATURES
        )
    )

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

    print(
        "COMPLETE ROWS:",
        f"{len(df):,}"
    )

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
        ]
        + OLD_FEATURES
        + MICRO_FEATURES
    ].copy()

    current["target_timestamp"] = (
        current["timestamp"]
        + TARGET_HORIZON_SECONDS
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
            "future_timestamp",
            "future_price"
        ]
    ).copy()

    # --------------------------------------------------------
    # Future return
    # --------------------------------------------------------

    labeled["future_return"] = (
        labeled["future_price"]
        - labeled["price"]
    ) / labeled["price"]

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    labeled["target"] = 1

    labeled.loc[
        labeled["future_return"] < -0.0002,
        "target"
    ] = 0

    labeled.loc[
        labeled["future_return"] > 0.0002,
        "target"
    ] = 2

    labeled = labeled.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    print(
        "LABELED ROWS:",
        f"{len(labeled):,}"
    )

    print()
    print("TARGET DISTRIBUTION")

    names = {
        0: "DOWN",
        1: "FLAT",
        2: "UP"
    }

    for target in [0, 1, 2]:

        count = (
            labeled["target"] == target
        ).sum()

        print(
            f"{names[target]:5s}: "
            f"{count:,} "
            f"({count / len(labeled):.2%})"
        )

    return labeled


# ============================================================
# XGBOOST
# ============================================================

def train_xgboost(
    train_df,
    features
):

    X_train = train_df[features]

    y_train = train_df["target"]

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


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def run_importance(
    labeled,
    features,
    feature_name
):

    print()
    print("==========================================")
    print(
        f"FEATURE IMPORTANCE: {feature_name}"
    )
    print(
        f"FEATURES: {len(features)}"
    )
    print("==========================================")

    indices = np.array_split(
        np.arange(len(labeled)),
        4
    )

    blocks = [
        labeled.iloc[index].copy()
        for index in indices
    ]

    walk_importances = []

    for i in range(1, 4):

        train_df = pd.concat(
            blocks[:i],
            ignore_index=True
        )

        model = train_xgboost(
            train_df,
            features
        )

        importance = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "importance",
            ascending=False
        )

        walk_importances.append(
            importance
        )

        print()
        print(
            f"TOP 10 FEATURES - WALK {i}"
        )
        print("------------------------------------------")

        print(
            importance.head(10).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Average importance across walks
    # --------------------------------------------------------

    combined = pd.concat(
        [
            result.assign(
                walk=walk
            )
            for walk, result
            in enumerate(
                walk_importances,
                start=1
            )
        ],
        ignore_index=True
    )

    average = (
        combined
        .groupby("feature")["importance"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print()
    print("==========================================")
    print(
        f"AVERAGE IMPORTANCE - {feature_name}"
    )
    print("==========================================")

    print(
        average.head(20).to_string()
    )

    return average


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("FEATURE IMPORTANCE EXPERIMENT")
    print("READ-ONLY DATABASE")
    print("==========================================")

    df = load_data()

    print(
        "RAW ROWS:",
        f"{len(df):,}"
    )

    labeled = prepare_data(
        df
    )

    run_importance(
        labeled,
        OLD_FEATURES,
        "OLD FEATURES"
    )

    run_importance(
        labeled,
        MICRO_FEATURES,
        "MICROSTRUCTURE FEATURES"
    )

    all_features = list(
        dict.fromkeys(
            OLD_FEATURES + MICRO_FEATURES
        )
    )

    all_importance = run_importance(
        labeled,
        all_features,
        "ALL FEATURES"
    )

    print()
    print("==========================================")
    print("FINAL TOP 20 FEATURES")
    print("==========================================")

    print(
        all_importance.head(20).to_string()
    )


if __name__ == "__main__":
    main()