import sqlite3
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score


DB_PATH = "historical_features.db"

START_TS = 1790605091.587
END_TS = 1791210807.566

TARGET_HORIZON_SECONDS = 30
MIN_FUTURE_GAP_SECONDS = 20
MAX_FUTURE_GAP_SECONDS = 40
MAX_CONTINUOUS_GAP_SECONDS = 60


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


# Feature sets to compare.
FEATURE_SETS = {

    "OLD_30": OLD_FEATURES,

    "MICRO_27": MICRO_FEATURES,

    "ALL_57": OLD_FEATURES + MICRO_FEATURES,

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


def load_data():

    print("==========================================")
    print("ABLATION EXPERIMENT")
    print("READ-ONLY DATABASE")
    print("==========================================")

    columns = list(dict.fromkeys(
        ["timestamp", "price"] + OLD_FEATURES + MICRO_FEATURES
    ))

    query = f"""
        SELECT {", ".join(columns)}
        FROM crypto_features
        WHERE timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """

    # READ-ONLY DATABASE CONNECTION
    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    df = pd.read_sql_query(
        query,
        conn,
        params=(START_TS, END_TS)
    )

    conn.close()

    print(f"RAW ROWS: {len(df):,}")

    return df


def prepare_data(df):

    all_features = OLD_FEATURES + MICRO_FEATURES

    # Require all features to be available.
    complete = df.dropna(
        subset=all_features + ["price"]
    ).copy()

    print(
        f"COMPLETE ROWS: {len(complete):,}"
    )

    complete = complete.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Identify continuous segments.
    gaps = complete["timestamp"].diff()

    complete["segment"] = (
        gaps > MAX_CONTINUOUS_GAP_SECONDS
    ).cumsum()

    segment_count = complete["segment"].nunique()

    print(
        f"CONTINUOUS SEGMENTS: {segment_count}"
    )

    # --------------------------------------------------
    # GAP-AWARE LABELING
    # --------------------------------------------------
    #
    # For every row:
    #   target timestamp = current timestamp + 30 seconds
    #
    # We search ONLY inside the same continuous segment.
    #
    # Future observation must be:
    #   >= 20 seconds
    #   <= 40 seconds
    #
    # This matches the labeling logic used in the
    # successful feature-importance experiment.
    # --------------------------------------------------

    labeled_parts = []

    for segment_id, segment_df in complete.groupby(
        "segment",
        sort=True
    ):

        segment_df = segment_df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        timestamps = segment_df[
            "timestamp"
        ].to_numpy()

        prices = segment_df[
            "price"
        ].to_numpy()

        rows = []

        for i in range(len(segment_df)):

            current_timestamp = timestamps[i]
            current_price = prices[i]

            target_timestamp = (
                current_timestamp
                + TARGET_HORIZON_SECONDS
            )

            # First future observation at or after
            # current timestamp + 30 seconds.
            future_index = np.searchsorted(
                timestamps,
                target_timestamp,
                side="left"
            )

            if future_index >= len(segment_df):
                continue

            future_timestamp = timestamps[
                future_index
            ]

            future_price = prices[
                future_index
            ]

            future_gap = (
                future_timestamp
                - current_timestamp
            )

            # Enforce valid future gap.
            if future_gap < MIN_FUTURE_GAP_SECONDS:
                continue

            if future_gap > MAX_FUTURE_GAP_SECONDS:
                continue

            row = segment_df.iloc[i].copy()

            future_return = (
                future_price - current_price
            ) / current_price

            # Same target definition as existing trainer.
            if future_return < -0.0002:
                target = 0       # DOWN

            elif future_return > 0.0002:
                target = 2       # UP

            else:
                target = 1       # FLAT

            row["target"] = target

            rows.append(row)

        if rows:
            labeled_parts.append(
                pd.DataFrame(rows)
            )

    if not labeled_parts:
        raise RuntimeError(
            "No labeled rows were produced. "
            "Check timestamp range and feature data."
        )

    labeled = pd.concat(
        labeled_parts,
        ignore_index=True
    )

    labeled = labeled.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"LABELED ROWS: {len(labeled):,}"
    )

    print()
    print("TARGET DISTRIBUTION")

    counts = labeled[
        "target"
    ].value_counts().sort_index()

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

    return labeled


def train_model(
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


def evaluate_feature_set(
    df,
    feature_name,
    features
):

    print()
    print("=" * 55)
    print(
        f"FEATURE SET: {feature_name}"
    )
    print(
        f"FEATURE COUNT: {len(features)}"
    )
    print("=" * 55)

    data = df.dropna(
        subset=features + ["target"]
    ).copy()

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Same 4-block chronological split used
    # in the previous experiments.
    blocks = np.array_split(
        data,
        4
    )

    walk_scores = []

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

        model = train_model(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        walk_scores.append(
            macro_f1
        )

        print(
            f"Walk {walk + 1}: "
            f"Train={len(train):,} "
            f"Test={len(test_part):,} "
            f"Accuracy={accuracy:.4f} "
            f"Macro F1={macro_f1:.4f}"
        )

    average_f1 = np.mean(
        walk_scores
    )

    print(
        f"AVERAGE MACRO F1: "
        f"{average_f1:.4f}"
    )

    return walk_scores, average_f1


def main():

    df = load_data()

    df = prepare_data(df)

    results = []

    for name, features in FEATURE_SETS.items():

        scores, average_f1 = evaluate_feature_set(
            df,
            name,
            features
        )

        results.append({
            "feature_set": name,
            "features": len(features),
            "walk_1": scores[0],
            "walk_2": scores[1],
            "walk_3": scores[2],
            "average_macro_f1": average_f1
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "average_macro_f1",
        ascending=False
    )

    print()
    print()
    print("=" * 65)
    print("FINAL ABLATION RESULTS")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "walk_1": "{:.4f}".format,
                "walk_2": "{:.4f}".format,
                "walk_3": "{:.4f}".format,
                "average_macro_f1": "{:.4f}".format
            }
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