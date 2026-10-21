import sqlite3
import bisect

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


DB_PATH = "historical_features.db"

HORIZON = 30
THRESHOLD = 0.0002

RECENT_TRAIN_ROWS = 2500


# ==========================================================
# FEATURE SET A
# ALL CURRENT FEATURES
# ==========================================================

ALL_FEATURES = [
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


# ==========================================================
# FEATURE SET B
# MOVEMENT / VOLATILITY / ORDER FLOW
# ==========================================================

PREDICTIVE_FEATURES = [

    # Short-term price movement
    "price_return_5s",
    "price_return_10s",
    "price_return_30s",

    # Momentum
    "price_momentum_30s",
    "price_momentum_60s",
    "momentum_acceleration",

    # Volatility
    "rolling_std_price_5s",
    "rolling_std_price_10s",
    "volatility_10s",
    "volatility_30s",

    # Price range
    "price_range_10s",
    "price_range_30s",
    "price_range_60s",

    # Volume behavior
    "rolling_volume_5s",
    "rolling_volume_10s",
    "rolling_volume_30s",
    "rolling_volume_60s",
    "volume_acceleration",

    # Trade activity
    "trade_frequency_5s",
    "trade_frequency_10s",
    "trade_frequency_30s",
    "trade_frequency_60s",
    "trade_frequency_change",

    # Order flow
    "volume_imbalance_60s",
]


def load_data():

    print()
    print("==========================================")
    print("LOADING DATA")
    print("==========================================")

    conn = sqlite3.connect(DB_PATH)

    columns = ", ".join(
        ["timestamp", "price"] + ALL_FEATURES
    )

    query = f"""
        SELECT {columns}
        FROM crypto_features
        WHERE price_range_60s IS NOT NULL
        ORDER BY timestamp
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    print(
        f"Feature rows loaded: {len(df)}"
    )

    return df


def create_labels(df):

    print()
    print("Creating labels...")

    timestamps = df["timestamp"].to_numpy()
    prices = df["price"].to_numpy()

    labels = []
    valid_indices = []

    for i in range(len(df)):

        future_index = bisect.bisect_left(
            timestamps,
            timestamps[i] + HORIZON
        )

        if future_index >= len(df):
            continue

        current_price = prices[i]
        future_price = prices[future_index]

        future_return = (
            future_price - current_price
        ) / current_price

        if future_return > THRESHOLD:

            label = 2

        elif future_return < -THRESHOLD:

            label = 0

        else:

            label = 1

        labels.append(label)
        valid_indices.append(i)

    result = df.iloc[
        valid_indices
    ].copy()

    result["target"] = labels

    print(
        f"Labeled rows: {len(result)}"
    )

    return result


def find_split(df):

    n = len(df)

    candidates = []

    for percent in np.arange(
        0.60,
        0.81,
        0.01
    ):

        split = int(
            n * percent
        )

        train = df.iloc[
            :split
        ]

        test = df.iloc[
            split:
        ]

        if (
            len(set(train["target"])) == 3
            and
            len(set(test["target"])) == 3
        ):

            candidates.append(
                (
                    abs(percent - 0.70),
                    percent,
                    split
                )
            )

    candidates.sort()

    return (
        candidates[0][2],
        candidates[0][1]
    )


def distribution(name, y):

    counts = pd.Series(
        y
    ).value_counts().sort_index()

    total = len(y)

    print()
    print(name)
    print("------------------------------------------")

    for label, name2 in [
        (0, "DOWN"),
        (1, "FLAT"),
        (2, "UP")
    ]:

        count = int(
            counts.get(
                label,
                0
            )
        )

        print(
            f"{name2:5s}: "
            f"{count:4d} "
            f"({count / total * 100:5.2f}%)"
        )


def build_logistic():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )


def build_xgb():

    return XGBClassifier(
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


def evaluate(
    name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print()
    print("==========================================")
    print(name)
    print("==========================================")

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "DOWN",
                "FLAT",
                "UP"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print("------------------------------------------")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    counts = pd.Series(
        predictions
    ).value_counts().sort_index()

    print()
    print("Predictions:")

    print(
        f"DOWN: {int(counts.get(0, 0))}"
    )

    print(
        f"FLAT: {int(counts.get(1, 0))}"
    )

    print(
        f"UP:   {int(counts.get(2, 0))}"
    )

    return {
        "accuracy": accuracy,
        "balanced": balanced,
        "macro_f1": macro_f1
    }


def run_experiment(
    feature_name,
    features,
    train,
    test
):

    print()
    print()
    print("##########################################")
    print(
        f"FEATURE SET: {feature_name}"
    )
    print("##########################################")

    print(
        f"Number of features: {len(features)}"
    )

    X_train = train[
        features
    ]

    y_train = train[
        "target"
    ]

    X_test = test[
        features
    ]

    y_test = test[
        "target"
    ]

    distribution(
        "Training distribution",
        y_train
    )

    # --------------------------------------
    # Logistic
    # --------------------------------------

    logistic = build_logistic()

    logistic.fit(
        X_train,
        y_train
    )

    logistic_result = evaluate(
        f"{feature_name} - LOGISTIC",
        logistic,
        X_test,
        y_test
    )

    # --------------------------------------
    # XGBoost
    # --------------------------------------

    xgb = build_xgb()

    xgb.fit(
        X_train,
        y_train
    )

    xgb_result = evaluate(
        f"{feature_name} - XGBOOST",
        xgb,
        X_test,
        y_test
    )

    return {
        "logistic": logistic_result,
        "xgb": xgb_result
    }


def main():

    print()
    print("==========================================")
    print("BTC FEATURE QUALITY EXPERIMENT")
    print("==========================================")

    df = load_data()

    df = create_labels(
        df
    )

    split, percent = find_split(
        df
    )

    train_full = df.iloc[
        :split
    ].copy()

    test = df.iloc[
        split:
    ].copy()

    recent_size = min(
        RECENT_TRAIN_ROWS,
        len(train_full)
    )

    train_recent = train_full.iloc[
        -recent_size:
    ].copy()

    print()
    print("==========================================")
    print("CHRONOLOGICAL SPLIT")
    print("==========================================")

    print(
        f"Split: {percent * 100:.0f}% / "
        f"{(1 - percent) * 100:.0f}%"
    )

    print(
        f"Full train: {len(train_full)}"
    )

    print(
        f"Recent train: {len(train_recent)}"
    )

    print(
        f"Test: {len(test)}"
    )

    distribution(
        "Test distribution",
        test["target"]
    )

    # ==================================================
    # EXPERIMENT A
    # ALL FEATURES + RECENT WINDOW
    # ==================================================

    result_all = run_experiment(
        "ALL FEATURES - RECENT",
        ALL_FEATURES,
        train_recent,
        test
    )

    # ==================================================
    # EXPERIMENT B
    # PREDICTIVE FEATURES + RECENT WINDOW
    # ==================================================

    result_predictive = run_experiment(
        "PREDICTIVE FEATURES - RECENT",
        PREDICTIVE_FEATURES,
        train_recent,
        test
    )

    # ==================================================
    # EXPERIMENT C
    # PREDICTIVE FEATURES + FULL HISTORY
    # ==================================================

    result_predictive_full = run_experiment(
        "PREDICTIVE FEATURES - FULL",
        PREDICTIVE_FEATURES,
        train_full,
        test
    )

    # ==================================================
    # FINAL COMPARISON
    # ==================================================

    print()
    print()
    print("==========================================")
    print("FINAL FEATURE COMPARISON")
    print("==========================================")

    experiments = [
        (
            "All + Recent Logistic",
            result_all["logistic"]
        ),
        (
            "All + Recent XGBoost",
            result_all["xgb"]
        ),
        (
            "Predictive + Recent Logistic",
            result_predictive["logistic"]
        ),
        (
            "Predictive + Recent XGBoost",
            result_predictive["xgb"]
        ),
        (
            "Predictive + Full Logistic",
            result_predictive_full["logistic"]
        ),
        (
            "Predictive + Full XGBoost",
            result_predictive_full["xgb"]
        ),
    ]

    print()

    for name, result in experiments:

        print(
            f"{name:32s} "
            f"Accuracy={result['accuracy']:.4f} | "
            f"Balanced={result['balanced']:.4f} | "
            f"Macro-F1={result['macro_f1']:.4f}"
        )

    print()
    print("==========================================")
    print("EXPERIMENT COMPLETE")
    print("==========================================")

    print()
    print(
        "No production models were changed."
    )


if __name__ == "__main__":
    main()