import sqlite3
import bisect
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


DB_PATH = "historical_features.db"

HORIZON = 30
THRESHOLD = 0.0002


FEATURES = [
    "price_return_5s",
    "price_return_10s",
    "price_return_30s",
    "price_momentum_30s",
    "price_momentum_60s",
    "momentum_acceleration",

    "rolling_std_price_5s",
    "rolling_std_price_10s",
    "volatility_10s",
    "volatility_30s",

    "price_range_10s",
    "price_range_30s",
    "price_range_60s",

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

    "volume_imbalance_60s"
]


conn = sqlite3.connect(DB_PATH)

columns = ", ".join(
    ["price", "timestamp"] + FEATURES
)

rows = conn.execute(
    f"""
    SELECT {columns}
    FROM crypto_features
    WHERE price_range_60s IS NOT NULL
    ORDER BY timestamp
    """
).fetchall()

conn.close()


print()
print("==========================================")
print("REGIME TRAINING EXPERIMENT")
print("==========================================")
print(f"Feature rows: {len(rows)}")
print()


timestamps = [
    row[1]
    for row in rows
]

prices = [
    row[0]
    for row in rows
]


# --------------------------------------------------
# Create labels
# --------------------------------------------------

samples = []

for i in range(len(rows)):

    future_index = bisect.bisect_left(
        timestamps,
        timestamps[i] + HORIZON
    )

    if future_index >= len(rows):
        continue

    future_price = prices[future_index]

    future_return = (
        future_price - prices[i]
    ) / prices[i]

    if future_return > THRESHOLD:
        label = 2       # UP

    elif future_return < -THRESHOLD:
        label = 0       # DOWN

    else:
        label = 1       # FLAT

    features = list(
        rows[i][2:]
    )

    if any(
        value is None or not np.isfinite(value)
        for value in features
    ):
        continue

    samples.append(
        (
            timestamps[i],
            features,
            label
        )
    )


print(f"Usable labeled samples: {len(samples)}")
print()


X = np.array(
    [sample[1] for sample in samples],
    dtype=float
)

y = np.array(
    [sample[2] for sample in samples],
    dtype=int
)

times = np.array(
    [sample[0] for sample in samples]
)


# --------------------------------------------------
# Chronological 70 / 30 split
# --------------------------------------------------

split = int(len(X) * 0.70)

X_train_full = X[:split]
y_train_full = y[:split]

X_test = X[split:]
y_test = y[split:]


print("CHRONOLOGICAL TEST SET")
print("------------------------------------------")

print(
    f"Training rows : {len(X_train_full)}"
)

print(
    f"Testing rows  : {len(X_test)}"
)

print()


def train_and_evaluate(
    name,
    X_train,
    y_train
):

    print()
    print("==========================================")
    print(name)
    print("==========================================")

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        "Training labels:"
    )

    for label, label_name in [
        (0, "DOWN"),
        (1, "FLAT"),
        (2, "UP")
    ]:

        count = np.sum(
            y_train == label
        )

        percentage = (
            count / len(y_train) * 100
        )

        print(
            f"{label_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
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
    print("RESULTS")
    print("------------------------------------------")

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
    print("CLASSIFICATION REPORT")
    print("------------------------------------------")

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


    print("CONFUSION MATRIX")
    print("------------------------------------------")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    print()
    print("PREDICTION COUNTS")
    print("------------------------------------------")

    for label, label_name in [
        (0, "DOWN"),
        (1, "FLAT"),
        (2, "UP")
    ]:

        count = np.sum(
            predictions == label
        )

        print(
            f"{label_name}: {count}"
        )


# --------------------------------------------------
# Experiment 1: Full history
# --------------------------------------------------

train_and_evaluate(
    "EXPERIMENT 1 — FULL HISTORY",
    X_train_full,
    y_train_full
)


# --------------------------------------------------
# Experiment 2: Recent 2500
# --------------------------------------------------

recent_2500_start = max(
    0,
    len(X_train_full) - 2500
)

train_and_evaluate(
    "EXPERIMENT 2 — RECENT 2500",
    X_train_full[recent_2500_start:],
    y_train_full[recent_2500_start:]
)


# --------------------------------------------------
# Experiment 3: Recent 1500
# --------------------------------------------------

recent_1500_start = max(
    0,
    len(X_train_full) - 1500
)

train_and_evaluate(
    "EXPERIMENT 3 — RECENT 1500",
    X_train_full[recent_1500_start:],
    y_train_full[recent_1500_start:]
)


print()
print("==========================================")
print("EXPERIMENT COMPLETE")
print("==========================================")