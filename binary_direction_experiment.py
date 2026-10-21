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


prices = [
    row[0]
    for row in rows
]

timestamps = [
    row[1]
    for row in rows
]


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


    # Ignore FLAT samples
    if abs(future_return) <= THRESHOLD:
        continue


    if future_return > THRESHOLD:
        label = 1       # UP

    else:
        label = 0       # DOWN


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
            features,
            label
        )
    )


X = np.array(
    [sample[0] for sample in samples],
    dtype=float
)

y = np.array(
    [sample[1] for sample in samples],
    dtype=int
)


print()
print("==========================================")
print("BINARY DIRECTION EXPERIMENT")
print("==========================================")

print(
    f"Usable samples: {len(X)}"
)


print()
print("LABEL DISTRIBUTION")
print("------------------------------------------")


down_count = np.sum(
    y == 0
)

up_count = np.sum(
    y == 1
)


print(
    f"DOWN: {down_count} "
    f"({down_count / len(y) * 100:.2f}%)"
)

print(
    f"UP  : {up_count} "
    f"({up_count / len(y) * 100:.2f}%)"
)


# --------------------------------------------------
# Chronological split
# --------------------------------------------------

split = int(
    len(X) * 0.70
)


X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]


print()
print("CHRONOLOGICAL SPLIT")
print("------------------------------------------")

print(
    f"Training: {len(X_train)}"
)

print(
    f"Testing : {len(X_test)}"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
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


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

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
print("RESULTS")
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
print("CLASSIFICATION REPORT")
print("------------------------------------------")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "DOWN",
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

print(
    f"DOWN: {np.sum(predictions == 0)}"
)

print(
    f"UP  : {np.sum(predictions == 1)}"
)


print()
print("==========================================")
print("EXPERIMENT COMPLETE")
print("==========================================")