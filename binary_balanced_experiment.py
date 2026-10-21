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


# ==========================================
# LOAD DATA
# ==========================================

conn = sqlite3.connect(DB_PATH)

columns = ", ".join(["price", "timestamp"] + FEATURES)

rows = conn.execute(
    f"""
    SELECT {columns}
    FROM crypto_features
    WHERE price_range_60s IS NOT NULL
    ORDER BY timestamp
    """
).fetchall()

conn.close()

prices = [row[0] for row in rows]
timestamps = [row[1] for row in rows]


# ==========================================
# BUILD BINARY DATASET
# ==========================================

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
        (future_price - prices[i])
        / prices[i]
    )

    # Ignore FLAT
    if abs(future_return) <= THRESHOLD:
        continue

    if future_return > THRESHOLD:
        label = 1       # UP
    else:
        label = 0       # DOWN

    features = list(rows[i][2:])

    if any(
        value is None or not np.isfinite(value)
        for value in features
    ):
        continue

    samples.append((features, label))


X = np.array(
    [sample[0] for sample in samples],
    dtype=float
)

y = np.array(
    [sample[1] for sample in samples],
    dtype=int
)


# ==========================================
# INFORMATION
# ==========================================

print()
print("==========================================")
print("BALANCED BINARY EXPERIMENT")
print("==========================================")

print(f"Usable samples: {len(X)}")


# ==========================================
# LABEL DISTRIBUTION
# ==========================================

down_count = np.sum(y == 0)
up_count = np.sum(y == 1)

print()
print("LABEL DISTRIBUTION")
print("------------------------------------------")

print(
    f"DOWN: {down_count} "
    f"({down_count / len(y) * 100:.2f}%)"
)

print(
    f"UP  : {up_count} "
    f"({up_count / len(y) * 100:.2f}%)"
)


# ==========================================
# CHRONOLOGICAL SPLIT
# ==========================================

split = int(len(X) * 0.70)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

print()
print("CHRONOLOGICAL SPLIT")
print("------------------------------------------")

print(f"Training: {len(X_train)}")
print(f"Testing : {len(X_test)}")


# ==========================================
# CALCULATE BALANCED CLASS WEIGHTS
# ==========================================

train_down = np.sum(y_train == 0)
train_up = np.sum(y_train == 1)

total_train = len(y_train)

weight_down = (
    total_train / (2 * train_down)
)

weight_up = (
    total_train / (2 * train_up)
)

sample_weights = np.where(
    y_train == 0,
    weight_down,
    weight_up
)

print()
print("BALANCED CLASS WEIGHTS")
print("------------------------------------------")

print(f"DOWN weight: {weight_down:.4f}")
print(f"UP weight  : {weight_up:.4f}")


# ==========================================
# MODEL
# ==========================================

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


# ==========================================
# TRAIN
# ==========================================

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)


# ==========================================
# PREDICT
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# METRICS
# ==========================================

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


# ==========================================
# RESULTS
# ==========================================

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


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

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


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("CONFUSION MATRIX")
print("------------------------------------------")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==========================================
# PREDICTION COUNTS
# ==========================================

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