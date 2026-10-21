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
# BUILD FUTURE RETURNS
# ==========================================

returns = []
feature_rows = []

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

    features = list(rows[i][2:])

    if any(
        value is None or not np.isfinite(value)
        for value in features
    ):
        continue

    returns.append(future_return)
    feature_rows.append(features)


returns = np.array(returns)
X_raw = np.array(feature_rows, dtype=float)


# ==========================================
# CALCULATE PERCENTILE THRESHOLDS
# ==========================================

lower_threshold = np.percentile(returns, 25)
upper_threshold = np.percentile(returns, 75)


print()
print("==========================================")
print("PERCENTILE LABEL EXPERIMENT")
print("==========================================")

print(f"Usable samples: {len(returns)}")

print()
print("PERCENTILE THRESHOLDS")
print("------------------------------------------")

print(
    f"25th percentile: "
    f"{lower_threshold * 100:.5f}%"
)

print(
    f"75th percentile: "
    f"{upper_threshold * 100:.5f}%"
)


# ==========================================
# CREATE LABELS
# ==========================================
#
# 0 = DOWN
# 1 = FLAT / NEUTRAL
# 2 = UP
#
# Bottom 25%  -> DOWN
# Middle 50%  -> FLAT
# Top 25%     -> UP
# ==========================================

y = np.zeros(len(returns), dtype=int)

y[
    returns > lower_threshold
] = 1

y[
    returns > upper_threshold
] = 2


# ==========================================
# LABEL DISTRIBUTION
# ==========================================

print()
print("LABEL DISTRIBUTION")
print("------------------------------------------")

down_count = np.sum(y == 0)
flat_count = np.sum(y == 1)
up_count = np.sum(y == 2)

print(
    f"DOWN : {down_count} "
    f"({down_count / len(y) * 100:.2f}%)"
)

print(
    f"FLAT : {flat_count} "
    f"({flat_count / len(y) * 100:.2f}%)"
)

print(
    f"UP   : {up_count} "
    f"({up_count / len(y) * 100:.2f}%)"
)


# ==========================================
# CHRONOLOGICAL SPLIT
# ==========================================

split = int(len(X_raw) * 0.70)

X_train = X_raw[:split]
y_train = y[:split]

X_test = X_raw[split:]
y_test = y[split:]


print()
print("CHRONOLOGICAL SPLIT")
print("------------------------------------------")

print(f"Training: {len(X_train)}")
print(f"Testing : {len(X_test)}")


# ==========================================
# MODEL
# ==========================================

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


# ==========================================
# TRAIN
# ==========================================

model.fit(
    X_train,
    y_train
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
            "FLAT",
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
    f"FLAT: {np.sum(predictions == 1)}"
)

print(
    f"UP  : {np.sum(predictions == 2)}"
)


print()
print("==========================================")
print("EXPERIMENT COMPLETE")
print("==========================================")