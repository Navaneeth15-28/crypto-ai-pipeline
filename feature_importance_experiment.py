import sqlite3
import bisect
import numpy as np

from xgboost import XGBClassifier


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
        label = 1
    else:
        label = 0

    features = list(rows[i][2:])

    if any(
        value is None or not np.isfinite(value)
        for value in features
    ):
        continue

    samples.append(
        (features, label)
    )


X = np.array(
    [sample[0] for sample in samples],
    dtype=float
)

y = np.array(
    [sample[1] for sample in samples],
    dtype=int
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
print("==========================================")
print("FEATURE IMPORTANCE EXPERIMENT")
print("==========================================")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ==========================================
# TRAIN MODEL
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


model.fit(
    X_train,
    y_train
)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

pairs = list(
    zip(FEATURES, importance)
)

pairs.sort(
    key=lambda x: x[1],
    reverse=True
)


print()
print("==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")

print(
    f"{'Rank':<6}"
    f"{'Feature':<32}"
    f"{'Importance':<12}"
)

print("-" * 50)


for rank, (feature, score) in enumerate(
    pairs,
    start=1
):

    print(
        f"{rank:<6}"
        f"{feature:<32}"
        f"{score:<12.6f}"
    )


print()
print("==========================================")
print("EXPERIMENT COMPLETE")
print("==========================================")