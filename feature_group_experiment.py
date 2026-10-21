import sqlite3
import bisect
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)


DB_PATH = "historical_features.db"
HORIZON = 30
THRESHOLD = 0.0002


ALL_FEATURES = [
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


ORDER_FLOW_FEATURES = [
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


PRICE_MOMENTUM_FEATURES = [
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
    "price_range_60s"
]


TOP_10_FEATURES = [
    "volume_imbalance_60s",
    "trade_frequency_60s",
    "price_range_30s",
    "price_range_60s",
    "rolling_volume_30s",
    "price_momentum_30s",
    "volume_acceleration",
    "price_momentum_60s",
    "price_return_30s",
    "trade_frequency_30s"
]


# ==========================================
# LOAD DATA
# ==========================================

conn = sqlite3.connect(DB_PATH)

columns = ", ".join(
    ["price", "timestamp"] + ALL_FEATURES
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
# BUILD DATASET
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


X_all = np.array(
    [sample[0] for sample in samples],
    dtype=float
)

y = np.array(
    [sample[1] for sample in samples],
    dtype=int
)


# ==========================================
# FEATURE INDEX
# ==========================================

feature_index = {
    name: index
    for index, name in enumerate(ALL_FEATURES)
}


# ==========================================
# SPLIT
# ==========================================

split = int(len(X_all) * 0.70)

y_train = y[:split]
y_test = y[split:]


# ==========================================
# EXPERIMENT FUNCTION
# ==========================================

def run_experiment(name, feature_list):

    indexes = [
        feature_index[feature]
        for feature in feature_list
    ]

    X = X_all[:, indexes]

    X_train = X[:split]
    X_test = X[split:]

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
        f"Features          : {len(feature_list)}"
    )

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
                "UP"
            ],
            zero_division=0
        )
    )


# ==========================================
# RUN
# ==========================================

print()
print("==========================================")
print("FEATURE GROUP EXPERIMENT")
print("==========================================")

print(f"Total samples: {len(X_all)}")
print(f"Training: {split}")
print(f"Testing : {len(X_all) - split}")


run_experiment(
    "MODEL A - ORDER FLOW",
    ORDER_FLOW_FEATURES
)


run_experiment(
    "MODEL B - PRICE + MOMENTUM",
    PRICE_MOMENTUM_FEATURES
)


run_experiment(
    "MODEL C - TOP 10",
    TOP_10_FEATURES
)


print()
print("==========================================")
print("EXPERIMENT COMPLETE")
print("==========================================")