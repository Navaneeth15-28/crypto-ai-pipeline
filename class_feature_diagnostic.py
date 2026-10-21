import sqlite3
import bisect
import numpy as np
import pandas as pd


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


    if future_return > THRESHOLD:
        label = "UP"

    elif future_return < -THRESHOLD:
        label = "DOWN"

    else:
        label = "FLAT"


    features = list(
        rows[i][2:]
    )


    if any(
        value is None or not np.isfinite(value)
        for value in features
    ):
        continue


    samples.append(
        features + [label]
    )


df = pd.DataFrame(
    samples,
    columns=FEATURES + ["label"]
)


print()
print("==========================================")
print("CLASS FEATURE DIAGNOSTIC")
print("==========================================")

print(
    f"Total usable samples: {len(df)}"
)

print()


print("CLASS COUNTS")
print("------------------------------------------")

print(
    df["label"].value_counts()
)


print()
print("==========================================")
print("FEATURE STATISTICS BY CLASS")
print("==========================================")


for feature in FEATURES:

    print()
    print("------------------------------------------")
    print(feature)
    print("------------------------------------------")

    for label in ["DOWN", "FLAT", "UP"]:

        values = df.loc[
            df["label"] == label,
            feature
        ]

        print(
            f"{label:5s} | "
            f"mean: {values.mean():.6f} | "
            f"median: {values.median():.6f}"
        )


print()
print("==========================================")
print("CLASS SEPARATION")
print("==========================================")


for feature in FEATURES:

    down_mean = df.loc[
        df["label"] == "DOWN",
        feature
    ].mean()

    flat_mean = df.loc[
        df["label"] == "FLAT",
        feature
    ].mean()

    up_mean = df.loc[
        df["label"] == "UP",
        feature
    ].mean()


    separation = max(
        abs(down_mean - flat_mean),
        abs(flat_mean - up_mean),
        abs(down_mean - up_mean)
    )


    print(
        f"{feature:30s} "
        f"separation: {separation:.6f}"
    )


print()
print("==========================================")
print("END")
print("==========================================")