import sqlite3
import bisect
import numpy as np


DB_PATH = "historical_features.db"
HORIZON = 30


conn = sqlite3.connect(DB_PATH)

rows = conn.execute(
    """
    SELECT price, timestamp
    FROM crypto_features
    WHERE price_range_60s IS NOT NULL
    ORDER BY timestamp
    """
).fetchall()

conn.close()


prices = [row[0] for row in rows]
timestamps = [row[1] for row in rows]


returns = []

for i in range(len(rows)):

    future_index = bisect.bisect_left(
        timestamps,
        timestamps[i] + HORIZON
    )

    if future_index >= len(rows):
        continue

    current_price = prices[i]
    future_price = prices[future_index]

    future_return = (
        future_price - current_price
    ) / current_price

    returns.append(future_return)


returns = np.array(returns)


print()
print("==========================================")
print("30-SECOND FUTURE RETURN DIAGNOSTIC")
print("==========================================")

print(
    f"Usable returns: {len(returns)}"
)

print()

print("RETURN STATISTICS")
print("------------------------------------------")

print(
    f"Minimum       : {returns.min() * 100:.5f}%"
)

print(
    f"Maximum       : {returns.max() * 100:.5f}%"
)

print(
    f"Mean          : {returns.mean() * 100:.5f}%"
)

print(
    f"Median        : {np.median(returns) * 100:.5f}%"
)

print(
    f"Std deviation : {returns.std() * 100:.5f}%"
)

print()

print("PERCENTILES")
print("------------------------------------------")

for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:

    value = np.percentile(
        returns,
        p
    )

    print(
        f"{p:2d}th percentile : "
        f"{value * 100:.5f}%"
    )


print()
print("THRESHOLD ANALYSIS")
print("------------------------------------------")


thresholds = [
    0.0001,
    0.0002,
    0.0003,
    0.0005,
    0.0010
]


for threshold in thresholds:

    down = np.sum(
        returns < -threshold
    )

    flat = np.sum(
        np.abs(returns) <= threshold
    )

    up = np.sum(
        returns > threshold
    )

    total = len(returns)

    print()

    print(
        f"Threshold ±{threshold * 100:.3f}%"
    )

    print(
        f"DOWN : {down:4d} "
        f"({down / total * 100:5.2f}%)"
    )

    print(
        f"FLAT : {flat:4d} "
        f"({flat / total * 100:5.2f}%)"
    )

    print(
        f"UP   : {up:4d} "
        f"({up / total * 100:5.2f}%)"
    )


print()
print("==========================================")
print("END")
print("==========================================")