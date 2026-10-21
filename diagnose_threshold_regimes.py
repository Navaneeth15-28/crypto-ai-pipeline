import sqlite3
import bisect
import numpy as np


DB_PATH = "historical_features.db"
HORIZON = 30
THRESHOLD = 0.0002


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


data = []

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

    data.append(
        (
            timestamps[i],
            future_return
        )
    )


data = np.array(data)


print()
print("==========================================")
print("THRESHOLD REGIME DIAGNOSTIC")
print("==========================================")

print(f"Total usable rows: {len(data)}")
print(f"Threshold: ±{THRESHOLD * 100:.3f}%")
print()


# Divide the data into 5 chronological windows

windows = np.array_split(data, 5)


for i, window in enumerate(windows, start=1):

    returns = window[:, 1]

    down = np.sum(
        returns < -THRESHOLD
    )

    flat = np.sum(
        np.abs(returns) <= THRESHOLD
    )

    up = np.sum(
        returns > THRESHOLD
    )

    total = len(returns)

    print("------------------------------------------")
    print(f"WINDOW {i}")
    print("------------------------------------------")

    print(
        f"Rows : {total}"
    )

    print(
        f"Mean return   : "
        f"{returns.mean() * 100:.5f}%"
    )

    print(
        f"Median return : "
        f"{np.median(returns) * 100:.5f}%"
    )

    print()

    print(
        f"DOWN : {down:4d} "
        f"({down / total * 100:6.2f}%)"
    )

    print(
        f"FLAT : {flat:4d} "
        f"({flat / total * 100:6.2f}%)"
    )

    print(
        f"UP   : {up:4d} "
        f"({up / total * 100:6.2f}%)"
    )


print()
print("==========================================")
print("END")
print("==========================================")