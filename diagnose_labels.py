import sqlite3
import bisect


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

labels = {
    "DOWN": 0,
    "FLAT": 0,
    "UP": 0
}

usable = 0
no_future_data = 0


for i in range(len(rows)):

    price = prices[i]
    timestamp = timestamps[i]

    future_index = bisect.bisect_left(
        timestamps,
        timestamp + HORIZON
    )

    if future_index >= len(rows):
        no_future_data += 1
        continue

    future_price = prices[future_index]

    future_return = (future_price - price) / price

    if future_return > THRESHOLD:
        labels["UP"] += 1

    elif future_return < -THRESHOLD:
        labels["DOWN"] += 1

    else:
        labels["FLAT"] += 1

    usable += 1


print("==========================================")
print("LABEL DIAGNOSTIC")
print("==========================================")
print(f"Total feature rows : {len(rows)}")
print(f"Usable rows        : {usable}")
print(f"No future data     : {no_future_data}")
print()
print("30-SECOND FUTURE LABELS")
print("------------------------------------------")
print(f"DOWN : {labels['DOWN']}")
print(f"FLAT : {labels['FLAT']}")
print(f"UP   : {labels['UP']}")
print("------------------------------------------")

if usable > 0:
    print()
    print("Percentages:")
    print(f"DOWN : {labels['DOWN'] / usable * 100:.2f}%")
    print(f"FLAT : {labels['FLAT'] / usable * 100:.2f}%")
    print(f"UP   : {labels['UP'] / usable * 100:.2f}%")

print("==========================================")