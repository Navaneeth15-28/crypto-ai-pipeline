import sqlite3
import bisect

DB_PATH = "historical_features.db"
HORIZON = 30
THRESHOLD = 0.0002
WINDOWS = 5


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

labels = []

for i, (price, timestamp) in enumerate(rows):

    future_index = bisect.bisect_left(
        timestamps,
        timestamp + HORIZON
    )

    if future_index >= len(rows):
        continue

    future_price = prices[future_index]

    future_return = (future_price - price) / price

    if future_return > THRESHOLD:
        label = 2       # UP
    elif future_return < -THRESHOLD:
        label = 0       # DOWN
    else:
        label = 1       # FLAT

    labels.append(label)


total = len(labels)
window_size = total // WINDOWS


print()
print("==========================================")
print("MARKET REGIME DIAGNOSTIC")
print("==========================================")
print(f"Total labeled rows: {total}")
print()


for i in range(WINDOWS):

    start = i * window_size

    if i == WINDOWS - 1:
        end = total
    else:
        end = (i + 1) * window_size

    window = labels[start:end]

    down = window.count(0)
    flat = window.count(1)
    up = window.count(2)

    n = len(window)

    print(
        f"WINDOW {i + 1}: "
        f"rows {start + 1}-{end}"
    )

    print(
        f"DOWN : {down:4d} "
        f"({down / n * 100:5.2f}%)"
    )

    print(
        f"FLAT : {flat:4d} "
        f"({flat / n * 100:5.2f}%)"
    )

    print(
        f"UP   : {up:4d} "
        f"({up / n * 100:5.2f}%)"
    )

    print("------------------------------------------")


print()
print("==========================================")
print("OVERALL")
print("==========================================")

print(f"DOWN : {labels.count(0)}")
print(f"FLAT : {labels.count(1)}")
print(f"UP   : {labels.count(2)}")

print("==========================================")