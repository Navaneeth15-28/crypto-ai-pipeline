from confluent_kafka import Consumer
import json
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from features.feature_engineer import StreamingFeatureEngine
from training.logger_and_trainer import DataLogger


# ==========================================================
# KAFKA / REDPANDA CONFIGURATION
# ==========================================================

consumer = Consumer({
    "bootstrap.servers": "localhost:19092",
    "group.id": "data-collector-v2",
    "auto.offset.reset": "latest"
})

consumer.subscribe(
    ["btc-trades"]
)


# ==========================================================
# FEATURE ENGINE
# ==========================================================

engine = StreamingFeatureEngine(
    window_size_seconds=60
)


# ==========================================================
# DATABASE LOGGER
# ==========================================================

logger = DataLogger(
    "historical_features.db"
)


# ==========================================================
# START
# ==========================================================

print("==========================================")
print("LIVE FEATURE DATA COLLECTOR")
print("==========================================")
print("Listening for BTC trades...")
print("Building streaming features...")
print("Saving data to SQLite...")
print("==========================================")


# ==========================================================
# MAIN LOOP
# ==========================================================

try:

    while True:

        msg = consumer.poll(
            1.0
        )

        # No message yet
        if msg is None:
            continue

        # Kafka error
        if msg.error():

            print(
                "Kafka error:",
                msg.error()
            )

            continue

        try:

            # ------------------------------------------------
            # DECODE MESSAGE
            # ------------------------------------------------

            raw_value = msg.value().decode(
                "utf-8"
            )

            data = json.loads(
                raw_value
            )

            # ------------------------------------------------
            # READ PRICE
            # ------------------------------------------------

            price = float(
                data["price"]
            )

            # ------------------------------------------------
            # READ TRADE QUANTITY
            # ------------------------------------------------

            quantity = float(
                data["quantity"]
            )

            # ------------------------------------------------
            # READ TIMESTAMP
            #
            # Actual producer uses event_time.
            # ------------------------------------------------

            timestamp = data.get(
                "event_time"
            )

            if timestamp is None:

                timestamp = data.get(
                    "timestamp"
                )

            if timestamp is None:

                print(
                    "Message has no timestamp. Skipping."
                )

                continue

            # Binance timestamps are milliseconds.
            #
            # Convert:
            #
            # milliseconds → seconds
            # ------------------------------------------------

            timestamp = float(
                timestamp
            ) / 1000.0

            # ------------------------------------------------
            # BUYER/SELLER INFORMATION
            #
            # Current producer doesn't provide this field.
            # Therefore it remains None.
            # ------------------------------------------------

            is_buyer_maker = data.get(
                "is_buyer_maker"
            )

            # ------------------------------------------------
            # CREATE FEATURES
            # ------------------------------------------------

            features = (
                engine.update_and_get_features(
                    current_price=price,
                    current_volume=quantity,
                    current_ts=timestamp,
                    is_buyer_maker=is_buyer_maker
                )
            )

            # Need at least 2 trades
            if features is None:

                continue

            # ------------------------------------------------
            # SAVE TO DATABASE
            # ------------------------------------------------

            logger.log_tick(
                timestamp,
                price,
                quantity,
                features
            )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            print(
                f"Saved | "
                f"Price: ${price:,.2f} | "
                f"Trades/60s: "
                f"{features['trade_frequency_60s']} | "
                f"Momentum: "
                f"{features['price_momentum_60s']:.4f}%"
            )

        except Exception as e:

            print(
                "Error processing trade:",
                e
            )


except KeyboardInterrupt:

    print()
    print(
        "Stopping collector..."
    )


finally:

    logger.close()

    consumer.close()

    print(
        "Collector stopped."
    )