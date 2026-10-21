import asyncio
import json
import time

from aiokafka import AIOKafkaProducer
import websockets


# ============================================================
# CONFIGURATION
# ============================================================

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

KAFKA_BOOTSTRAP = "localhost:19092"

TOPIC = "btc-trades"


# ============================================================
# PRODUCER
# ============================================================

async def produce():

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP
    )

    await producer.start()

    print("==========================================")
    print("BINANCE BTC TRADE PRODUCER")
    print("==========================================")

    print(
        f"WebSocket : {BINANCE_WS_URL}"
    )

    print(
        f"Kafka     : {KAFKA_BOOTSTRAP}"
    )

    print(
        f"Topic     : {TOPIC}"
    )

    print("==========================================")
    print("Connecting to Binance...")
    print()

    try:

        async with websockets.connect(
            BINANCE_WS_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10
        ) as websocket:

            print(
                "Connected to Binance WebSocket."
            )

            print(
                "Streaming BTC trades..."
            )

            print()

            while True:

                try:

                    message = await websocket.recv()

                    data = json.loads(
                        message
                    )

                    # ------------------------------------------------
                    # Binance trade event
                    # ------------------------------------------------

                    event_time = data.get("E")
                    trade_id = data.get("t")
                    price = data.get("p")
                    quantity = data.get("q")

                    # IMPORTANT:
                    # True Binance trade-side information.
                    #
                    # True  = buyer is maker
                    # False = buyer is taker
                    #
                    # This allows the feature engine to calculate
                    # directional volume imbalance.
                    # ------------------------------------------------

                    is_buyer_maker = data.get(
                        "m",
                        None
                    )

                    if (
                        event_time is None
                        or trade_id is None
                        or price is None
                        or quantity is None
                    ):

                        print(
                            "Incomplete Binance message. Skipping."
                        )

                        continue

                    # ------------------------------------------------
                    # Build our Kafka message
                    # ------------------------------------------------

                    trade = {

                        "event_time":
                            int(event_time),

                        "trade_id":
                            int(trade_id),

                        "price":
                            float(price),

                        "quantity":
                            float(quantity),

                        "is_buyer_maker":
                            bool(is_buyer_maker)
                    }

                    # ------------------------------------------------
                    # Send to Redpanda
                    # ------------------------------------------------

                    await producer.send_and_wait(
                        TOPIC,
                        json.dumps(
                            trade
                        ).encode("utf-8")
                    )

                    # ------------------------------------------------
                    # Display
                    # ------------------------------------------------

                    print(
                        f"Trade | "
                        f"Price: ${price} | "
                        f"Qty: {quantity} | "
                        f"Maker: {is_buyer_maker}"
                    )

                except websockets.exceptions.ConnectionClosed:

                    print()
                    print(
                        "Binance connection closed."
                    )

                    print(
                        "Reconnecting..."
                    )

                    break

                except json.JSONDecodeError:

                    print(
                        "Invalid JSON received. Skipping."
                    )

                except Exception as e:

                    print(
                        f"Error processing trade: {e}"
                    )

                    await asyncio.sleep(1)

    except Exception as e:

        print(
            f"WebSocket connection error: {e}"
        )

    finally:

        await producer.stop()

        print()
        print(
            "Producer stopped."
        )


# ============================================================
# AUTO RECONNECT LOOP
# ============================================================

async def main():

    while True:

        try:

            await produce()

        except KeyboardInterrupt:

            print()
            print(
                "Producer stopped by user."
            )

            break

        except Exception as e:

            print()
            print(
                f"Producer crashed: {e}"
            )

            print(
                "Restarting in 5 seconds..."
            )

            await asyncio.sleep(5)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "Producer stopped."
        )