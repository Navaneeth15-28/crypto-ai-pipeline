import asyncio
import websockets


async def test():
    print("Connecting to Binance...")

    async with websockets.connect(
        "wss://stream.binance.com:9443/ws/btcusdt@trade",
        open_timeout=20
    ):
        print("BINANCE WEBSOCKET OK")


asyncio.run(test())