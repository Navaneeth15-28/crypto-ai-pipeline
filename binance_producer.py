import asyncio
import websockets
import json
from aiokafka import AIOKafkaProducer

async def stream_to_redpanda():
    # 1. Initialize the Asynchronous Redpanda/Kafka Producer
    # We connect to localhost:19092, which is the external port we mapped in Docker
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:19092'
    )
    await producer.start()
    print("Successfully connected to Redpanda! Starting firehose...")

    # 2. Connect to the Binance WebSocket for BTC/USDT live trades
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    
    try:
        async with websockets.connect(url) as websocket:
            print("Connected to Binance! Streaming live market ticks directly to Redpanda...")
            
            while True:
                # Receive raw message from Binance
                raw_message = await websocket.recv()
                data = json.loads(raw_message)
                
                # Extract and clean up only the fields we care about
                # E is event time, p is price, q is quantity, t is trade ID
                payload = {
                    "event_time": data['E'],
                    "trade_id": data['t'],
                    "price": float(data['p']),
                    "quantity": float(data['q'])
                }
                
                # 3. Publish the structured data to the 'btc-trades' topic
                # We serialize the JSON payload to bytes before sending
                await producer.send_and_wait(
                    "btc-trades", 
                    json.dumps(payload).encode('utf-8')
                )
                
                print(f"Sent to Redpanda: {payload}")
                
    except Exception as e:
        print(f"Error in streaming: {e}")
    finally:
        # Gracefully stop the producer if we interrupt the loop
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(stream_to_redpanda())
""""
eof

### How to execute Step 2:

<Sequence>
  <Step title="Create the Topic in Redpanda Console">
    Since your browser is already open to the Redpanda Console from **Screenshot 2026-07-12 230041.jpg**:
    1. Click on **Topics** in the left sidebar.
    2. Click the **Create Topic** button in the top right.
    3. Name the topic exactly: `btc-trades`
    4. Leave all other default settings as they are, and click **Create**.
  </Step>
  <Step title="Install Python Libraries">
    We need the standard websocket client and `aiokafka` (which handles async Redpanda/Kafka communication). Run this in your terminal:
    ```bash
    pip install websockets aiokafka
    ```
  </Step>
  <Step title="Create the Script">
    Save the code above as `binance_producer.py` inside your `ai-streaming-pipeline` folder in VS Code.
  </Step>
  <Step title="Launch the Stream">
    Run your script:
    ```bash
    python binance_producer.py
    ```
    Your terminal will immediately begin printing out the sent payloads as they fly into your Redpanda container!
  </Step>
</Sequence>

"Once it's running, go back to your browser window at `localhost:8080/topics/btc-trades`. You will see the message counter going up live, and you can click the "Messages" tab to inspect the raw JSON data landing inside your streaming cluster!"

"Let me know when your python script starts writing live data into Redpanda."""