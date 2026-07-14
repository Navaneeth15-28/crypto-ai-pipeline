import asyncio
import json
from aiokafka import AIOKafkaConsumer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
from sentence_transformers import SentenceTransformer

async def consume_from_redpanda():
    # 1. Initialize the AI Model (This downloads a small, lightning-fast model on the first run)
    print("Loading AI Embedding Model... (This takes a few seconds)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("AI Model Loaded!")

    # 2. Connect to Qdrant
    qdrant = QdrantClient("http://localhost:6333")
    
    # We must delete the old dummy collection because our new AI vectors are size 384, not 1!
    if qdrant.collection_exists("crypto_memory"):
        qdrant.delete_collection("crypto_memory")
        
    qdrant.create_collection(
        collection_name="crypto_memory",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("Created fresh Qdrant collection: crypto_memory (Size: 384)")

    # 3. Initialize the Redpanda Consumer
    consumer = AIOKafkaConsumer(
        "btc-trades",
        bootstrap_servers='localhost:19092',
        group_id="my-ai-consumer-group",
        auto_offset_reset="earliest"
    )
    
    await consumer.start()
    print("Consumer connected! Listening for new trades...")

    try:
        async for msg in consumer:
            payload = json.loads(msg.value.decode('utf-8'))
            
            price = payload['price']
            quantity = payload['quantity']
            
            # The exact text we want the AI to understand
            memory_text = f"Someone just traded {quantity} BTC at ${price:,.2f}"
            
            # --- THE AI TRANSFORMATION ---
            # Convert the English sentence into a 384-dimensional mathematical vector
            embedding = model.encode(memory_text).tolist()
            
            # Save the true vector to Qdrant
            qdrant.upsert(
                collection_name="crypto_memory",
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=embedding,       # The real AI vector!
                        payload={"text": memory_text, "price": price, "quantity": quantity}
                    )
                ]
            )
            
            print(f"Vectorized & Stored -> {memory_text}")
            
    except Exception as e:
        print(f"Error in consuming: {e}")
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(consume_from_redpanda())