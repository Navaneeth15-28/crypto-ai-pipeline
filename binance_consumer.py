import asyncio
import json
import uuid

from aiokafka import AIOKafkaConsumer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer


async def consume_from_redpanda():

    # 1. Load AI embedding model
    print("Loading AI Embedding Model... (This takes a few seconds)")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("AI Model Loaded!")

    # 2. Connect to Qdrant
    qdrant = QdrantClient("http://localhost:6333")

    # Delete old collection if it exists
    try:
        qdrant.get_collection("crypto_memory")
        qdrant.delete_collection("crypto_memory")
        print("Deleted old Qdrant collection.")
    except Exception:
        pass

    # Create collection for 384-dimensional embeddings
    qdrant.create_collection(
        collection_name="crypto_memory",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        ),
    )

    print("Created fresh Qdrant collection: crypto_memory (Size: 384)")

    # 3. Connect to Redpanda
    consumer = AIOKafkaConsumer(
        "btc-trades",
        bootstrap_servers="localhost:19092",
        group_id="my-ai-consumer-group",
        auto_offset_reset="earliest"
    )

    await consumer.start()

    print("Consumer connected! Listening for new trades...")

    try:
        async for msg in consumer:

            payload = json.loads(
                msg.value.decode("utf-8")
            )

            price = payload["price"]
            quantity = payload["quantity"]

            # Convert trade into text
            memory_text = (
                f"Someone just traded "
                f"{quantity} BTC at ${price:,.2f}"
            )

            # Create 384-dimensional embedding
            embedding = model.encode(memory_text).tolist()

            # Store embedding in Qdrant
            qdrant.upsert(
                collection_name="crypto_memory",
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": memory_text,
                            "price": price,
                            "quantity": quantity
                        }
                    )
                ]
            )

            print(
                f"Vectorized & Stored -> {memory_text}"
            )

    except Exception as e:
        print(f"Error in consuming: {e}")

    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(consume_from_redpanda())