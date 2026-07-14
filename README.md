# 📈 Crypto-AI Live Search (Streaming RAG Pipeline)

An end-to-end real-time Data Engineering and AI pipeline that ingests live cryptocurrency trades, vectorizes them on the fly, and allows users to query the live stream using Generative AI.

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Ingestion
    A["Binance WebSocket"] -->|Live BTC Trades| B["Redpanda Broker"]
    end
    subgraph Processing_Storage
    B -->|Raw JSON| C["Python Consumer"]
    C -->|Text Formatted| D(("HuggingFace MiniLM"))
    D -->|384d Vectors| E(("Qdrant Vector DB"))
    end
    subgraph UI_AI
    F["User"] -->|Asks Question| G["Streamlit Web App"]
    G -->|Similarity Search| E
    E -.->|Top 3 Matches| G
    G -->|Context + Prompt| H{"Google Gemini API"}
    H -.->|Natural Answer| G
    end
    ```
🛠️ Tech Stack
Language: Python 3.x

Streaming/Broker: Redpanda (Kafka API), WebSockets

Vector Database: Qdrant (Dockerized)

AI/ML: Google Generative AI (Gemini), SentenceTransformers

Frontend: Streamlit

🚀 How to Run Locally
1. Start Infrastructure
Make sure Docker Desktop is running, then spin up the Redpanda and Qdrant containers:
docker compose up -d

2. Start the Pipeline
You need three separate terminal windows to run the microservices simultaneously:

Terminal 1 (Producer): python binance_producer.py

Terminal 2 (Consumer): python binance_consumer.py

Terminal 3 (Web UI): python -m streamlit run streamlit_app.py

3. Chat
Open your browser to http://localhost:8501, enter your free Gemini API key, and ask questions about the live trades!