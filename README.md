# 📈 Crypto-AI Live Search (Streaming RAG Pipeline)

An end-to-end real-time Data Engineering and AI pipeline that ingests live cryptocurrency trades, vectorizes them on the fly, and allows users to query the live stream using Generative AI.

## 🏗️ Architecture
The pipeline consists of three main stages:
1. **Ingestion:** Binance WebSocket streams live BTC trades to a Redpanda message broker.
2. **Processing:** A Python consumer formats the data and uses HuggingFace MiniLM to create 384-dimensional vector embeddings.
3. **AI & UI:** A Streamlit web application allows users to query the live trade history stored in Qdrant using the Google Gemini API.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Streaming/Broker:** Redpanda (Kafka API), WebSockets
* **Vector Database:** Qdrant (Dockerized)
* **AI/ML:** Google Generative AI (Gemini), SentenceTransformers
* **Frontend:** Streamlit

## 🚀 How to Run Locally

### 1. Start Infrastructure
Make sure Docker Desktop is running, then spin up the Redpanda and Qdrant containers:
`docker compose up -d`

### 2. Start the Pipeline
You need three separate terminal windows to run the microservices simultaneously:

* **Terminal 1 (Producer):** `python binance_producer.py`
* **Terminal 2 (Consumer):** `python binance_consumer.py`
* **Terminal 3 (Web UI):** `python -m streamlit run streamlit_app.py`

### 3. Chat
Open your browser to `http://localhost:8501`, enter your free Gemini API key, and ask questions about the live trades!