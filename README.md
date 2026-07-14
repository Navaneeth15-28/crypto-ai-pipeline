📈 Crypto-AI Live Search (Streaming RAG Pipeline)

An end-to-end real-time Data Engineering and AI pipeline that ingests live cryptocurrency trades, vectorizes them on the fly, and allows users to query the live stream using Generative AI.

🏗️ Architecture
graph TD
    subgraph Ingestion
        A[Binance WebSocket] -->|Live BTC Trades| B(Redpanda Broker)
    end
    
    subgraph Processing & Storage
        B -->|Raw JSON| C[Python Consumer]
        C -->|Text Formatted| D((HuggingFace MiniLM))
        D -->|384d Vectors| E[(Qdrant Vector DB)]
    end
    
    subgraph User Interface & AI
        F[User] -->|Asks Question| G[Streamlit Web App]
        G -->|Similarity Search| E
        E -.->|Top 3 Matches| G
        G -->|Context + Prompt| H{Google Gemini API}
        H -.->|Natural Answer| G
    end

Ingestion: Live Bitcoin trades are pulled via WebSockets from the Binance API.

Message Broker: The raw trades are pushed into Redpanda (a high-performance Kafka alternative) to handle high-throughput streaming.

Stream Processing & Embedding: A consumer reads the Redpanda stream, formats the data, and uses HuggingFace (all-MiniLM-L6-v2) to convert the text into 384-dimensional vector embeddings.

Vector Database: The embeddings are upserted into a local Qdrant database in real-time.

Generative AI UI: A Streamlit web application takes user questions, performs a similarity search in Qdrant, and uses the Google Gemini API to generate a natural, conversational answer based only on the live data stream.

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

Terminal 1 (Producer): Connects to Binance and streams to Redpanda.

python binance_producer.py


Terminal 2 (Consumer): Reads from Redpanda, vectorizes, and saves to Qdrant.

python binance_consumer.py


Terminal 3 (Web UI): Launches the AI chat interface.

python -m streamlit run streamlit_app.py


3. Chat

Open your browser to http://localhost:8501, enter your free Gemini API key, and ask questions about the live trades flowing into the system!