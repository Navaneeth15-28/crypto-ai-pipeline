# 📈 Crypto-AI MLOps: Deep Recurrent Sequence Modeling (PyTorch LSTM) & Streaming Pipeline

An end-to-end, academic and production-grade real-time MLOps pipeline that ingests live cryptocurrency trades from Binance, computes 57 high-frequency rolling microstructure features on the fly, trains an academic **PyTorch Deep Recurrent Neural Network (Stacked LSTM with Temporal Self-Attention)** alongside **XGBoost**, serves real-time directional predictions via Redpanda (Kafka), and provides an interactive Streamlit dashboard featuring live performance/drift monitoring, fee-adjusted paper trading, and an Explainable AI (XAI) + Generative AI RAG interface.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Ingestion
    A["Binance WebSocket (btcusdt@trade)"] -->|Live Ticks with Maker/Taker| B["Redpanda Broker: 'btc-trades'"]
    end

    subgraph Deep_Learning_and_MLOps
    B --> C["Feature Store (SQLite: historical_features.db)"]
    C -.->|Sequence Modeling| D["PyTorch Stacked LSTM + Temporal Attention (models/dl_lstm_model.pth)"]
    C -.->|Comparative Baseline| E["XGBoost Model (models/model.pkl)"]
    B --> F["Streaming Feature Engine (57 Features)"]
    F --> G["Live Inference Service (inference/realtime_predictor.py)"]
    D -.->|Loads PyTorch Model| G
    E -.->|Loads XGBoost Model| G
    G -->|Dual Predictions + Confidence| H["Redpanda Broker: 'btc-predictions'"]
    end

    subgraph UI_Monitoring_and_Paper_Trading
    H --> I["Streamlit Dashboard (ui/app.py)"]
    I --> J["Live Charts, XAI Copilot & Drift Metrics"]
    I --> K["Fee-Adjusted Paper Trading (Binance 0.04% Fees)"]
    I --> L["PyTorch Loss Curves & Comparative Evaluation"]
    end

    subgraph Semantic_RAG
    B --> M["Consumer & MiniLM Embedding (384-dim)"]
    M --> N["Qdrant Vector DB"]
    N --> O["Gemini GenAI RAG Search"]
    end
```

---

## 🎯 Prediction Task Definition

* **Target Variable:** Short-term price direction (**Binary Classification: `1` for UP, `0` for DOWN/FLAT**).
* **Prediction Horizon:** Next 60 seconds ($T + 60\text{s}$).
* **Why this task?** In high-frequency financial markets, predicting raw price regression targets is noisy and susceptible to tick-level microstructure variance. Predicting directional momentum over a fixed 60-second horizon is mathematically stable, highly actionable, and directly captures order-flow dynamics.

---

## 🔬 Deep Learning Architecture (PyTorch LSTM + Attention)

The core Deep Learning model is implemented in PyTorch (`training/train_dl_model.py`):

1. **Sequential Input Windows:** Sliding windows of shape $(B, 10, 53)$ representing the past 10 consecutive ticks across 53 standardized microstructure features.
2. **Stacked Recurrent LSTM:** 2-layer stacked LSTM with 64 hidden units and recurrent dropout ($0.25$) capturing long-range temporal momentum.
3. **Temporal Self-Attention Mechanism:**
   $$\mathbf{u}_t = \tanh(W_a \mathbf{h}_t + b_a), \quad \alpha_t = \frac{\exp(\mathbf{u}_t^T \mathbf{v})}{\sum_k \exp(\mathbf{u}_k^T \mathbf{v})}, \quad \mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t$$
   Dynamically weights which time steps in the 10-tick sequence carry the highest predictive signal.
4. **Classification MLP Head:** Context vector passes through `Linear(64 -> 32) -> BatchNorm1d -> ReLU -> Dropout(0.25) -> Linear(32 -> 1)`.
5. **Loss & Optimization:** Weighted Binary Cross-Entropy (`BCEWithLogitsLoss`) trained with Adam optimizer and `ReduceLROnPlateau` scheduler over 15 epochs.

---

## 📁 Project Structure

```text
ai-streaming-pipeline/
├── config.py                    # Centralized settings (Kafka, Qdrant, paths, thresholds)
├── features/
│   └── feature_engineer.py      # Real-time state management & 57 rolling window features
├── training/
│   ├── train_dl_model.py        # PyTorch LSTM + Self-Attention training pipeline
│   ├── offline_trainer.py       # Binary XGBoost baseline trainer
│   └── logger_and_trainer.py    # Walk-forward evaluation benchmark framework
├── inference/
│   └── realtime_predictor.py    # Live streaming inference service (Dual: PyTorch + XGBoost)
├── ui/
│   └── app.py                   # Streamlit dashboard (Live charts, Paper Trading, XAI, RAG, DL vs ML)
├── models/
│   ├── dl_lstm_model.pth        # Trained PyTorch Deep Learning LSTM weights
│   ├── dl_model_metadata.json   # PyTorch architecture, epoch loss history & test metrics
│   ├── model.pkl                # Trained production XGBoost binary model
│   └── model_metadata.json      # XGBoost metadata & drift baselines
├── historical_features.db       # Historical feature store (140,000+ ticks across 60 dimensions)
├── binance_producer.py          # WebSocket ingestion stream to Redpanda topic 'btc-trades'
├── binance_consumer.py          # Vector embedding ingestion to Qdrant vector database
├── docker-compose.yml           # Redpanda broker, Redpanda Console, and Qdrant DB
├── requirements.txt             # Project Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

* **Deep Learning Framework:** PyTorch 2.14 (`torch.nn`, `LSTM`, `TemporalAttention`)
* **Classical ML Baseline:** XGBoost, Scikit-Learn
* **Streaming Broker:** Redpanda (Kafka API on port `19092`), WebSockets
* **Vector Database:** Qdrant (Dockerized on port `6333`)
* **AI & NLP:** Google Gemini (`gemini-1.5-flash`), HuggingFace SentenceTransformers (`all-MiniLM-L6-v2`)
* **Frontend UI:** Streamlit, Plotly, Streamlit-Autorefresh

---

## 🚀 Quickstart Guide

### 1. Set Up Environment
```bash
git clone https://github.com/Navaneeth15-28/crypto-ai-pipeline.git
cd ai-streaming-pipeline

# Activate your virtual environment
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### 2. Start Infrastructure Containers
Ensure Docker Desktop is running, then start Redpanda and Qdrant:
```bash
docker compose up -d
```
* Redpanda Console: `http://localhost:8080`
* Qdrant Web UI: `http://localhost:6333/dashboard`

### 3. Train Models

* **Train PyTorch Deep Learning (LSTM + Attention):**
  ```bash
  python training/train_dl_model.py
  ```
  *(Saves `models/dl_lstm_model.pth` and `models/dl_model_metadata.json`)*

* **Train Classical ML Baseline (XGBoost):**
  ```bash
  python training/offline_trainer.py
  ```
  *(Saves `models/model.pkl` and `models/model_metadata.json`)*

### 4. Run the Pipeline Components
Open 3 separate terminals:

* **Terminal 1 (Ingestion Producer):**
  ```bash
  python binance_producer.py
  ```
* **Terminal 2 (Dual Streaming Predictor - PyTorch & XGBoost):**
  ```bash
  python inference/realtime_predictor.py
  ```
* **Terminal 3 (Unified Dashboard & UI):**
  ```bash
  streamlit run ui/app.py
  ```

---

## 🔬 Key Research & Quant Features

1. **Academic Comparative Study:** Directly benchmark Deep Sequence Modeling (**PyTorch LSTM**) against Tree Ensembles (**XGBoost**) on high-frequency financial tabular data.
2. **Order Flow Microstructure:** Ingests maker/taker flags (`is_buyer_maker`) from Binance to compute aggressive volume imbalance across 5s, 10s, 30s, and 60s windows.
3. **Fee-Adjusted Paper Trading:** Live simulation testing net returns against Binance VIP 0 taker fees ($0.04\%$) with an interactive equity curve compared to a Buy & Hold benchmark.
4. **Confidence Deadband:** Quant filter separating high-conviction trades ($P \ge 0.55 \implies \text{UP}$, $P \le 0.45 \implies \text{DOWN}$) from random noise ($0.45 < P < 0.55 \implies \text{NEUTRAL}$).
5. **Explainable AI (XAI) Trading Copilot:** Prompts Google Gemini to translate live 53-dimensional order-flow imbalances into institutional English explanations.
