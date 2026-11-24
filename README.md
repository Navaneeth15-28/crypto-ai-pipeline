# 📈 Crypto-AI MLOps: Deep Recurrent Sequence Modeling (PyTorch LSTM) & Streaming Pipeline

An end-to-end, academic and production-grade real-time MLOps pipeline that ingests live cryptocurrency trades from Binance, computes 57 high-frequency rolling microstructure features on the fly, trains an academic **PyTorch Deep Recurrent Neural Network (Stacked LSTM with Temporal Self-Attention)** alongside **XGBoost**, serves real-time directional predictions via Redpanda (Kafka), and provides an interactive Streamlit dashboard featuring live performance/drift monitoring, fee-adjusted paper trading, and an Explainable AI (XAI) + Generative AI RAG interface.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    %% SUBGRAPH 1: INGESTION
    subgraph S1 ["1. Live Ingestion & Message Broker"]
        A["📡 Binance WebSocket<br/><code>wss://stream.binance.com</code>"]
        B["⚡ binance_producer.py<br/><i>AsyncIO WebSocket Client</i>"]
        C{{"📬 Redpanda / Kafka Broker<br/><b>Topic: 'btc-trades'</b><br/><i>Port: 19092</i>"}}
        A -->|Live Ticks: Price, Qty, Maker Flag| B
        B -->|Sub-millisecond JSON Pub| C
    end

    %% SUBGRAPH 2: FEATURE ENGINEERING
    subgraph S2 ["2. Microstructure Feature Store"]
        D["⚙️ StreamingFeatureEngine<br/><i>57 Rolling Window Features</i><br/>(Imbalance, Momentum, Volatility)"]
        E[("💾 SQLite Feature Store<br/><b>historical_features.db</b><br/><i>140,000+ Real Historical Ticks</i>")]
        C -->|Live Tick Stream| D
        D -->|Persist Features| E
    end

    %% SUBGRAPH 3: DUAL ML INFERENCE
    subgraph S3 ["3. Dual Model Inference Engine"]
        F["🧠 PyTorch Deep Learning Model<br/><b>Stacked 2-Layer LSTM + Self-Attention</b><br/><i>(10-Tick &times; 53-Feature Sliding Window)</i>"]
        G["🌲 XGBoost Baseline Model<br/><b>Gradient Boosted Trees (Depth 5)</b><br/><i>Tabular Benchmark</i>"]
        H["🚀 realtime_predictor.py<br/><b>Dual Inference Service</b><br/><i>Latency: &lt; 3.5ms | Deadband: 55%</i>"]
        I{{"🎯 Redpanda / Kafka Broker<br/><b>Topic: 'btc-predictions'</b><br/><i>P(UP), Signal, Confidence</i>"}}
        
        E -.->|Sequence Tensor [B, 10, 53]| F
        E -.->|Tabular Feature Vector| G
        F -->|Temporal Prediction| H
        G -->|Baseline Prediction| H
        H -->|Streaming Output| I
    end

    %% SUBGRAPH 4: DASHBOARD & PAPER TRADING
    subgraph S4 ["4. Interactive Streamlit Dashboard & MLOps"]
        J["📊 Streamlit Web App (ui/app.py)<br/><i>Localhost: 8501</i>"]
        K["💰 $10,000 Paper Trading Simulator<br/><i>Fee-Adjusted: Binance 0.04% Taker</i>"]
        L["📈 Production MLOps Monitoring<br/><i>Concept Drift & Latency Gauges</i>"]
        
        I -->|Sub-second Consumer| J
        J --> K
        J --> L
    end

    %% SUBGRAPH 5: GENAI & STREAMING RAG
    subgraph S5 ["5. Explainable AI & Streaming RAG"]
        M["🔍 binance_consumer.py<br/><i>Embeddings: all-MiniLM-L6-v2</i>"]
        N[("🗄️ Qdrant Vector Database<br/><b>Port: 6333</b><br/><i>384-Dim Trade Embeddings</i>")]
        O["🤖 Google Gemini 3.8 Flash<br/><b>Explainable AI (XAI) & RAG</b><br/><i>Natural Language Order-Flow Reasoning</i>"]
        
        C -->|Raw Trade Stream| M
        M -->|Upsert Dense Vectors| N
        N <-->|Semantic Vector Search| J
        J <-->|Microstructure Prompt & Features| O
    end

    %% Styling
    style S1 fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style S2 fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#f8fafc
    style S3 fill:#0f172a,stroke:#c084fc,stroke-width:2px,color:#f8fafc
    style S4 fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#f8fafc
    style S5 fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#f8fafc

    style A fill:#1e293b,stroke:#f59e0b,stroke-width:1.5px,color:#f8fafc
    style B fill:#1e293b,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
    style C fill:#78350f,stroke:#f59e0b,stroke-width:2px,color:#fef3c7
    style D fill:#1e293b,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
    style E fill:#1e293b,stroke:#64748b,stroke-width:1.5px,color:#f8fafc
    style F fill:#4c1d95,stroke:#c084fc,stroke-width:2px,color:#fae8ff
    style G fill:#1e293b,stroke:#f59e0b,stroke-width:1.5px,color:#f8fafc
    style H fill:#1e293b,stroke:#34d399,stroke-width:2px,color:#f8fafc
    style I fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5
    style J fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5
    style K fill:#1e293b,stroke:#34d399,stroke-width:1.5px,color:#f8fafc
    style L fill:#1e293b,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
    style M fill:#1e293b,stroke:#a855f7,stroke-width:1.5px,color:#f8fafc
    style N fill:#4c1d95,stroke:#a855f7,stroke-width:2px,color:#f5d0fe
    style O fill:#1e3a8a,stroke:#38bdf8,stroke-width:2px,color:#e0f2fe
```

<div align="center">
  <p><sub><i>💡 Interactive, native GitHub Mermaid architecture diagram. You can also inspect the vector SVG layout at <a href="./assets/architecture.svg">assets/architecture.svg</a>.</i></sub></p>
</div>

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
* **AI & NLP:** Google Gemini (`gemini-3.8-flash`), HuggingFace SentenceTransformers (`all-MiniLM-L6-v2`)
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
