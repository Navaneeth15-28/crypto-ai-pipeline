"""
Centralized Configuration for Crypto-AI MLOps Pipeline
------------------------------------------------------
Consolidates connection strings, topic names, file paths, model parameters,
and quant trading thresholds across the entire pipeline.
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------
# STREAMING & INFRASTRUCTURE
# ------------------------------------------------------------
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")

TOPIC_TRADES = "btc-trades"
TOPIC_PREDICTIONS = "btc-predictions"

CONSUMER_GROUP_INFERENCE = "streaming-ml-inference-v1"
CONSUMER_GROUP_RAG = "crypto-rag-consumer-v1"

# ------------------------------------------------------------
# DATABASE & STORAGE
# ------------------------------------------------------------
DB_PATH = os.path.join(BASE_DIR, "historical_features.db")
MODEL_PKL_PATH = os.path.join(MODELS_DIR, "model.pkl")
ROOT_MODEL_PKL_PATH = os.path.join(BASE_DIR, "model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Deep Learning (PyTorch LSTM + Attention)
DL_MODEL_PATH = os.path.join(MODELS_DIR, "dl_lstm_model.pth")
DL_METADATA_PATH = os.path.join(MODELS_DIR, "dl_model_metadata.json")
DL_SEQUENCE_LENGTH = 10

# Vector Database (Qdrant)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION = "crypto_memory"

# ------------------------------------------------------------
# ML & QUANT TRADING PARAMETERS
# ------------------------------------------------------------
PREDICTION_HORIZON_SECONDS = 60
FEATURE_WINDOW_SECONDS = 60

# Quant Deadband Thresholds (High Conviction Filters)
DEADBAND_UP_THRESHOLD = 0.55
DEADBAND_DOWN_THRESHOLD = 0.45

# Binance VIP 0 Taker Fee (0.04% per trade)
BINANCE_TAKER_FEE_PCT = 0.0004

# Paper Trading Simulation
INITIAL_PAPER_CAPITAL = 10000.0  # $10,000 USDT starting portfolio

# ------------------------------------------------------------
# GENAI & EMBEDDING MODELS
# ------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
GEMINI_MODEL_NAME = "gemini-3.8-flash"
