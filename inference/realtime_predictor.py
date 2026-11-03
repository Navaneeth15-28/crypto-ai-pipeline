"""
Real-Time Streaming Inference Service (Dual Engine: PyTorch Deep Learning & XGBoost)
------------------------------------------------------------------------------------
Consumes live BTC trade ticks from Redpanda ('btc-trades'), computes rolling
microstructure features via StreamingFeatureEngine, performs real-time inference
using BOTH PyTorch Deep Learning (LSTM + Attention) and XGBoost, and publishes
directional predictions + confidence scores to 'btc-predictions'.
"""

# IMPORTANT FOR WINDOWS: import torch first to prevent MKL/DLL conflicts
import torch
import torch.nn as nn

import os
import sys
import json
import time
import pickle
import asyncio
from collections import deque
import numpy as np
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

# Add workspace root to sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from features.feature_engineer import StreamingFeatureEngine

# Import centralized configuration
try:
    from config import (
        KAFKA_BOOTSTRAP,
        TOPIC_TRADES as INPUT_TOPIC,
        TOPIC_PREDICTIONS as OUTPUT_TOPIC,
        CONSUMER_GROUP_INFERENCE as CONSUMER_GROUP,
        MODEL_PKL_PATH,
        ROOT_MODEL_PKL_PATH,
        METADATA_PATH,
        DL_MODEL_PATH,
        DL_METADATA_PATH,
        DL_SEQUENCE_LENGTH,
        DEADBAND_UP_THRESHOLD,
        DEADBAND_DOWN_THRESHOLD,
        FEATURE_WINDOW_SECONDS
    )
except ImportError:
    KAFKA_BOOTSTRAP = "localhost:19092"
    INPUT_TOPIC = "btc-trades"
    OUTPUT_TOPIC = "btc-predictions"
    CONSUMER_GROUP = "streaming-ml-inference-v1"
    MODEL_PKL_PATH = os.path.join(WORKSPACE_DIR, "models", "model.pkl")
    ROOT_MODEL_PKL_PATH = os.path.join(WORKSPACE_DIR, "model.pkl")
    METADATA_PATH = os.path.join(WORKSPACE_DIR, "models", "model_metadata.json")
    DL_MODEL_PATH = os.path.join(WORKSPACE_DIR, "models", "dl_lstm_model.pth")
    DL_METADATA_PATH = os.path.join(WORKSPACE_DIR, "models", "dl_model_metadata.json")
    DL_SEQUENCE_LENGTH = 10
    DEADBAND_UP_THRESHOLD = 0.55
    DEADBAND_DOWN_THRESHOLD = 0.45
    FEATURE_WINDOW_SECONDS = 60


# ============================================================
# PYTORCH DEEP LEARNING MODEL DEFINITION
# ============================================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_output):
        scores = self.attn(rnn_output)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * rnn_output, dim=1)
        return context, weights


class CryptoLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=53, hidden_dim=64, num_layers=2, dropout=0.25):
        super(CryptoLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights


# ============================================================
# MODEL LOADERS
# ============================================================

def load_models():
    """Loads both XGBoost and PyTorch Deep Learning models."""
    # 1. Load XGBoost
    xgb_model = None
    for p in [MODEL_PKL_PATH, ROOT_MODEL_PKL_PATH]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    xgb_model = pickle.load(f)
                print(f"[OK] Loaded Classical XGBoost from: {p}")
                break
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")

    # 2. Load PyTorch Deep Learning LSTM
    dl_model = None
    dl_scaler_mean = None
    dl_scaler_scale = None
    feature_cols = None

    if os.path.exists(DL_METADATA_PATH):
        try:
            with open(DL_METADATA_PATH, "r") as f:
                dl_meta = json.load(f)
                feature_cols = dl_meta.get("features")
                dl_scaler_mean = np.array(dl_meta.get("scaler_mean"), dtype=np.float32)
                dl_scaler_scale = np.array(dl_meta.get("scaler_scale"), dtype=np.float32)
                # prevent divide by zero
                dl_scaler_scale = np.where(dl_scaler_scale == 0, 1.0, dl_scaler_scale)
        except Exception as e:
            print(f"[WARN] Could not parse DL metadata: {e}")

    if os.path.exists(DL_MODEL_PATH) and feature_cols:
        try:
            dl_model = CryptoLSTMWithAttention(input_dim=len(feature_cols))
            dl_model.load_state_dict(torch.load(DL_MODEL_PATH, map_location="cpu"))
            dl_model.eval()
            print(f"[OK] Loaded PyTorch Deep Learning (LSTM+Attention) from: {DL_MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Could not load PyTorch weights: {e}")

    if not feature_cols and os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                feature_cols = json.load(f).get("features")
        except Exception:
            pass

    return xgb_model, dl_model, feature_cols, dl_scaler_mean, dl_scaler_scale


# ============================================================
# STREAMING PREDICTOR
# ============================================================

async def run_predictor():
    print("======================================================================")
    print("CRYPTO-AI MLOps: REAL-TIME STREAMING INFERENCE SERVICE")
    print("Dual Engine: PyTorch Deep Learning (LSTM) & XGBoost (Classical ML)")
    print("======================================================================")
    print(f"Kafka Broker     : {KAFKA_BOOTSTRAP}")
    print(f"Input Topic      : {INPUT_TOPIC}")
    print(f"Output Topic     : {OUTPUT_TOPIC}")
    print(f"Consumer Group   : {CONSUMER_GROUP}")
    print("======================================================================")

    xgb_model, dl_model, feature_cols, dl_mean, dl_scale = load_models()
    engine = StreamingFeatureEngine(window_size_seconds=FEATURE_WINDOW_SECONDS)
    sequence_buffer = deque(maxlen=DL_SEQUENCE_LENGTH)

    consumer = AIOKafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="latest",
        enable_auto_commit=True
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP
    )

    print("\nConnecting to Redpanda...")
    await consumer.start()
    await producer.start()
    print("[OK] Connected to Redpanda! Listening for live trades...\n")

    tick_count = 0
    pred_count = 0

    try:
        async for msg in consumer:
            recv_time = time.time()
            tick_count += 1

            try:
                trade_data = json.loads(msg.value.decode("utf-8"))
            except Exception as e:
                print(f"[ERR] Failed to decode trade JSON: {e}")
                continue

            price = float(trade_data.get("price", 0.0))
            quantity = float(trade_data.get("quantity", 0.0))
            is_buyer_maker = trade_data.get("is_buyer_maker", None)

            raw_ts = trade_data.get("event_time", trade_data.get("timestamp"))
            if raw_ts is None:
                timestamp = recv_time
            else:
                raw_ts = float(raw_ts)
                timestamp = raw_ts / 1000.0 if raw_ts > 1e11 else raw_ts

            features = engine.update_and_get_features(
                current_price=price,
                current_volume=quantity,
                current_ts=timestamp,
                is_buyer_maker=is_buyer_maker
            )

            if features is None:
                if tick_count % 10 == 0:
                    print(f"[WARMUP] Buffering initial trade ticks ({len(engine.history)}/2)...")
                continue

            try:
                feature_row = [float(features[col]) for col in feature_cols]
                X_live = np.array([feature_row], dtype=np.float32)
            except KeyError as ke:
                print(f"[ERR] Missing required feature: {ke}")
                continue

            # Standardize for PyTorch sequence buffer
            if dl_mean is not None and dl_scale is not None:
                scaled_row = (np.array(feature_row, dtype=np.float32) - dl_mean) / dl_scale
                sequence_buffer.append(scaled_row)

            # ----------------------------------------------------
            # 1. Classical ML (XGBoost) Prediction
            # ----------------------------------------------------
            xgb_prob_up = 0.5
            xgb_latency_ms = 0.0
            if xgb_model is not None:
                t0 = time.perf_counter()
                xgb_prob_up = float(xgb_model.predict_proba(X_live)[0, 1])
                xgb_latency_ms = (time.perf_counter() - t0) * 1000.0

            # ----------------------------------------------------
            # 2. Deep Learning (PyTorch LSTM + Attention) Prediction
            # ----------------------------------------------------
            dl_prob_up = xgb_prob_up
            dl_latency_ms = 0.0
            if dl_model is not None and len(sequence_buffer) >= DL_SEQUENCE_LENGTH:
                t1 = time.perf_counter()
                with torch.no_grad():
                    seq_arr = np.array([list(sequence_buffer)], dtype=np.float32)
                    seq_tensor = torch.tensor(seq_arr, dtype=torch.float32)
                    logits, _ = dl_model(seq_tensor)
                    dl_prob_up = float(torch.sigmoid(logits).item())
                dl_latency_ms = (time.perf_counter() - t1) * 1000.0

            # Active Primary Model: Use PyTorch Deep Learning if available, else XGBoost
            primary_model_type = "PyTorch_LSTM" if (dl_model and len(sequence_buffer) >= DL_SEQUENCE_LENGTH) else "XGBoost"
            active_prob_up = dl_prob_up if primary_model_type == "PyTorch_LSTM" else xgb_prob_up
            active_latency = dl_latency_ms if primary_model_type == "PyTorch_LSTM" else xgb_latency_ms

            # Quant Deadband Signal
            binary_pred = 1 if active_prob_up >= 0.50 else 0
            if active_prob_up >= DEADBAND_UP_THRESHOLD:
                direction_label = "UP"
                confidence = active_prob_up
            elif active_prob_up <= DEADBAND_DOWN_THRESHOLD:
                direction_label = "DOWN"
                confidence = 1.0 - active_prob_up
            else:
                direction_label = "NEUTRAL"
                confidence = max(active_prob_up, 1.0 - active_prob_up)

            pred_count += 1

            prediction_payload = {
                "timestamp": timestamp,
                "price": price,
                "quantity": quantity,
                "primary_model": primary_model_type,
                "binary_prediction": binary_pred,
                "direction": direction_label,
                "prob_up": round(active_prob_up, 4),
                "prob_down": round(1.0 - active_prob_up, 4),
                "confidence": round(confidence, 4),
                "latency_ms": round(active_latency, 2),
                "models": {
                    "pytorch_lstm": {
                        "prob_up": round(dl_prob_up, 4),
                        "direction": "UP" if dl_prob_up >= DEADBAND_UP_THRESHOLD else ("DOWN" if dl_prob_up <= DEADBAND_DOWN_THRESHOLD else "NEUTRAL"),
                        "latency_ms": round(dl_latency_ms, 2)
                    },
                    "xgboost": {
                        "prob_up": round(xgb_prob_up, 4),
                        "direction": "UP" if xgb_prob_up >= DEADBAND_UP_THRESHOLD else ("DOWN" if xgb_prob_up <= DEADBAND_DOWN_THRESHOLD else "NEUTRAL"),
                        "latency_ms": round(xgb_latency_ms, 2)
                    }
                },
                "features_snapshot": {
                    "price_momentum_60s": round(features.get("price_momentum_60s", 0.0), 4),
                    "volume_imbalance_60s": round(features.get("volume_imbalance_60s", 0.0), 4),
                    "volume_imbalance_30s": round(features.get("volume_imbalance_30s", 0.0), 4),
                    "buy_sell_ratio_60s": round(features.get("buy_sell_ratio_60s", 0.0), 4),
                    "buy_volume_60s": round(features.get("buy_volume_60s", 0.0), 4),
                    "sell_volume_60s": round(features.get("sell_volume_60s", 0.0), 4),
                    "volatility_30s": round(features.get("volatility_30s", 0.0), 6)
                }
            }

            # Publish prediction to Redpanda
            await producer.send_and_wait(
                OUTPUT_TOPIC,
                json.dumps(prediction_payload).encode("utf-8")
            )

            color_sig = (
                "\033[92m▲ UP\033[0m" if direction_label == "UP"
                else ("\033[91m▼ DOWN\033[0m" if direction_label == "DOWN"
                else "\033[93m━ NEUTRAL\033[0m")
            )

            print(
                f"[PRED #{pred_count:05d}] Price: ${price:>9.2f} | "
                f"Model: {primary_model_type:<12} | "
                f"Signal: {color_sig:<18} | "
                f"P(UP): {active_prob_up*100:5.1f}% | "
                f"DL Latency: {dl_latency_ms:4.2f}ms | "
                f"XGB Latency: {xgb_latency_ms:4.2f}ms"
            )

    except asyncio.CancelledError:
        print("\nPredictor shutting down...")
    except Exception as e:
        print(f"\n[ERR] Predictor error: {e}")
    finally:
        await consumer.stop()
        await producer.stop()
        print("[OK] Stopped Redpanda consumer and producer.")


def main():
    try:
        asyncio.run(run_predictor())
    except KeyboardInterrupt:
        print("\nStreaming Predictor stopped by user.")


if __name__ == "__main__":
    main()
