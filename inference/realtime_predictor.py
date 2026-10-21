"""
Real-Time Streaming Inference Service
-------------------------------------
Consumes live BTC trade ticks from Redpanda ('btc-trades'), computes rolling
microstructure features via StreamingFeatureEngine, performs real-time XGBoost
inference, and publishes directional predictions + confidence scores to 'btc-predictions'.
"""

import os
import sys
import json
import time
import pickle
import asyncio
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
        DEADBAND_UP_THRESHOLD,
        DEADBAND_DOWN_THRESHOLD,
        FEATURE_WINDOW_SECONDS
    )
    MODEL_PATHS = [MODEL_PKL_PATH, ROOT_MODEL_PKL_PATH]
except ImportError:
    KAFKA_BOOTSTRAP = "localhost:19092"
    INPUT_TOPIC = "btc-trades"
    OUTPUT_TOPIC = "btc-predictions"
    CONSUMER_GROUP = "streaming-ml-inference-v1"
    MODEL_PATHS = [
        os.path.join(WORKSPACE_DIR, "models", "model.pkl"),
        os.path.join(WORKSPACE_DIR, "model.pkl")
    ]
    METADATA_PATH = os.path.join(WORKSPACE_DIR, "models", "model_metadata.json")
    DEADBAND_UP_THRESHOLD = 0.55
    DEADBAND_DOWN_THRESHOLD = 0.45
    FEATURE_WINDOW_SECONDS = 60


def load_model_and_metadata():
    """Loads trained XGBoost model and feature schema."""
    model = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    model = pickle.load(f)
                print(f"[OK] Loaded XGBoost model from: {p}")
                break
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")

    if model is None:
        raise FileNotFoundError(
            "Could not find model.pkl! Please run 'python training/offline_trainer.py' first."
        )

    feature_cols = None
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
                feature_cols = meta.get("features")
                print(f"[OK] Loaded feature schema ({len(feature_cols)} features) from metadata.")
        except Exception as e:
            print(f"[WARN] Could not parse metadata: {e}")

    if not feature_cols:
        raise ValueError("Could not determine feature schema. Ensure model_metadata.json exists.")

    return model, feature_cols


async def run_predictor():
    print("======================================================================")
    print("CRYPTO-AI MLOps: STREAMING INFERENCE SERVICE")
    print("======================================================================")
    print(f"Kafka Broker     : {KAFKA_BOOTSTRAP}")
    print(f"Input Topic      : {INPUT_TOPIC}")
    print(f"Output Topic     : {OUTPUT_TOPIC}")
    print(f"Consumer Group   : {CONSUMER_GROUP}")
    print("======================================================================")

    model, feature_cols = load_model_and_metadata()
    engine = StreamingFeatureEngine(window_size_seconds=FEATURE_WINDOW_SECONDS)

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

            # Extract timestamp in seconds
            raw_ts = trade_data.get("event_time", trade_data.get("timestamp"))
            if raw_ts is None:
                timestamp = recv_time
            else:
                raw_ts = float(raw_ts)
                timestamp = raw_ts / 1000.0 if raw_ts > 1e11 else raw_ts

            # Update feature engine deque
            features = engine.update_and_get_features(
                current_price=price,
                current_volume=quantity,
                current_ts=timestamp,
                is_buyer_maker=is_buyer_maker
            )

            # Need at least 2 ticks to calculate rolling differences
            if features is None:
                if tick_count % 10 == 0:
                    print(f"[WARMUP] Buffering initial trade ticks ({len(engine.history)}/2)...")
                continue

            # Build feature vector in exact order expected by XGBoost
            try:
                feature_row = [float(features[col]) for col in feature_cols]
                X_live = np.array([feature_row], dtype=np.float32)
            except KeyError as ke:
                print(f"[ERR] Missing required feature: {ke}")
                continue

            # Run inference
            t_infer_start = time.perf_counter()
            prob_up = float(model.predict_proba(X_live)[0, 1])
            latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

            # Binary prediction: 1 = UP, 0 = DOWN/FLAT
            binary_pred = 1 if prob_up >= 0.50 else 0

            # Quant Deadband Signal
            if prob_up >= DEADBAND_UP_THRESHOLD:
                direction_label = "UP"
                confidence = prob_up
            elif prob_up <= DEADBAND_DOWN_THRESHOLD:
                direction_label = "DOWN"
                confidence = 1.0 - prob_up
            else:
                direction_label = "NEUTRAL"
                confidence = max(prob_up, 1.0 - prob_up)

            pred_count += 1

            # Build outgoing prediction message
            prediction_payload = {
                "timestamp": timestamp,
                "price": price,
                "quantity": quantity,
                "binary_prediction": binary_pred,
                "direction": direction_label,
                "prob_up": round(prob_up, 4),
                "prob_down": round(1.0 - prob_up, 4),
                "confidence": round(confidence, 4),
                "latency_ms": round(latency_ms, 2),
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

            # Terminal Display
            color_sig = (
                "\033[92m▲ UP\033[0m" if direction_label == "UP"
                else ("\033[91m▼ DOWN\033[0m" if direction_label == "DOWN"
                else "\033[93m━ NEUTRAL\033[0m")
            )

            print(
                f"[PRED #{pred_count:05d}] Price: ${price:>9.2f} | "
                f"Signal: {color_sig:<18} | "
                f"P(UP): {prob_up*100:5.1f}% | "
                f"Conf: {confidence*100:5.1f}% | "
                f"Imbal(60s): {features.get('volume_imbalance_60s', 0):+0.2f} | "
                f"Latency: {latency_ms:4.2f}ms"
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
