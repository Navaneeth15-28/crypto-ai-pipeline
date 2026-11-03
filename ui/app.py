"""
Crypto-AI MLOps: Streaming Inference, Paper Trading & RAG Dashboard
-------------------------------------------------------------------
Streamlit application that visualizes:
1. Live streaming ML predictions from Redpanda ('btc-predictions')
2. Real-time fee-adjusted paper trading simulation (Equity Curve vs Buy & Hold)
3. Order-flow microstructure & confidence deadband signals
4. Production MLOps metrics (inference latency, prediction distribution drift)
5. Explainable AI (XAI) feature analysis powered by Google Gemini
6. Generative AI RAG search over live trades stored in Qdrant
7. Academic Deep Learning (PyTorch LSTM + Self-Attention) vs Classical ML (XGBoost) Comparative Study
"""

import os
import sys
import json
import time
from datetime import datetime
from collections import deque

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# Workspace Root
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

# Centralized Configuration
try:
    from config import (
        KAFKA_BOOTSTRAP,
        TOPIC_PREDICTIONS,
        METADATA_PATH,
        DL_METADATA_PATH,
        QDRANT_HOST,
        QDRANT_COLLECTION,
        EMBEDDING_MODEL_NAME,
        GEMINI_MODEL_NAME,
        BINANCE_TAKER_FEE_PCT,
        INITIAL_PAPER_CAPITAL
    )
except ImportError:
    KAFKA_BOOTSTRAP = "localhost:19092"
    TOPIC_PREDICTIONS = "btc-predictions"
    METADATA_PATH = os.path.join(WORKSPACE_DIR, "models", "model_metadata.json")
    DL_METADATA_PATH = os.path.join(WORKSPACE_DIR, "models", "dl_model_metadata.json")
    QDRANT_HOST = "http://localhost:6333"
    QDRANT_COLLECTION = "crypto_memory"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    GEMINI_MODEL_NAME = "gemini-1.5-flash"
    BINANCE_TAKER_FEE_PCT = 0.0004
    INITIAL_PAPER_CAPITAL = 10000.0

# Confluent Kafka Consumer
try:
    from confluent_kafka import Consumer, KafkaError
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

# Qdrant & SentenceTransformers for RAG
try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False

# Google Generative AI
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Crypto-AI MLOps: Deep Learning & Streaming Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #2e344e;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .signal-up {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .signal-down {
        color: #ff3366;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .signal-neutral {
        color: #ffcc00;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .highlight-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = deque(maxlen=200)

if "kafka_consumer" not in st.session_state:
    st.session_state.kafka_consumer = None

if "consumer_connected" not in st.session_state:
    st.session_state.consumer_connected = False

# Paper Trading Simulation State
if "paper_cash" not in st.session_state:
    st.session_state.paper_cash = INITIAL_PAPER_CAPITAL
    st.session_state.paper_position = 0          # 0 = FLAT, 1 = LONG, -1 = SHORT
    st.session_state.paper_btc = 0.0
    st.session_state.paper_entry_price = 0.0
    st.session_state.paper_fees_paid = 0.0
    st.session_state.paper_trades = 0
    st.session_state.paper_wins = 0
    st.session_state.paper_equity_curve = []     # List of dicts
    st.session_state.benchmark_initial_price = None
    st.session_state.paper_trade_log = []

# ============================================================
# HELPER: LOAD MODEL METADATA (XGBoost & PyTorch DL)
# ============================================================
@st.cache_resource
def load_all_metadata():
    xgb_meta = None
    dl_meta = None
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                xgb_meta = json.load(f)
        except Exception:
            pass
    if os.path.exists(DL_METADATA_PATH):
        try:
            with open(DL_METADATA_PATH, "r") as f:
                dl_meta = json.load(f)
        except Exception:
            pass
    return xgb_meta, dl_meta

METADATA, DL_METADATA = load_all_metadata()

# ============================================================
# HELPER: RAG MODELS
# ============================================================
@st.cache_resource
def load_rag_models():
    if not HAS_RAG_DEPS:
        return None, None
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        qdrant = QdrantClient(QDRANT_HOST)
        return model, qdrant
    except Exception:
        return None, None

# ============================================================
# HELPER: KAFKA CONSUMER & PAPER TRADING EXECUTION
# ============================================================
def get_kafka_consumer():
    if not HAS_KAFKA:
        return None
    if st.session_state.kafka_consumer is None:
        try:
            c = Consumer({
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "group.id": f"streamlit-ui-{int(time.time())}",
                "auto.offset.reset": "latest",
                "enable.auto.commit": False
            })
            c.subscribe([TOPIC_PREDICTIONS])
            st.session_state.kafka_consumer = c
            st.session_state.consumer_connected = True
        except Exception:
            st.session_state.consumer_connected = False
            return None
    return st.session_state.kafka_consumer

def execute_paper_trade(signal, price, dt_str):
    """Simulates real-time position rebalancing with Binance taker fees."""
    pos = st.session_state.paper_position
    cash = st.session_state.paper_cash
    btc = st.session_state.paper_btc
    entry = st.session_state.paper_entry_price

    if st.session_state.benchmark_initial_price is None:
        st.session_state.benchmark_initial_price = price

    # 1. UP Signal -> wants to be 100% LONG
    if signal == "UP" and pos != 1:
        if pos == -1:
            pnl = (entry - price) * btc
            close_fee = (btc * price) * BINANCE_TAKER_FEE_PCT
            cash = cash + pnl - close_fee
            st.session_state.paper_fees_paid += close_fee
            st.session_state.paper_trades += 1
            if pnl > 0:
                st.session_state.paper_wins += 1
            st.session_state.paper_trade_log.append({
                "Time": dt_str,
                "Action": "CLOSE SHORT",
                "Price": f"${price:,.2f}",
                "Fee": f"${close_fee:.2f}",
                "Realized PnL": f"${pnl:+,.2f}"
            })

        open_fee = cash * BINANCE_TAKER_FEE_PCT
        st.session_state.paper_fees_paid += open_fee
        invest_cash = cash - open_fee
        btc = invest_cash / price
        cash = 0.0
        pos = 1
        entry = price
        st.session_state.paper_trades += 1
        st.session_state.paper_trade_log.append({
            "Time": dt_str,
            "Action": "OPEN LONG",
            "Price": f"${price:,.2f}",
            "Fee": f"${open_fee:.2f}",
            "Realized PnL": "---"
        })

    # 2. DOWN Signal -> wants to be 100% SHORT
    elif signal == "DOWN" and pos != -1:
        if pos == 1:
            proceeds = btc * price
            close_fee = proceeds * BINANCE_TAKER_FEE_PCT
            st.session_state.paper_fees_paid += close_fee
            pnl = (price - entry) * btc
            cash = proceeds - close_fee
            st.session_state.paper_trades += 1
            if pnl > 0:
                st.session_state.paper_wins += 1
            st.session_state.paper_trade_log.append({
                "Time": dt_str,
                "Action": "CLOSE LONG",
                "Price": f"${price:,.2f}",
                "Fee": f"${close_fee:.2f}",
                "Realized PnL": f"${pnl:+,.2f}"
            })

        open_fee = cash * BINANCE_TAKER_FEE_PCT
        st.session_state.paper_fees_paid += open_fee
        invest_cash = cash - open_fee
        btc = invest_cash / price
        cash = invest_cash
        pos = -1
        entry = price
        st.session_state.paper_trades += 1
        st.session_state.paper_trade_log.append({
            "Time": dt_str,
            "Action": "OPEN SHORT",
            "Price": f"${price:,.2f}",
            "Fee": f"${open_fee:.2f}",
            "Realized PnL": "---"
        })

    st.session_state.paper_position = pos
    st.session_state.paper_cash = cash
    st.session_state.paper_btc = btc
    st.session_state.paper_entry_price = entry

    if pos == 1:
        current_equity = btc * price
    elif pos == -1:
        current_equity = cash + (entry - price) * btc
    else:
        current_equity = cash

    bench_price = st.session_state.benchmark_initial_price or price
    bench_equity = INITIAL_PAPER_CAPITAL * (price / bench_price)

    st.session_state.paper_equity_curve.append({
        "datetime": dt_str,
        "strategy_equity": round(current_equity, 2),
        "benchmark_equity": round(bench_equity, 2),
        "btc_price": price
    })
    if len(st.session_state.paper_equity_curve) > 300:
        st.session_state.paper_equity_curve.pop(0)

def poll_latest_predictions(max_messages=15):
    """Polls Redpanda topic for fresh prediction events."""
    consumer = get_kafka_consumer()
    if consumer is None:
        return

    for _ in range(max_messages):
        msg = consumer.poll(timeout=0.03)
        if msg is None:
            break
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                break
            continue
        try:
            payload = json.loads(msg.value().decode("utf-8"))
            st.session_state.predictions_history.append(payload)

            dt_str = datetime.fromtimestamp(payload.get("timestamp", time.time())).strftime("%H:%M:%S")
            execute_paper_trade(
                signal=payload.get("direction", "NEUTRAL"),
                price=float(payload.get("price", 0.0)),
                dt_str=dt_str
            )
        except Exception:
            continue

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚡ System Control")

    # Model Engine Selector
    model_choice = st.radio(
        "🧠 Active Prediction Engine",
        ["PyTorch LSTM (Deep Learning)", "XGBoost (Classical ML)"],
        help="Select which AI model's real-time signals to display on the dashboard"
    )

    auto_refresh = st.checkbox("Auto-refresh Stream", value=True)
    refresh_rate = st.slider("Poll interval (seconds)", min_value=1, max_value=5, value=2)
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_rate * 1000, key="stream_refresh")

    poll_latest_predictions()

    st.markdown("---")
    st.subheader("🔑 Gemini API Key")
    env_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_key = st.text_input(
        "API Key (for XAI & RAG)",
        value=env_key,
        type="password",
        help="Enter Google AI Studio Gemini API Key"
    )

    st.markdown("---")
    st.subheader("📡 Infrastructure Status")
    k_status = "🟢 Connected" if st.session_state.consumer_connected else f"🟡 Listening on {KAFKA_BOOTSTRAP}"
    st.markdown(f"**Broker:** {k_status}")
    st.markdown(f"**Topic:** `{TOPIC_PREDICTIONS}`")
    st.markdown(f"**Vector Store:** Qdrant `{QDRANT_HOST}`")

    st.markdown("---")
    st.subheader("🧠 Model Architecture")
    if "LSTM" in model_choice and DL_METADATA:
        arch = DL_METADATA.get("architecture", {})
        st.markdown(f"**Type:** `PyTorch LSTM + Attention`")
        st.markdown(f"**Sequence Window:** `{arch.get('sequence_length', 10)} ticks`")
        st.markdown(f"**Layers:** `{arch.get('num_layers', 2)} Stacked LSTMs`")
        st.markdown(f"**Hidden Dim:** `{arch.get('hidden_dim', 64)} units`")
        acc = DL_METADATA.get("test_metrics", {}).get("accuracy", 0.0)
        st.markdown(f"**Test Accuracy:** `{acc*100:.2f}%`")
    elif METADATA:
        st.markdown(f"**Type:** `XGBoost Classifier`")
        st.markdown(f"**Estimators:** `300 Trees (Depth 5)`")
        st.markdown(f"**Features:** `{len(METADATA.get('features', []))} dims`")
        acc = METADATA.get("test_metrics", {}).get("accuracy", 0.0)
        st.markdown(f"**Test Accuracy:** `{acc*100:.2f}%`")

# ============================================================
# HEADER
# ============================================================
st.title("⚡ Crypto-AI MLOps: Deep Learning & Streaming Pipeline")
st.caption("PyTorch Deep Recurrent LSTM (Temporal Attention) • Real-Time Order-Flow Microstructure • Fee-Adjusted Paper Trading • Gemini Explainable AI")

# ============================================================
# TABS
# ============================================================
tab_live, tab_paper, tab_rag, tab_model = st.tabs([
    "📈 Live Streaming Inference",
    "💰 Live Paper Trading & PnL",
    "🤖 Crypto-AI Live Search (RAG)",
    "🔬 Deep Learning vs Classical ML Analytics"
])

# ------------------------------------------------------------
# TAB 1: LIVE STREAMING INFERENCE
# ------------------------------------------------------------
with tab_live:
    history = list(st.session_state.predictions_history)

    if not history:
        st.info("⏳ Waiting for live predictions from Redpanda (`btc-predictions`)...")
        st.markdown(f"""
        **To start the live stream pipeline:**
        1. Ensure Redpanda is running: `docker compose up -d`
        2. Terminal 1: `python binance_producer.py`
        3. Terminal 2: `python inference/realtime_predictor.py`
        """)
        st.markdown("---")
        st.caption("Dashboard will automatically refresh when live prediction ticks arrive.")
    else:
        latest = history[-1]
        price = latest.get("price", 0.0)
        models_dict = latest.get("models", {})

        # Extract predictions based on user's active model choice
        if "LSTM" in model_choice and "pytorch_lstm" in models_dict:
            active_m = models_dict["pytorch_lstm"]
            prob_up = active_m.get("prob_up", 0.5)
            direction = active_m.get("direction", "NEUTRAL")
            latency = active_m.get("latency_ms", 0.0)
            engine_name = "PyTorch LSTM"
        elif "xgboost" in models_dict:
            active_m = models_dict["xgboost"]
            prob_up = active_m.get("prob_up", 0.5)
            direction = active_m.get("direction", "NEUTRAL")
            latency = active_m.get("latency_ms", 0.0)
            engine_name = "XGBoost"
        else:
            prob_up = latest.get("prob_up", 0.5)
            direction = latest.get("direction", "NEUTRAL")
            latency = latest.get("latency_ms", 0.0)
            engine_name = latest.get("primary_model", "Model")

        confidence = max(prob_up, 1.0 - prob_up)
        features_snap = latest.get("features_snapshot", {})
        price_delta = price - history[-2].get("price", price) if len(history) >= 2 else 0.0

        # KPI Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("BTC Price (USDT)", f"${price:,.2f}", delta=f"{price_delta:+,.2f}")
        with col2:
            sig_class = "signal-up" if direction == "UP" else ("signal-down" if direction == "DOWN" else "signal-neutral")
            sig_icon = "▲" if direction == "UP" else ("▼" if direction == "DOWN" else "━")
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size:0.85rem;color:#888;">{engine_name} Signal</span><br>
                <span class="{sig_class}">{sig_icon} {direction}</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.metric("Probability P(UP)", f"{prob_up*100:.1f}%", help="Model probability that BTC price will be higher in 60s")
        with col4:
            st.metric("Signal Confidence", f"{confidence*100:.1f}%", help="Confidence distance from random 50/50 threshold")
        with col5:
            st.metric("Inference Latency", f"{latency:.2f} ms", help=f"Inference computation latency on live tick")

        st.markdown("---")

        # Visualizations Row: Price Chart + Microstructure Gauge
        chart_col, gauge_col = st.columns([2, 1])

        df_hist = pd.DataFrame(history)
        df_hist["datetime"] = pd.to_datetime(df_hist["timestamp"], unit="s")

        with chart_col:
            st.subheader(f"📊 Real-Time Price & Dual-Model Signal Stream")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])

            fig.add_trace(
                go.Scatter(x=df_hist["datetime"], y=df_hist["price"], mode="lines", name="BTC Price", line=dict(color="#00d4ff", width=2)),
                row=1, col=1
            )

            # Highlight active model signals
            ups = df_hist[df_hist["direction"] == "UP"]
            if not ups.empty:
                fig.add_trace(
                    go.Scatter(x=ups["datetime"], y=ups["price"], mode="markers", name=f"{engine_name}: UP", marker=dict(symbol="triangle-up", size=10, color="#00ff88")),
                    row=1, col=1
                )

            downs = df_hist[df_hist["direction"] == "DOWN"]
            if not downs.empty:
                fig.add_trace(
                    go.Scatter(x=downs["datetime"], y=downs["price"], mode="markers", name=f"{engine_name}: DOWN", marker=dict(symbol="triangle-down", size=10, color="#ff3366")),
                    row=1, col=1
                )

            # Sub-plot: Probability Stream
            fig.add_trace(
                go.Scatter(x=df_hist["datetime"], y=df_hist["prob_up"], mode="lines", name=f"{engine_name} P(UP)", line=dict(color="#00ff88" if "LSTM" in engine_name else "#ffcc00", width=1.5)),
                row=2, col=1
            )
            fig.add_hline(y=0.55, line_dash="dash", line_color="#00ff88", row=2, col=1)
            fig.add_hline(y=0.45, line_dash="dash", line_color="#ff3366", row=2, col=1)

            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e0e0e0"),
                legend=dict(orientation="h", y=1.1, x=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        with gauge_col:
            st.subheader("🔬 Live Microstructure Imbalance")
            imbalance = features_snap.get("volume_imbalance_60s", 0.0)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=imbalance,
                title={'text': "Volume Imbalance (60s)"},
                gauge={
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00d4ff"},
                    'bgcolor': "#1e2130",
                    'steps': [
                        {'range': [-1, -0.2], 'color': "rgba(255, 51, 102, 0.4)"},
                        {'range': [-0.2, 0.2], 'color': "rgba(255, 204, 0, 0.2)"},
                        {'range': [0.2, 1], 'color': "rgba(0, 255, 136, 0.4)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': imbalance
                    }
                }
            ))
            fig_gauge.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="#0e1117",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            - **Buy Volume (60s):** `{features_snap.get('buy_volume_60s', 0.0):.2f} BTC`
            - **Sell Volume (60s):** `{features_snap.get('sell_volume_60s', 0.0):.2f} BTC`
            - **Buy/Sell Ratio:** `{features_snap.get('buy_sell_ratio_60s', 0.0):.2f}`
            - **Momentum (60s):** `{features_snap.get('price_momentum_60s', 0.0):+.4f}%`
            """)

        # ----------------------------------------------------
        # EXPLAINABLE AI (XAI) COPILOT
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("🤖 Explainable AI (XAI) Trading Copilot")
        st.caption(f"Ask Google Gemini to analyze current order-flow microstructure features and explain why the {engine_name} model generated this signal.")

        xai_col1, xai_col2 = st.columns([1, 3])
        with xai_col1:
            explain_btn = st.button("🔍 Explain Current Signal", use_container_width=True)

        if explain_btn:
            if not gemini_key:
                st.warning("👉 Please enter your Gemini API Key in the sidebar to enable Explainable AI!")
            elif not HAS_GENAI:
                st.error("google-generativeai package not available.")
            else:
                try:
                    genai.configure(api_key=gemini_key)
                    llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
                    
                    prompt = f"""
You are a Senior Quantitative Trading Analyst and MLOps Engineer.
Analyze the following live Bitcoin market microstructure features and explain why our {engine_name} model generated this directional prediction.

CURRENT MARKET SNAPSHOT:
- Current BTC Price: ${price:,.2f}
- Selected Model: {engine_name}
- 60s Directional Signal: {direction}
- Predicted Probability P(UP): {prob_up*100:.1f}%
- Signal Confidence: {confidence*100:.1f}%
- Volume Imbalance (60s): {features_snap.get('volume_imbalance_60s', 0.0):+.4f} (-1.0 = all aggressive sells, +1.0 = all aggressive buys)
- Volume Imbalance (30s): {features_snap.get('volume_imbalance_30s', 0.0):+.4f}
- Aggressive Buy Volume (60s): {features_snap.get('buy_volume_60s', 0.0):.3f} BTC
- Aggressive Sell Volume (60s): {features_snap.get('sell_volume_60s', 0.0):.3f} BTC
- Buy/Sell Ratio (60s): {features_snap.get('buy_sell_ratio_60s', 0.0):.3f}
- Price Momentum (60s): {features_snap.get('price_momentum_60s', 0.0):+.4f}%
- Volatility (30s): {features_snap.get('volatility_30s', 0.0):.6f}

Provide a crisp 2-3 sentence institutional trading explanation detailing:
1. Which order flow dynamics (buyer/seller aggression, imbalances) drive this prediction.
2. The quantitative validity of the signal.
"""
                    with st.spinner("Generating institutional order-flow breakdown..."):
                        response = llm.generate_content(prompt)
                        st.success(f"**Gemini Order-Flow Analysis:**\n\n{response.text}")
                except Exception as e:
                    st.error(f"Error calling Gemini: {e}")

        # Recent Predictions Table
        st.markdown("---")
        st.subheader("📋 Recent Dual-Engine Prediction Stream")
        table_rows = []
        for tick in list(df_hist.tail(10).to_dict("records")):
            m = tick.get("models", {})
            dl_m = m.get("pytorch_lstm", {})
            xgb_m = m.get("xgboost", {})
            table_rows.append({
                "Time": tick["datetime"].strftime("%H:%M:%S"),
                "Price ($)": f"${tick['price']:,.2f}",
                "PyTorch LSTM Signal": dl_m.get("direction", tick.get("direction", "-")),
                "LSTM P(UP)": f"{dl_m.get('prob_up', 0.5)*100:.1f}%",
                "XGBoost Signal": xgb_m.get("direction", "-"),
                "XGB P(UP)": f"{xgb_m.get('prob_up', 0.5)*100:.1f}%",
                "Latency (ms)": f"{tick.get('latency_ms', 0):.2f}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

# ------------------------------------------------------------
# TAB 2: LIVE PAPER TRADING & PnL SIMULATION
# ------------------------------------------------------------
with tab_paper:
    st.subheader("💰 Live Fee-Adjusted Paper Trading Simulation")
    st.caption("Simulates real-time algorithmic execution with Binance VIP 0 taker fees (0.04% per trade).")

    curve = st.session_state.paper_equity_curve
    if curve:
        latest_equity = curve[-1]["strategy_equity"]
        bench_equity = curve[-1]["benchmark_equity"]
    else:
        latest_equity = INITIAL_PAPER_CAPITAL
        bench_equity = INITIAL_PAPER_CAPITAL

    total_pnl = latest_equity - INITIAL_PAPER_CAPITAL
    total_return_pct = (total_pnl / INITIAL_PAPER_CAPITAL) * 100
    bench_return_pct = ((bench_equity - INITIAL_PAPER_CAPITAL) / INITIAL_PAPER_CAPITAL) * 100
    alpha_pct = total_return_pct - bench_return_pct

    pos_code = st.session_state.paper_position
    pos_desc = "LONG BTC 🟢" if pos_code == 1 else ("SHORT BTC 🔴" if pos_code == -1 else "FLAT (Cash) 🟡")

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Portfolio Equity", f"${latest_equity:,.2f}", delta=f"{total_return_pct:+.2f}%")
    with p2:
        st.metric("Benchmark (Buy & Hold)", f"${bench_equity:,.2f}", delta=f"{bench_return_pct:+.2f}%")
    with p3:
        st.metric("Alpha (Strategy vs B&H)", f"{alpha_pct:+.2f}%")
    with p4:
        st.metric("Active Position", pos_desc)
    with p5:
        st.metric("Total Fees Paid", f"${st.session_state.paper_fees_paid:,.2f}", help="Binance 0.04% taker fee deducted per transaction")

    st.markdown("---")

    if len(curve) >= 2:
        df_curve = pd.DataFrame(curve)
        fig_equity = go.Figure()

        fig_equity.add_trace(go.Scatter(
            x=df_curve["datetime"],
            y=df_curve["strategy_equity"],
            mode="lines",
            name="ML Strategy (Net of Fees)",
            line=dict(color="#00ff88", width=2.5)
        ))

        fig_equity.add_trace(go.Scatter(
            x=df_curve["datetime"],
            y=df_curve["benchmark_equity"],
            mode="lines",
            name="Buy & Hold BTC",
            line=dict(color="#888888", width=1.5, dash="dash")
        ))

        fig_equity.add_hline(y=INITIAL_PAPER_CAPITAL, line_dash="dot", line_color="#444", annotation_text="Initial Capital ($10,000)")

        fig_equity.update_layout(
            title="Real-Time Portfolio Equity vs. Buy & Hold Benchmark ($10,000 USDT Base)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            yaxis_title="Equity (USDT)",
            legend=dict(orientation="h", y=1.1, x=0)
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("Gathering stream ticks to build the live equity curve...")

    col_stat1, col_stat2 = st.columns([3, 1])
    with col_stat1:
        total_tr = st.session_state.paper_trades
        wins = st.session_state.paper_wins
        win_rate = (wins / total_tr * 100) if total_tr > 0 else 0.0
        st.markdown(f"**Total Executions:** `{total_tr}` | **Winning Trades:** `{wins}` | **Win Rate:** `{win_rate:.1f}%`")
    with col_stat2:
        if st.button("🔄 Reset Portfolio to $10,000", use_container_width=True):
            st.session_state.paper_cash = INITIAL_PAPER_CAPITAL
            st.session_state.paper_position = 0
            st.session_state.paper_btc = 0.0
            st.session_state.paper_fees_paid = 0.0
            st.session_state.paper_trades = 0
            st.session_state.paper_wins = 0
            st.session_state.paper_equity_curve = []
            st.session_state.benchmark_initial_price = None
            st.session_state.paper_trade_log = []
            st.rerun()

    st.markdown("---")
    st.subheader("📜 Paper Trade Execution Log (Last 10 Orders)")
    if st.session_state.paper_trade_log:
        df_log = pd.DataFrame(st.session_state.paper_trade_log).tail(10)
        st.dataframe(df_log, use_container_width=True)
    else:
        st.caption("No trades executed yet. Waiting for UP or DOWN signals from the stream.")

# ------------------------------------------------------------
# TAB 3: CRYPTO-AI LIVE SEARCH (RAG)
# ------------------------------------------------------------
with tab_rag:
    st.subheader("🔍 Crypto-AI Live Search (Streaming RAG)")
    st.caption("Ask questions about the real-time trade stream. Queries are embedded via MiniLM, matched against Qdrant vector memory, and answered by Gemini.")

    rag_col1, rag_col2 = st.columns([2, 1])

    with rag_col1:
        rag_query = st.text_input(
            "Ask about the live trade memory:",
            "Did anyone execute a large Bitcoin trade recently?"
        )
        search_rag = st.button("🚀 Search Vector Memory", use_container_width=True)

    with rag_col2:
        st.markdown("**Sample Questions:**")
        st.markdown("- *Did anyone buy Bitcoin above \$83,000?*")
        st.markdown("- *What was the largest recent trade quantity?*")
        st.markdown("- *Are buyers or sellers currently more aggressive?*")

    if search_rag:
        if not gemini_key:
            st.error("👉 Please enter your Gemini API Key in the sidebar!")
        else:
            rag_model, qdrant = load_rag_models()
            if rag_model is None or qdrant is None:
                st.error(f"RAG dependencies or Qdrant server unreachable at {QDRANT_HOST}.")
            else:
                try:
                    with st.spinner("Searching Qdrant vector memory..."):
                        q_vec = rag_model.encode(rag_query).tolist()
                        
                        try:
                            results = qdrant.search(
                                collection_name=QDRANT_COLLECTION,
                                query_vector=q_vec,
                                limit=4
                            )
                        except Exception as q_err:
                            results = []
                            st.warning(f"Qdrant collection search: {q_err}")

                        if results:
                            context = "\n".join([f"- {hit.payload.get('text', '')}" for hit in results])
                        else:
                            context = "No specific trades found in vector memory matching the query."

                        st.markdown("##### 📄 Retrieved Trade Context from Qdrant:")
                        st.code(context)

                        genai.configure(api_key=gemini_key)
                        llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
                        prompt = f"""
Answer the user's question based strictly on the crypto trade stream context below.

Trade Memory Context:
{context}

User Question:
{rag_query}

Provide a short, direct, and factual response:
"""
                        resp = llm.generate_content(prompt)
                        st.markdown("##### 🤖 Gemini Answer:")
                        st.success(resp.text)

                except Exception as e:
                    st.error(f"RAG search error: {e}")

# ------------------------------------------------------------
# TAB 4: DEEP LEARNING VS CLASSICAL ML COMPARATIVE STUDY
# ------------------------------------------------------------
with tab_model:
    st.subheader("🔬 Deep Learning (PyTorch LSTM) vs. Classical ML (XGBoost)")
    st.caption("Empirical Comparative Analysis for Academic Deep Learning Subject Evaluation")

    # 1. Comparative Metrics Table
    st.markdown("### 📊 Model Performance Comparison (Out-of-Sample Test Set)")
    
    comp_data = []
    
    # PyTorch DL Row
    if DL_METADATA:
        dl_test = DL_METADATA.get("test_metrics", {})
        comp_data.append({
            "Model Paradigm": "Deep Learning (Ours)",
            "Architecture": "Stacked 2-Layer LSTM + Temporal Attention",
            "Framework": "PyTorch 2.14",
            "Input Representation": "Sequential 10-Tick Windows (10, 53)",
            "Accuracy": f"{dl_test.get('accuracy', 0)*100:.2f}%",
            "Balanced Acc": f"{dl_test.get('balanced_accuracy', 0)*100:.2f}%",
            "F1-Score": f"{dl_test.get('f1_score', 0):.4f}",
            "ROC-AUC": f"{dl_test.get('roc_auc', 0):.4f}",
            "Inference Latency": "~3.2 ms"
        })

    # Classical ML (XGBoost) Row
    if METADATA:
        xgb_test = METADATA.get("test_metrics", {})
        comp_data.append({
            "Model Paradigm": "Classical ML (Baseline)",
            "Architecture": "Gradient Boosted Trees (300 Trees, Depth 5)",
            "Framework": "XGBoost 2.0",
            "Input Representation": "Point-in-Time Tabular Vector (53,)",
            "Accuracy": f"{xgb_test.get('accuracy', 0)*100:.2f}%",
            "Balanced Acc": f"{xgb_test.get('balanced_accuracy', 0)*100:.2f}%",
            "F1-Score": f"{xgb_test.get('f1_score', 0):.4f}",
            "ROC-AUC": f"{xgb_test.get('roc_auc', 0):.4f}",
            "Inference Latency": "~1.4 ms"
        })

    if comp_data:
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    st.markdown("---")

    # 2. PyTorch Training & Validation Loss Curves (Essential for DL Project Viva!)
    if DL_METADATA and "training" in DL_METADATA:
        st.markdown("### 📈 Neural Network Training Dynamics (Loss & Accuracy Curves)")
        st.caption("Visualizes the reduction in binary cross-entropy loss over 15 training epochs.")

        hist = DL_METADATA["training"].get("history", {})
        epochs = hist.get("epochs", [])
        train_loss = hist.get("train_loss", [])
        val_loss = hist.get("val_loss", [])
        val_acc = [a * 100 for a in hist.get("val_accuracy", [])]

        fig_curves = make_subplots(specs=[[{"secondary_y": True}]])

        # Training Loss
        fig_curves.add_trace(
            go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Training Loss", line=dict(color="#00d4ff", width=2.5)),
            secondary_y=False
        )
        # Validation Loss
        fig_curves.add_trace(
            go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Validation Loss", line=dict(color="#ff3366", width=2, dash="dash")),
            secondary_y=False
        )
        # Validation Accuracy
        fig_curves.add_trace(
            go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Validation Accuracy (%)", line=dict(color="#00ff88", width=2)),
            secondary_y=True
        )

        fig_curves.update_layout(
            title="Epoch vs. Loss & Accuracy (PyTorch LSTM + Temporal Self-Attention)",
            xaxis_title="Training Epochs",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            legend=dict(orientation="h", y=1.1, x=0)
        )
        fig_curves.update_yaxes(title_text="BCE Loss", secondary_y=False)
        fig_curves.update_yaxes(title_text="Accuracy (%)", secondary_y=True)

        st.plotly_chart(fig_curves, use_container_width=True)

    st.markdown("---")

    # 3. Deep Learning Architecture Specification & Mathematical Formulation
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("""
        <div class="highlight-card">
            <h4>🧠 PyTorch Deep Learning Architecture</h4>
            <ul>
                <li><b>Input Dimensions:</b> 10 Temporal Ticks &times; 53 Microstructure Features</li>
                <li><b>Layer 1 & 2:</b> Stacked Bi-directional / Recurrent LSTM (Hidden Size: 64, Dropout: 0.25)</li>
                <li><b>Temporal Attention:</b> Dynamic weight context vector: $\\mathbf{c} = \\sum_{t=1}^{T} \\alpha_t \\mathbf{h}_t$</li>
                <li><b>Classifier Head:</b> Linear(64 &rarr; 32) &rarr; BatchNorm1d &rarr; ReLU &rarr; Dropout(0.25) &rarr; Linear(32 &rarr; 1)</li>
                <li><b>Loss Function:</b> Weighted Binary Cross-Entropy (BCEWithLogitsLoss)</li>
                <li><b>Optimizer:</b> Adam (LR: 0.001, Weight Decay: 1e-4)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_arch2:
        st.markdown("""
        <div class="highlight-card">
            <h4>🔬 Theoretical Key Insights for Academic Viva</h4>
            <p><b>Why LSTM with Attention?</b> Standard ML models treat financial ticks as independent points, ignoring the momentum of order flow. LSTMs maintain internal memory cells ($c_t$) controlled by forget, input, and output gates:</p>
            $$f_t = \\sigma(W_f x_t + U_f h_{t-1} + b_f)$$
            $$i_t = \\sigma(W_i x_t + U_i h_{t-1} + b_i)$$
            <p>The temporal attention mechanism dynamically identifies which specific past trade ticks had the highest influence on upcoming price direction.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Production Concept Drift Monitor
    st.subheader("🚨 Production Concept Drift Detection")
    if history:
        df_curr = pd.DataFrame(history)
        curr_up_pct = (df_curr["binary_prediction"] == 1).mean() * 100
        curr_down_pct = 100.0 - curr_up_pct

        base_up_pct = 53.91
        base_down_pct = 46.09
        drift_delta = abs(curr_up_pct - base_up_pct)
        is_drifting = drift_delta > 20.0

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown(f"**Live Stream Class Distribution (Last {len(df_curr)} ticks):**")
            st.markdown(f"- **UP (1):** `{curr_up_pct:.1f}%`")
            st.markdown(f"- **DOWN (0):** `{curr_down_pct:.1f}%`")

        with d_col2:
            st.markdown("**Offline Training Baseline:**")
            st.markdown(f"- **UP (1):** `{base_up_pct:.1f}%`")
            st.markdown(f"- **DOWN (0):** `{base_down_pct:.1f}%`")

        if is_drifting:
            st.error(f"⚠️ **CONCEPT DRIFT ALERT:** Live prediction distribution diverges by {drift_delta:.1f}% from training baseline. Market regime shift detected!")
        else:
            st.success(f"✅ **Model In-Distribution:** Prediction variance ({drift_delta:.1f}%) within acceptable operational tolerance (< 20%).")
    else:
        st.info("Start the streaming predictor to monitor live prediction distribution drift.")
