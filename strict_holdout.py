import sqlite3
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


DB_PATH = "historical_features.db"

# Strict holdout boundary
HOLDOUT_START = 1790950000

TARGET_HORIZON_SECONDS = 30
MIN_FUTURE_GAP_SECONDS = 20
MAX_FUTURE_GAP_SECONDS = 40
MAX_CONTINUOUS_GAP_SECONDS = 60


TOP_MICRO_10 = [
    "sell_volume_60s",
    "buy_volume_60s",
    "buy_volume_30s",
    "buy_sell_ratio_30s",
    "sell_trade_count_30s",
    "buy_sell_ratio_60s",
    "sell_volume_30s",
    "buy_trade_count_60s",
    "sell_trade_count_60s",
    "volume_imbalance_30s",
]


def load_data():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    query = """
    SELECT *
    FROM crypto_features
    WHERE timestamp >= ?
    ORDER BY timestamp
    """

    train = pd.read_sql_query(
        query,
        conn,
        params=(0,)
    )

    conn.close()

    return train


def make_segments(df):
    df = df.sort_values("timestamp").reset_index(drop=True)

    gaps = df["timestamp"].diff()

    segment_id = (gaps > MAX_CONTINUOUS_GAP_SECONDS).cumsum()

    df["segment_id"] = segment_id

    return df


def create_labels(df):
    labeled_parts = []

    for _, segment in df.groupby("segment_id"):
        segment = segment.sort_values("timestamp").reset_index(drop=True)

        timestamps = segment["timestamp"].to_numpy()
        prices = segment["price"].to_numpy()

        target_times = timestamps + TARGET_HORIZON_SECONDS

        future_indices = np.searchsorted(
            timestamps,
            target_times,
            side="left"
        )

        valid_rows = []

        for i, j in enumerate(future_indices):

            if j >= len(segment):
                continue

            future_gap = timestamps[j] - timestamps[i]

            if not (
                MIN_FUTURE_GAP_SECONDS
                <= future_gap
                <= MAX_FUTURE_GAP_SECONDS
            ):
                continue

            future_return = (
                prices[j] - prices[i]
            ) / prices[i]

            if future_return < -0.0002:
                target = 0       # DOWN

            elif future_return > 0.0002:
                target = 2       # UP

            else:
                target = 1       # FLAT

            valid_rows.append((i, target))

        if not valid_rows:
            continue

        indices = [x[0] for x in valid_rows]
        targets = [x[1] for x in valid_rows]

        part = segment.iloc[indices].copy()
        part["target"] = targets

        labeled_parts.append(part)

    if not labeled_parts:
        return pd.DataFrame()

    return pd.concat(
        labeled_parts,
        ignore_index=True
    )


def train_xgb(X_train, y_train):

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4
    )

    model.fit(X_train, y_train)

    return model


print("=" * 70)
print("STRICT UNSEEN HOLDOUT VALIDATION")
print("=" * 70)

print("\nLoading database...")

df = load_data()

print("Total raw rows loaded:", len(df))


# ---------------------------------------------------------
# Split BEFORE creating labels
# ---------------------------------------------------------

train_raw = df[df["timestamp"] < HOLDOUT_START].copy()

holdout_raw = df[df["timestamp"] >= HOLDOUT_START].copy()

print("\nRAW DATA SPLIT")
print("-" * 70)

print("Training raw rows :", len(train_raw))
print("Holdout raw rows  :", len(holdout_raw))

print(
    "Training range:",
    train_raw["timestamp"].min(),
    "->",
    train_raw["timestamp"].max()
)

print(
    "Holdout range:",
    holdout_raw["timestamp"].min(),
    "->",
    holdout_raw["timestamp"].max()
)


# ---------------------------------------------------------
# Create continuous segments separately
# ---------------------------------------------------------

train_raw = make_segments(train_raw)
holdout_raw = make_segments(holdout_raw)


# ---------------------------------------------------------
# Create labels
# ---------------------------------------------------------

print("\nCreating training labels...")

train = create_labels(train_raw)

print("Training labeled rows:", len(train))


print("\nCreating holdout labels...")

holdout = create_labels(holdout_raw)

print("Holdout labeled rows:", len(holdout))


# ---------------------------------------------------------
# Remove incomplete feature rows
# ---------------------------------------------------------

train = train.dropna(
    subset=TOP_MICRO_10 + ["target"]
).reset_index(drop=True)

holdout = holdout.dropna(
    subset=TOP_MICRO_10 + ["target"]
).reset_index(drop=True)


print("\nAFTER FEATURE CLEANING")
print("-" * 70)

print("Training rows:", len(train))
print("Holdout rows :", len(holdout))


# ---------------------------------------------------------
# Target distribution
# ---------------------------------------------------------

print("\nTRAINING TARGET DISTRIBUTION")

print(
    train["target"]
    .value_counts()
    .sort_index()
)

print("\nHOLDOUT TARGET DISTRIBUTION")

print(
    holdout["target"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# Prepare X and y
# ---------------------------------------------------------

X_train = train[TOP_MICRO_10]
y_train = train["target"]

X_holdout = holdout[TOP_MICRO_10]
y_holdout = holdout["target"]


# ---------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------

print("\nTraining XGBoost...")

model = train_xgb(
    X_train,
    y_train
)


# ---------------------------------------------------------
# Predict holdout
# ---------------------------------------------------------

print("Predicting unseen holdout...")

pred = model.predict(X_holdout)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_holdout,
    pred
)

macro_f1 = f1_score(
    y_holdout,
    pred,
    average="macro"
)


print("\n" + "=" * 70)
print("STRICT HOLDOUT RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")


# ---------------------------------------------------------
# Per-class F1
# ---------------------------------------------------------

class_f1 = f1_score(
    y_holdout,
    pred,
    average=None,
    labels=[0, 1, 2]
)

print("\nPER-CLASS F1")

print(f"DOWN : {class_f1[0]:.4f}")
print(f"FLAT : {class_f1[1]:.4f}")
print(f"UP   : {class_f1[2]:.4f}")


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_holdout,
    pred,
    labels=[0, 1, 2]
)

print("\nCONFUSION MATRIX")
print("Rows = Actual")
print("Columns = Predicted")
print()
print("          DOWN  FLAT  UP")
print(f"DOWN      {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}")
print(f"FLAT      {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}")
print(f"UP        {cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)