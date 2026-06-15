"""
==========================================================================
 Real-Time Stock Market Intelligence Platform - Phase 2
 Script 2: ml_model.py

 What this script does:
   1. Loads historical OHLCV data, technical indicators, and daily news
      sentiment from the database (built in Phase 1 / news_sentiment.py).
   2. Engineers features:
        - Lagged close prices & returns (t-1, t-2, t-3, t-5)
        - Rolling statistics (rolling mean/std of returns & volume)
        - Existing technical indicators (SMA, RSI, MACD, Bollinger Bands)
        - Daily aggregated news sentiment score (merged by date)
   3. Builds the target label: next-day price direction
        - 1 = next day's close > today's close (price goes UP)
        - 0 = next day's close <= today's close (price goes DOWN)
   4. Trains an XGBoost classifier (one model across all 10 stocks,
      with `symbol` as a categorical feature).
   5. Evaluates with accuracy, precision, recall, F1 on a held-out
      time-based test split (no shuffling -- avoids lookahead bias).
   6. Logs parameters, metrics, and the trained model to MLflow.
   7. Saves the trained model + feature list + label encoder to disk
      using joblib.
   8. Provides a predict(ticker) function that:
        - Loads the latest feature row for that ticker from the DB
        - Loads the saved model
        - Returns {"prediction": "Up"/"Down", "confidence": float}

 Run:
    python ml_model.py train             -> train + evaluate + log to MLflow
    python ml_model.py predict TCS.NS    -> predict next-day direction
==========================================================================
"""

import os
import sys
import logging
import argparse
from logging.handlers import RotatingFileHandler
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb
import mlflow
import mlflow.xgboost


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

# --- Database configuration (mirrors Phase 1 / news_sentiment.py) ---
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
SQLITE_PATH = os.getenv("SQLITE_PATH", "stocks.db")

if all([PG_HOST, PG_DB, PG_USER, PG_PASSWORD]):
    DATABASE_URL = (
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
        f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )
else:
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"

# --- Model / artifact paths ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_direction_model.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "symbol_encoder.joblib")

# --- MLflow configuration ---
MLFLOW_EXPERIMENT_NAME = "stock_direction_prediction"
# By default MLflow logs to a local ./mlruns directory. To use a remote
# tracking server, set MLFLOW_TRACKING_URI env var before running.

# --- Train/test split configuration ---
TEST_SIZE = 0.2  # last 20% of each stock's timeline (time-based, not random)

# --- XGBoost hyperparameters ---
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
}


# ==========================================================================
# 2. LOGGING SETUP
# ==========================================================================

logger = logging.getLogger("ml_model")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = RotatingFileHandler(
    "ml_model.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ==========================================================================
# 3. DATABASE ACCESS
# ==========================================================================

def get_engine():
    """Create and return a SQLAlchemy engine for the configured database."""
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect():
            pass
        logger.info("Connected to database.")
        return engine
    except Exception:
        logger.exception("Failed to connect to database.")
        raise


def load_raw_data(engine) -> pd.DataFrame:
    """
    Load and merge data from the three Phase 1/2 tables:
        - stock_prices      (OHLCV)
        - stock_indicators  (SMA, RSI, MACD, Bollinger Bands)
        - news_sentiment_daily (avg sentiment score per day, if available)

    Returns one merged DataFrame, one row per (symbol, date).
    Missing sentiment data is filled with 0 (neutral) so the model
    still works even before any news data has been collected.
    """
    logger.info("Loading price data from 'stock_prices'...")
    prices = pd.read_sql(
        "SELECT symbol, name, date, open, high, low, close, volume "
        "FROM stock_prices ORDER BY symbol, date",
        engine,
        parse_dates=["date"],
    )

    logger.info("Loading technical indicators from 'stock_indicators'...")
    indicators = pd.read_sql(
        "SELECT symbol, date, sma_20, sma_50, rsi_14, macd, macd_signal, "
        "macd_hist, bb_upper, bb_middle, bb_lower "
        "FROM stock_indicators ORDER BY symbol, date",
        engine,
        parse_dates=["date"],
    )

    # Merge prices + indicators on (symbol, date)
    df = pd.merge(prices, indicators, on=["symbol", "date"], how="left")

    # --- News sentiment is optional: table may not exist yet ---
    try:
        logger.info("Loading daily sentiment from 'news_sentiment_daily'...")
        sentiment = pd.read_sql(
            "SELECT symbol, date, avg_sentiment_score "
            "FROM news_sentiment_daily ORDER BY symbol, date",
            engine,
            parse_dates=["date"],
        )
        # Normalize date dtype to match `df['date']` (both should be midnight timestamps)
        sentiment["date"] = pd.to_datetime(sentiment["date"])
        df = pd.merge(df, sentiment, on=["symbol", "date"], how="left")
    except Exception:
        logger.warning(
            "Could not load 'news_sentiment_daily' table (it may not exist yet). "
            "Proceeding without sentiment data."
        )
        df["avg_sentiment_score"] = np.nan

    # Fill missing sentiment with 0 (neutral) -- common for older historical
    # dates before the news pipeline was running.
    df["avg_sentiment_score"] = df["avg_sentiment_score"].fillna(0.0)

    logger.info(f"Loaded merged dataset with {len(df)} rows across {df['symbol'].nunique()} stocks.")
    return df


# ==========================================================================
# 4. FEATURE ENGINEERING
# ==========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the merged raw dataset (one row per symbol/date), engineer
    additional features and the prediction target.

    Features added:
        - daily_return        : % change in close vs previous day
        - lag_return_1/2/3/5  : daily_return shifted back N days
        - lag_close_1/2/3     : close price shifted back N days
        - roll_mean_5/10      : rolling mean of daily_return
        - roll_std_5/10       : rolling std of daily_return (volatility)
        - volume_change       : % change in volume vs previous day
        - roll_vol_mean_5     : rolling mean of volume

    Target added:
        - target_direction : 1 if next day's close > today's close, else 0

    IMPORTANT: All rolling/lag operations are computed PER STOCK (groupby
    symbol) and sorted by date, so we never leak data across stocks or
    look into the future when computing features for "today".
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = df.groupby("symbol", group_keys=False)

    # --- Daily return (used as the base for lag/rolling features) ---
    df["daily_return"] = grouped["close"].pct_change()

    # --- Lagged returns (yesterday's, 2 days ago, etc.) ---
    for lag in [1, 2, 3, 5]:
        df[f"lag_return_{lag}"] = grouped["daily_return"].shift(lag)

    # --- Lagged close prices ---
    for lag in [1, 2, 3]:
        df[f"lag_close_{lag}"] = grouped["close"].shift(lag)

    # --- Rolling statistics on returns (volatility / momentum) ---
    df["roll_mean_5"] = grouped["daily_return"].transform(lambda s: s.rolling(5).mean())
    df["roll_std_5"] = grouped["daily_return"].transform(lambda s: s.rolling(5).std())
    df["roll_mean_10"] = grouped["daily_return"].transform(lambda s: s.rolling(10).mean())
    df["roll_std_10"] = grouped["daily_return"].transform(lambda s: s.rolling(10).std())

    # --- Volume features ---
    df["volume_change"] = grouped["volume"].pct_change()
    df["roll_vol_mean_5"] = grouped["volume"].transform(lambda s: s.rolling(5).mean())

    # --- Price relative to indicators (normalized, scale-free features) ---
    # e.g. how far is close from its 20-day SMA, as a percentage
    df["close_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["close_vs_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]
    df["close_vs_bb_upper"] = (df["close"] - df["bb_upper"]) / df["bb_upper"]
    df["close_vs_bb_lower"] = (df["close"] - df["bb_lower"]) / df["bb_lower"]

    # --- TARGET: next-day price direction ---
    # shift(-1) looks at TOMORROW's close relative to TODAY's close.
    # This is the value we want to predict using TODAY's features.
    df["next_close"] = grouped["close"].shift(-1)
    df["target_direction"] = (df["next_close"] > df["close"]).astype(int)

    # The last row per stock has no "next_close" (no future data yet),
    # so its target is meaningless -- we'll drop it before training but
    # keep it for the predict() use case (it's the row we want to predict on).
    df["is_latest_row"] = grouped["date"].transform(lambda s: s == s.max())

    return df


# Feature columns used by the model.
# 'symbol_encoded' is added separately after label encoding.
FEATURE_COLUMNS = [
    "sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower",
    "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_5",
    "lag_close_1", "lag_close_2", "lag_close_3",
    "roll_mean_5", "roll_std_5", "roll_mean_10", "roll_std_10",
    "volume_change", "roll_vol_mean_5",
    "close_vs_sma20", "close_vs_sma50", "close_vs_bb_upper", "close_vs_bb_lower",
    "avg_sentiment_score",
    "symbol_encoded",
]


# ==========================================================================
# 5. TRAINING
# ==========================================================================

def train_model():
    """
    Full training pipeline:
        1. Load + merge data from DB
        2. Engineer features & target
        3. Drop rows with NaNs (from rolling/lag warmup periods, and the
           final row per stock which has no target yet)
        4. Encode the 'symbol' column numerically
        5. Time-based train/test split (per stock, last TEST_SIZE% as test)
        6. Train XGBoost classifier
        7. Evaluate (accuracy, precision, recall, F1)
        8. Log everything to MLflow
        9. Save model + feature list + encoder with joblib
    """
    logger.info("=" * 70)
    logger.info("Starting model training pipeline...")

    engine = get_engine()
    raw_df = load_raw_data(engine)

    if raw_df.empty:
        logger.error("No data loaded from database. Cannot train. Aborting.")
        return

    df = engineer_features(raw_df)

    # Keep a copy of the very latest row per stock for later (predict() use)
    # BEFORE we drop rows with NaN targets.
    latest_rows = df[df["is_latest_row"]].copy()

    # --- Drop rows that can't be used for training ---
    # (rows with NaN in any feature column, or NaN target)
    df[FEATURE_COLUMNS[:-1]] = df[FEATURE_COLUMNS[:-1]].replace([np.inf, -np.inf], np.nan)
    model_df = df.dropna(subset=FEATURE_COLUMNS[:-1] + ["target_direction"]).copy()

    if model_df.empty:
        logger.error("No usable rows after dropping NaNs. Need more historical data. Aborting.")
        return

    logger.info(f"Usable rows for training/testing: {len(model_df)}")

    # --- Encode 'symbol' as a numeric feature ---
    encoder = LabelEncoder()
    model_df["symbol_encoded"] = encoder.fit_transform(model_df["symbol"])

    # Also encode the latest_rows using the SAME encoder (for predict() later)
    # Some symbols might not appear in model_df if they had too little data,
    # but typically all 10 will be present.
    latest_rows = latest_rows[latest_rows["symbol"].isin(encoder.classes_)].copy()
    latest_rows["symbol_encoded"] = encoder.transform(latest_rows["symbol"])

    # --- Time-based train/test split (per stock) ---
    # We sort by date and take the last TEST_SIZE fraction of each stock's
    # rows as the test set. This avoids shuffling time series data, which
    # would leak future information into training.
    train_parts, test_parts = [], []
    for symbol, group in model_df.groupby("symbol"):
        group = group.sort_values("date")
        split_idx = int(len(group) * (1 - TEST_SIZE))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_direction"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_direction"]

    logger.info(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    logger.info(f"Train target balance:\n{y_train.value_counts(normalize=True)}")

    # --- Train XGBoost classifier ---
    logger.info("Training XGBoost classifier...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    logger.info("Evaluation metrics on held-out test set:")
    for k, v in metrics.items():
        logger.info(f"  {k:>10}: {v:.4f}")

    # --- Log to MLflow ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"xgb_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log hyperparameters
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Log the trained model itself (MLflow's native XGBoost flavor)
        mlflow.xgboost.log_model(model, artifact_path="model")

        logger.info(f"MLflow run logged under experiment '{MLFLOW_EXPERIMENT_NAME}'.")

    # --- Save model + supporting artifacts with joblib ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_LIST_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    # Also save the latest feature row per stock -- this lets predict()
    # work without re-querying/re-engineering features from scratch.
    latest_path = os.path.join(MODEL_DIR, "latest_features.parquet")
    latest_rows[["symbol", "date"] + FEATURE_COLUMNS].to_parquet(latest_path, index=False)

    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Feature list saved to {FEATURE_LIST_PATH}")
    logger.info(f"Symbol encoder saved to {ENCODER_PATH}")
    logger.info(f"Latest feature rows saved to {latest_path}")
    logger.info("Training pipeline complete.")
    logger.info("=" * 70)


# ==========================================================================
# 6. PREDICTION
# ==========================================================================

def predict(ticker: str) -> dict:
    """
    Predict the next-day price direction for a given stock ticker
    (e.g. 'TCS.NS').

    Loads:
        - the saved XGBoost model
        - the saved feature column list
        - the saved symbol encoder
        - the saved "latest feature row" for this ticker (computed during
          the most recent training run)

    Returns:
        {
            "symbol": "TCS.NS",
            "prediction": "Up" | "Down",
            "confidence": float (0-1, probability of the predicted class),
            "as_of_date": "YYYY-MM-DD"  (the date of the feature row used)
        }

    Raises FileNotFoundError if the model hasn't been trained yet, and
    ValueError if the ticker is unknown or has no available feature row.
    """
    # --- Load saved artifacts ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            f"Run 'python ml_model.py train' first."
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_LIST_PATH)
    encoder = joblib.load(ENCODER_PATH)

    latest_path = os.path.join(MODEL_DIR, "latest_features.parquet")
    if not os.path.exists(latest_path):
        raise FileNotFoundError(
            f"Latest feature snapshot not found at '{latest_path}'. "
            f"Run 'python ml_model.py train' first."
        )

    latest_df = pd.read_parquet(latest_path)

    # --- Validate ticker ---
    if ticker not in encoder.classes_:
        raise ValueError(
            f"Unknown ticker '{ticker}'. Known tickers: {list(encoder.classes_)}"
        )

    row = latest_df[latest_df["symbol"] == ticker]
    if row.empty:
        raise ValueError(
            f"No feature row available for ticker '{ticker}'. "
            f"It may not have enough historical data. Re-run training "
            f"after collecting more data."
        )

    row = row.iloc[[-1]]  # take the single (most recent) row

    # --- Prepare the feature vector ---
    X = row[feature_columns]

    # Safety check: model expects no NaNs
    if X.isnull().any(axis=None):
        raise ValueError(
            f"Latest feature row for '{ticker}' contains missing values "
            f"(likely insufficient history for rolling/lag features)."
        )

    # --- Predict ---
    pred_class = model.predict(X)[0]                 # 0 = Down, 1 = Up
    pred_proba = model.predict_proba(X)[0]           # [P(Down), P(Up)]
    confidence = float(pred_proba[pred_class])

    result = {
        "symbol": ticker,
        "prediction": "Up" if pred_class == 1 else "Down",
        "confidence": round(confidence, 4),
        "as_of_date": str(row["date"].iloc[0].date()),
    }

    return result


# ==========================================================================
# 7. ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model for next-day stock direction prediction, "
                     "or predict using a previously trained model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `train` subcommand
    subparsers.add_parser("train", help="Train the model and log to MLflow.")

    # `predict` subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict next-day direction for a ticker.")
    predict_parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker as stored in the DB, e.g. TCS.NS",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model()

    elif args.command == "predict":
        try:
            result = predict(args.ticker)
            logger.info(f"Prediction result: {result}")
            print(result)
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            print(f"Error: {e}")
            sys.exit(1)
