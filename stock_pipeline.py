"""
==========================================================================
 Real-Time Stock Market Intelligence Platform - Phase 1
 Data Pipeline + Database

 What this script does:
   1. Fetches 1 year of historical OHLCV data + latest real-time price
      for 10 popular Indian stocks using yfinance.
   2. Calculates technical indicators:
        - 20-day & 50-day Simple Moving Averages (SMA)
        - RSI (Relative Strength Index, 14-day)
        - MACD (12, 26, 9)
        - Bollinger Bands (20-day, 2 std dev)
   3. Stores raw OHLCV + indicators in a database.
        - Uses PostgreSQL if env vars are configured.
        - Falls back to a local SQLite file (stocks.db) otherwise.
   4. Uses APScheduler to automatically refresh data every 30 minutes.
   5. Logs everything to console + a rotating log file.

 Run:
    python stock_pipeline.py            -> runs once immediately, then
                                            keeps running and refreshes
                                            every 30 minutes.
    python stock_pipeline.py --once      -> runs a single fetch & exit
                                            (useful for testing / cron).
==========================================================================
"""

import os
import sys
import logging
import argparse
from logging.handlers import RotatingFileHandler
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    BigInteger, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

# --- Stock universe: 10 popular Indian stocks (NSE tickers for yfinance) ---
# yfinance requires the ".NS" suffix for NSE-listed stocks.
STOCKS = {
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFC_BANK": "HDFCBANK.NS",
    "WIPRO": "WIPRO.NS",
    "ICICI_BANK": "ICICIBANK.NS",
    "HCL_TECH": "HCLTECH.NS",
    "BAJAJ_FINANCE": "BAJFINANCE.NS",
    "ASIAN_PAINTS": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
}

# --- How much historical data to pull ---
HISTORY_PERIOD = "1y"      # 1 year of historical data
HISTORY_INTERVAL = "1d"    # daily candles

# --- Scheduler refresh interval (in minutes) ---
REFRESH_INTERVAL_MINUTES = 30

# --- Database configuration ---
# If these PostgreSQL env vars are all set, we use PostgreSQL.
# Otherwise we automatically fall back to a local SQLite file.
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")

SQLITE_PATH = os.getenv("SQLITE_PATH", "stocks.db")

# Decide which DB URL to use
if all([PG_HOST, PG_DB, PG_USER, PG_PASSWORD]):
    DATABASE_URL = (
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
        f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )
    DB_KIND = "postgresql"
else:
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
    DB_KIND = "sqlite"


# ==========================================================================
# 2. LOGGING SETUP
# ==========================================================================
# Logs go to both the console (for live monitoring) and a rotating file
# (so logs don't grow forever).

LOG_FILE = "stock_pipeline.log"

logger = logging.getLogger("stock_pipeline")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Rotating file handler: max 5MB per file, keep 3 backups
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ==========================================================================
# 3. DATABASE MODELS (SQLAlchemy ORM)
# ==========================================================================
# We define two tables:
#   - stock_prices : raw daily OHLCV data
#   - stock_indicators : calculated technical indicators per day
#
# Both tables have a UNIQUE constraint on (symbol, date) so that
# re-running the pipeline does NOT create duplicate rows -- instead
# we UPSERT (insert or update) existing rows.

Base = declarative_base()


class StockPrice(Base):
    """Raw OHLCV (Open, High, Low, Close, Volume) data per stock per day."""
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)   # e.g. TCS.NS
    name = Column(String(50), nullable=False)                 # e.g. TCS
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_price_symbol_date"),
    )


class StockIndicator(Base):
    """Technical indicators calculated per stock per day."""
    __tablename__ = "stock_indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    name = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False, index=True)

    sma_20 = Column(Float)        # 20-day Simple Moving Average
    sma_50 = Column(Float)        # 50-day Simple Moving Average

    rsi_14 = Column(Float)        # 14-day Relative Strength Index

    macd = Column(Float)          # MACD line (EMA12 - EMA26)
    macd_signal = Column(Float)   # Signal line (EMA9 of MACD)
    macd_hist = Column(Float)     # MACD histogram (MACD - signal)

    bb_upper = Column(Float)      # Bollinger Band upper
    bb_middle = Column(Float)     # Bollinger Band middle (= SMA20)
    bb_lower = Column(Float)      # Bollinger Band lower

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_ind_symbol_date"),
    )


class LatestQuote(Base):
    """Latest real-time / near-real-time quote snapshot per stock."""
    __tablename__ = "latest_quotes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(50), nullable=False)
    price = Column(Float)
    previous_close = Column(Float)
    day_high = Column(Float)
    day_low = Column(Float)
    volume = Column(BigInteger)
    fetched_at = Column(DateTime, nullable=False)


# ==========================================================================
# 4. DATABASE ENGINE / SESSION SETUP
# ==========================================================================

def get_engine():
    """
    Create and return a SQLAlchemy engine.
    Falls back to SQLite automatically if PostgreSQL connection fails,
    so the script always remains runnable for local development.
    """
    global DATABASE_URL, DB_KIND
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        # Test the connection immediately
        with engine.connect() as conn:
            pass
        logger.info(f"Connected to database ({DB_KIND}): {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
        return engine
    except Exception as e:
        if DB_KIND == "postgresql":
            logger.error(f"PostgreSQL connection failed ({e}). Falling back to SQLite.")
            DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
            DB_KIND = "sqlite"
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            logger.info(f"Connected to fallback SQLite database: {SQLITE_PATH}")
            return engine
        else:
            # SQLite itself failed -- something is seriously wrong (permissions etc.)
            logger.exception("Failed to connect to SQLite database.")
            raise


# ==========================================================================
# 5. DATA FETCHING (yfinance)
# ==========================================================================

def fetch_historical_data(ticker: str, name: str) -> pd.DataFrame:
    """
    Fetch 1 year of daily OHLCV data for a given ticker using yfinance.

    Returns a DataFrame with columns:
        date, open, high, low, close, volume, symbol, name
    Returns an empty DataFrame on failure (and logs the error) so the
    pipeline can continue with other stocks.
    """
    try:
        logger.info(f"Fetching {HISTORY_PERIOD} historical data for {name} ({ticker})...")
        df = yf.Ticker(ticker).history(
            period=HISTORY_PERIOD,
            interval=HISTORY_INTERVAL,
            auto_adjust=False,  # keep raw OHLC, not split/dividend-adjusted
        )

        if df.empty:
            logger.warning(f"No historical data returned for {ticker}. Skipping.")
            return pd.DataFrame()

        # yfinance returns the date as the index -> move it into a column
        df = df.reset_index()

        # Standardize column names (yfinance sometimes returns 'Date' or
        # 'Datetime' depending on interval)
        df.rename(columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }, inplace=True)

        # Keep only the columns we need
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()

        # Drop timezone info if present (DB columns are timezone-naive)
        if pd.api.types.is_datetime64tz_dtype(df["date"]):
            df["date"] = df["date"].dt.tz_localize(None)

        df["symbol"] = ticker
        df["name"] = name

        logger.info(f"Fetched {len(df)} rows of historical data for {name}.")
        return df

    except Exception:
        logger.exception(f"Error fetching historical data for {ticker} ({name}).")
        return pd.DataFrame()


def fetch_latest_quote(ticker: str, name: str) -> dict:
    """
    Fetch the latest real-time (or near-real-time, ~15min delayed for NSE
    via Yahoo Finance) quote for a given ticker.

    Returns a dict ready to be stored in the LatestQuote table, or an
    empty dict on failure.
    """
    try:
        logger.info(f"Fetching latest quote for {name} ({ticker})...")
        tk = yf.Ticker(ticker)

        # fast_info gives quick access to the latest price data
        fast = tk.fast_info

        quote = {
            "symbol": ticker,
            "name": name,
            "price": float(fast.get("lastPrice", np.nan)) if fast.get("lastPrice") is not None else None,
            "previous_close": float(fast.get("previousClose", np.nan)) if fast.get("previousClose") is not None else None,
            "day_high": float(fast.get("dayHigh", np.nan)) if fast.get("dayHigh") is not None else None,
            "day_low": float(fast.get("dayLow", np.nan)) if fast.get("dayLow") is not None else None,
            "volume": int(fast.get("lastVolume", 0)) if fast.get("lastVolume") is not None else None,
            "fetched_at": datetime.now(),
        }

        logger.info(f"Latest price for {name}: {quote['price']}")
        return quote

    except Exception:
        logger.exception(f"Error fetching latest quote for {ticker} ({name}).")
        return {}


# ==========================================================================
# 6. TECHNICAL INDICATOR CALCULATIONS
# ==========================================================================
# All functions below take a DataFrame sorted by date (ascending) with a
# 'close' column, and return Series aligned to the same index.

def calculate_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average over `window` days."""
    return close.rolling(window=window, min_periods=window).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) using Wilder's smoothing method.

    RSI = 100 - (100 / (1 + RS))
    where RS = average gain / average loss over `period` days.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = an EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle the edge case where avg_loss == 0 (RSI should be 100)
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def calculate_macd(close: pd.Series, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence).

    Returns a tuple of three Series: (macd_line, signal_line, histogram)
        macd_line   = EMA(fast) - EMA(slow)
        signal_line = EMA(signal) of macd_line
        histogram   = macd_line - signal_line
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands.

    Returns a tuple of three Series: (upper_band, middle_band, lower_band)
        middle_band = SMA(window)
        upper_band  = middle_band + num_std * rolling_std
        lower_band  = middle_band - num_std * rolling_std
    """
    middle_band = close.rolling(window=window, min_periods=window).mean()
    rolling_std = close.rolling(window=window, min_periods=window).std()

    upper_band = middle_band + (num_std * rolling_std)
    lower_band = middle_band - (num_std * rolling_std)

    return upper_band, middle_band, lower_band


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a historical OHLCV DataFrame (sorted by date ascending),
    compute all technical indicators and return a new DataFrame with
    one row per date containing: symbol, name, date, and all indicators.
    """
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"]

    indicators = pd.DataFrame()
    indicators["symbol"] = df["symbol"]
    indicators["name"] = df["name"]
    indicators["date"] = df["date"]

    # --- Moving Averages ---
    indicators["sma_20"] = calculate_sma(close, 20)
    indicators["sma_50"] = calculate_sma(close, 50)

    # --- RSI ---
    indicators["rsi_14"] = calculate_rsi(close, 14)

    # --- MACD ---
    macd_line, signal_line, hist = calculate_macd(close)
    indicators["macd"] = macd_line
    indicators["macd_signal"] = signal_line
    indicators["macd_hist"] = hist

    # --- Bollinger Bands ---
    upper, middle, lower = calculate_bollinger_bands(close)
    indicators["bb_upper"] = upper
    indicators["bb_middle"] = middle
    indicators["bb_lower"] = lower

    # Replace any NaN/inf with None so they store as SQL NULL
    indicators = indicators.replace([np.inf, -np.inf], np.nan)

    return indicators


# ==========================================================================
# 7. DATABASE WRITE (UPSERT) FUNCTIONS
# ==========================================================================
# We use "upsert" (INSERT ... ON CONFLICT DO UPDATE) so that re-running the
# pipeline updates existing rows instead of creating duplicates.
# SQLAlchemy provides dialect-specific insert() functions for this.

def upsert_dataframe(engine, df: pd.DataFrame, model, conflict_cols, update_cols):
    """
    Generic upsert helper.

    Args:
        engine: SQLAlchemy engine
        df: DataFrame containing the rows to insert/update
        model: the ORM model class (table) to write to
        conflict_cols: list of column names that form the UNIQUE constraint
        update_cols: list of column names to update on conflict
    """
    if df.empty:
        logger.warning(f"No data to write to {model.__tablename__}. Skipping.")
        return

    # Convert NaN -> None so SQL stores NULL instead of NaN
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    table = model.__table__

    # Choose the correct dialect-specific insert() based on DB type
    if DB_KIND == "postgresql":
        insert_stmt = pg_insert(table).values(records)
    else:
        insert_stmt = sqlite_insert(table).values(records)

    # Build the "ON CONFLICT DO UPDATE" clause
    update_dict = {col: getattr(insert_stmt.excluded, col) for col in update_cols}

    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=conflict_cols,
        set_=update_dict,
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)

    logger.info(f"Upserted {len(records)} rows into {model.__tablename__}.")


def upsert_latest_quote(engine, quote: dict):
    """Upsert a single LatestQuote row (one row per symbol, always updated)."""
    if not quote:
        return

    table = LatestQuote.__table__
    update_cols = ["price", "previous_close", "day_high", "day_low", "volume", "fetched_at", "name"]

    if DB_KIND == "postgresql":
        insert_stmt = pg_insert(table).values(**quote)
    else:
        insert_stmt = sqlite_insert(table).values(**quote)

    update_dict = {col: getattr(insert_stmt.excluded, col) for col in update_cols}

    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=["symbol"],
        set_=update_dict,
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)

    logger.info(f"Upserted latest quote for {quote['symbol']}.")


# ==========================================================================
# 8. MAIN PIPELINE JOB
# ==========================================================================

def run_pipeline():
    """
    The main job that:
      1. Connects to the database (creating tables if needed)
      2. Loops through all configured stocks
      3. Fetches historical OHLCV + latest quote
      4. Calculates technical indicators
      5. Upserts everything into the database

    This function is what gets called immediately on startup AND on every
    scheduled run.
    """
    logger.info("=" * 70)
    logger.info("Starting stock data pipeline run...")

    try:
        engine = get_engine()
    except Exception:
        logger.exception("Could not establish any database connection. Aborting this run.")
        return

    # Create tables if they don't exist yet (safe to call every run)
    Base.metadata.create_all(engine)

    success_count = 0
    failure_count = 0

    for name, ticker in STOCKS.items():
        logger.info("-" * 50)
        logger.info(f"Processing {name} ({ticker})")

        try:
            # --- Step 1: Historical OHLCV data ---
            hist_df = fetch_historical_data(ticker, name)
            if hist_df.empty:
                failure_count += 1
                continue

            # Write raw price data to the database
            upsert_dataframe(
                engine,
                hist_df,
                StockPrice,
                conflict_cols=["symbol", "date"],
                update_cols=["open", "high", "low", "close", "volume", "name"],
            )

            # --- Step 2: Calculate technical indicators ---
            indicators_df = calculate_all_indicators(hist_df)

            # Write indicators to the database
            upsert_dataframe(
                engine,
                indicators_df,
                StockIndicator,
                conflict_cols=["symbol", "date"],
                update_cols=[
                    "sma_20", "sma_50", "rsi_14",
                    "macd", "macd_signal", "macd_hist",
                    "bb_upper", "bb_middle", "bb_lower", "name",
                ],
            )

            # --- Step 3: Latest real-time quote ---
            quote = fetch_latest_quote(ticker, name)
            upsert_latest_quote(engine, quote)

            success_count += 1

        except Exception:
            # Catch-all so one bad stock doesn't crash the whole run
            logger.exception(f"Unexpected error while processing {name} ({ticker}).")
            failure_count += 1
            continue

    logger.info("-" * 50)
    logger.info(
        f"Pipeline run complete. Success: {success_count}, "
        f"Failures: {failure_count}, Total: {len(STOCKS)}"
    )
    logger.info("=" * 70)


# ==========================================================================
# 9. SCHEDULER SETUP
# ==========================================================================

def start_scheduler():
    """
    Start an APScheduler BlockingScheduler that runs `run_pipeline()`
    immediately, then every REFRESH_INTERVAL_MINUTES minutes thereafter.

    BlockingScheduler runs in the current thread and blocks forever --
    this is ideal for a standalone "data pipeline service" process.
    """
    scheduler = BlockingScheduler()

    # Schedule the recurring job
    scheduler.add_job(
        run_pipeline,
        trigger=IntervalTrigger(minutes=REFRESH_INTERVAL_MINUTES),
        id="stock_data_refresh",
        name="Refresh stock data and indicators",
        next_run_time=datetime.now(),  # run immediately on startup
        max_instances=1,               # don't allow overlapping runs
        coalesce=True,                 # if a run is missed, just run once
    )

    logger.info(
        f"Scheduler started. Pipeline will run immediately, then every "
        f"{REFRESH_INTERVAL_MINUTES} minutes. Press Ctrl+C to stop."
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user. Exiting cleanly.")


# ==========================================================================
# 10. ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Stock Market Intelligence Platform - Data Pipeline"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the pipeline a single time and exit (no scheduler).",
    )
    args = parser.parse_args()

    if args.once:
        run_pipeline()
    else:
        start_scheduler()
