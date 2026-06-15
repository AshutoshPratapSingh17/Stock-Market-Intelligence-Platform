"""
==========================================================================
 Real-Time Stock Market Intelligence Platform - Phase 3
 api.py - FastAPI Backend

 What this application does:
   1. Connects to the database (PostgreSQL if configured, else SQLite),
      reusing the same connection logic as Phases 1 & 2.
   2. Loads the saved XGBoost model + supporting artifacts (encoder,
      feature list, latest feature rows) produced by ml_model.py.
   3. Exposes REST endpoints for:
        - listing stocks
        - latest price + indicators
        - historical OHLCV data
        - latest AI news sentiment
        - ML next-day direction prediction
        - dashboard summary across all stocks
        - creating price alerts (POST)
   4. Uses Pydantic models for request/response validation.
   5. Enables CORS so a Streamlit frontend can call this API.
   6. Protects all endpoints (except /health and docs) with a simple
      API key passed via the `X-API-Key` header.
   7. Exposes Swagger UI at /docs and ReDoc at /redoc.
   8. Returns proper HTTP status codes (404, 401, 422, 500, etc.) with
      clear error messages.

 Environment variables:
   API_KEY            -> the secret key clients must send in X-API-Key
   PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD -> PostgreSQL config (optional)
   SQLITE_PATH        -> SQLite fallback path (default: stocks.db)

 Run:
   uvicorn api:app --reload --host 0.0.0.0 --port 8000

 Then open:
   http://localhost:8000/docs   (Swagger UI)
==========================================================================
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List

import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Depends, Query, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from pydantic import BaseModel, Field

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

# --- Database configuration (same pattern as Phases 1 & 2) ---
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
    DB_KIND = "postgresql"
else:
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
    DB_KIND = "sqlite"

# --- ML model artifact paths (must match ml_model.py) ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_direction_model.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "symbol_encoder.joblib")
LATEST_FEATURES_PATH = os.path.join(MODEL_DIR, "latest_features.parquet")

# --- API key for simple authentication ---
# In production, set this via an environment variable / secrets manager.
# Default value below is ONLY for local development convenience.
API_KEY = os.getenv("API_KEY", "stockplatform-shared-key-2026")

# --- CORS configuration ---
# Allow all origins by default (convenient for local Streamlit dev on
# any port). For production, restrict this to your Streamlit app's URL.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


# ==========================================================================
# 2. LOGGING SETUP
# ==========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")


# ==========================================================================
# 3. DATABASE ENGINE
# ==========================================================================

def create_db_engine() -> Engine:
    """
    Create a SQLAlchemy engine, falling back to SQLite if PostgreSQL is
    configured but unreachable (mirrors Phase 1/2 behaviour).
    """
    global DATABASE_URL, DB_KIND
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect():
            pass
        logger.info(f"Connected to database ({DB_KIND}).")
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
            logger.exception("Failed to connect to SQLite database.")
            raise


# Create the engine once at module load time, reused across requests.
engine: Engine = create_db_engine()


# ==========================================================================
# 4. ML MODEL LOADING
# ==========================================================================
# We load the model + supporting artifacts once at startup. If they don't
# exist (e.g. training hasn't run yet), we log a warning and the /predict
# endpoint will return a 503 Service Unavailable until the model is trained.

ml_model = None
feature_columns: Optional[list] = None
symbol_encoder = None
latest_features_df: Optional[pd.DataFrame] = None


def load_ml_artifacts():
    """Load the XGBoost model and supporting artifacts from disk, if present."""
    global ml_model, feature_columns, symbol_encoder, latest_features_df

    try:
        if not all(os.path.exists(p) for p in [MODEL_PATH, FEATURE_LIST_PATH, ENCODER_PATH, LATEST_FEATURES_PATH]):
            logger.warning(
                "ML model artifacts not found. The /predict endpoint will be "
                "unavailable until 'python ml_model.py train' has been run."
            )
            return

        ml_model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURE_LIST_PATH)
        symbol_encoder = joblib.load(ENCODER_PATH)
        latest_features_df = pd.read_parquet(LATEST_FEATURES_PATH)
        logger.info("ML model and supporting artifacts loaded successfully.")

    except Exception:
        logger.exception("Failed to load ML model artifacts.")
        ml_model = None


load_ml_artifacts()


# ==========================================================================
# 5. AUTHENTICATION (API KEY)
# ==========================================================================
# Clients must send header:  X-API-Key: <API_KEY>
# This is intentionally simple (suitable for a portfolio project). For
# production, consider OAuth2 / JWT and per-user keys.

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: Optional[str] = Depends(api_key_header)):
    """
    Dependency that validates the X-API-Key header.
    Raises 401 if missing or incorrect.
    """
    if provided_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the 'X-API-Key' header.",
        )
    if provided_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return provided_key


# ==========================================================================
# 6. PYDANTIC MODELS (request/response schemas)
# ==========================================================================

class StockInfo(BaseModel):
    """Basic info about a tracked stock, returned by GET /stocks."""
    symbol: str = Field(..., example="TCS.NS")
    name: str = Field(..., example="TCS")


class LatestPriceResponse(BaseModel):
    """Response for GET /stocks/{ticker} - latest price + indicators."""
    symbol: str
    name: str
    date: date
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[int]

    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None


class OHLCVRow(BaseModel):
    """A single day's OHLCV record, used in historical data responses."""
    date: date
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[int]


class HistoryResponse(BaseModel):
    """Response for GET /stocks/{ticker}/history"""
    symbol: str
    name: str
    days_requested: int
    rows_returned: int
    data: List[OHLCVRow]


class SentimentHeadline(BaseModel):
    """A single news headline + its Claude-derived sentiment."""
    headline: str
    source: Optional[str]
    url: Optional[str]
    published_at: datetime
    sentiment_score: Optional[int]
    sentiment_label: Optional[str]
    key_reason: Optional[str]


class SentimentResponse(BaseModel):
    """Response for GET /stocks/{ticker}/sentiment"""
    symbol: str
    name: str
    date: Optional[date] = None
    avg_sentiment_score: Optional[float] = None
    headline_count: Optional[int] = None
    bullish_count: Optional[int] = None
    bearish_count: Optional[int] = None
    neutral_count: Optional[int] = None
    recent_headlines: List[SentimentHeadline] = []


class PredictionResponse(BaseModel):
    """Response for GET /stocks/{ticker}/predict"""
    symbol: str
    prediction: str = Field(..., example="Up")
    confidence: float = Field(..., example=0.7321, description="Probability of the predicted class (0-1)")
    confidence_pct: str = Field(..., example="73.21%")
    as_of_date: date


class DashboardStockSummary(BaseModel):
    """Per-stock row inside the dashboard summary."""
    symbol: str
    name: str
    date: Optional[date] = None
    close: Optional[float] = None
    daily_change_pct: Optional[float] = None
    rsi_14: Optional[float] = None
    avg_sentiment_score: Optional[float] = None
    sentiment_label_today: Optional[str] = None


class DashboardSummaryResponse(BaseModel):
    """Response for GET /dashboard/summary"""
    as_of: datetime
    total_stocks: int
    stocks: List[DashboardStockSummary]


class AlertCreateRequest(BaseModel):
    """Request body for POST /alerts"""
    symbol: str = Field(..., example="TCS.NS", description="Stock ticker as stored in the DB")
    target_price: float = Field(..., gt=0, example=3500.0, description="Price level that triggers the alert")
    condition: str = Field(
        ..., example="above",
        description="Trigger condition: 'above' (price >= target) or 'below' (price <= target)"
    )
    note: Optional[str] = Field(None, example="Sell if TCS crosses 3500", max_length=255)


class AlertResponse(BaseModel):
    """Response after creating an alert."""
    id: int
    symbol: str
    target_price: float
    condition: str
    note: Optional[str]
    created_at: datetime
    is_active: bool


# ==========================================================================
# 7. DATABASE HELPERS
# ==========================================================================
# We use plain SQL via SQLAlchemy's `text()` for simplicity and to avoid
# tightly coupling this API to the ORM models defined in Phases 1 & 2.
# Both PostgreSQL and SQLite support the parameterized queries used here.

# Master list of known stocks. Kept in sync with Phase 1's STOCKS dict.
# Using a static list here is faster than querying DISTINCT every time,
# and guarantees /stocks always returns all 10 even if some have no data yet.
KNOWN_STOCKS = {
    "TCS.NS": "TCS",
    "INFY.NS": "INFOSYS",
    "RELIANCE.NS": "RELIANCE",
    "HDFCBANK.NS": "HDFC_BANK",
    "WIPRO.NS": "WIPRO",
    "ICICIBANK.NS": "ICICI_BANK",
    "HCLTECH.NS": "HCL_TECH",
    "BAJFINANCE.NS": "BAJAJ_FINANCE",
    "ASIANPAINT.NS": "ASIAN_PAINTS",
    "MARUTI.NS": "MARUTI",
}


def validate_ticker(ticker: str) -> str:
    """
    Validate that `ticker` is one of our known stocks.
    Raises HTTP 404 if not. Returns the ticker unchanged (uppercased)
    for convenience.
    """
    ticker = ticker.upper()
    if ticker not in KNOWN_STOCKS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Unknown stock ticker '{ticker}'. "
                f"Valid tickers: {list(KNOWN_STOCKS.keys())}"
            ),
        )
    return ticker


def fetch_latest_price_and_indicators(ticker: str) -> Optional[dict]:
    """
    Fetch the most recent row from stock_prices joined with
    stock_indicators for the given ticker.
    Returns None if no data exists for this ticker yet.
    """
    query = text("""
        SELECT p.symbol, p.name, p.date, p.open, p.high, p.low, p.close, p.volume,
               i.sma_20, i.sma_50, i.rsi_14, i.macd, i.macd_signal, i.macd_hist,
               i.bb_upper, i.bb_middle, i.bb_lower
        FROM stock_prices p
        LEFT JOIN stock_indicators i
               ON p.symbol = i.symbol AND p.date = i.date
        WHERE p.symbol = :symbol
        ORDER BY p.date DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        row = conn.execute(query, {"symbol": ticker}).mappings().fetchone()

    return dict(row) if row else None


def fetch_history(ticker: str, days: int) -> pd.DataFrame:
    """
    Fetch the last `days` rows of OHLCV data for `ticker`, ordered by
    date ascending (oldest first) -- convenient for charting.
    """
    query = text("""
        SELECT date, open, high, low, close, volume
        FROM stock_prices
        WHERE symbol = :symbol
        ORDER BY date DESC
        LIMIT :days
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": ticker, "days": days})

    # Re-sort ascending (oldest -> newest) for charting convenience
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def fetch_latest_sentiment(ticker: str) -> dict:
    """
    Fetch the most recent daily sentiment aggregate AND the most recent
    raw headlines (up to 5) for `ticker`.

    Gracefully handles the case where the sentiment tables don't exist
    yet (Phase 2's news_sentiment.py hasn't been run) by returning empty
    results instead of raising an error.
    """
    result = {
        "date": None,
        "avg_sentiment_score": None,
        "headline_count": None,
        "bullish_count": None,
        "bearish_count": None,
        "neutral_count": None,
        "recent_headlines": [],
    }

    # --- Daily aggregate ---
    try:
        daily_query = text("""
            SELECT date, avg_sentiment_score, headline_count,
                   bullish_count, bearish_count, neutral_count
            FROM news_sentiment_daily
            WHERE symbol = :symbol
            ORDER BY date DESC
            LIMIT 1
        """)
        with engine.connect() as conn:
            row = conn.execute(daily_query, {"symbol": ticker}).mappings().fetchone()
        if row:
            result.update(dict(row))
    except Exception:
        logger.warning(
            f"Could not read 'news_sentiment_daily' for {ticker} "
            f"(table may not exist yet)."
        )

    # --- Recent raw headlines ---
    try:
        headlines_query = text("""
            SELECT headline, source, url, published_at,
                   sentiment_score, sentiment_label, key_reason
            FROM news_sentiment_raw
            WHERE symbol = :symbol
            ORDER BY published_at DESC
            LIMIT 5
        """)
        with engine.connect() as conn:
            rows = conn.execute(headlines_query, {"symbol": ticker}).mappings().fetchall()
        result["recent_headlines"] = [dict(r) for r in rows]
    except Exception:
        logger.warning(
            f"Could not read 'news_sentiment_raw' for {ticker} "
            f"(table may not exist yet)."
        )

    return result


# ==========================================================================
# 8. FASTAPI APP SETUP
# ==========================================================================

app = FastAPI(
    title="Real-Time Stock Market Intelligence Platform API",
    description=(
        "Backend API providing historical stock data, technical indicators, "
        "AI-driven news sentiment, and ML-based next-day price direction "
        "predictions for 10 popular Indian stocks.\n\n"
        "**Authentication:** All endpoints (except `/` and `/health`) require "
        "an `X-API-Key` header."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc UI
)

# --- CORS middleware: allows a Streamlit frontend (any origin/port) to
#     call this API from the browser. ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# 9. IN-MEMORY ALERTS STORE (for the POST /alerts endpoint)
# ==========================================================================
# For a portfolio project, we persist alerts in the same database using a
# simple table created on startup. This keeps alerts durable across
# restarts without requiring a separate Phase-1-style ORM model file.

def ensure_alerts_table():
    """Create the 'price_alerts' table if it doesn't already exist."""
    if DB_KIND == "postgresql":
        ddl = """
            CREATE TABLE IF NOT EXISTS price_alerts (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                target_price DOUBLE PRECISION NOT NULL,
                condition VARCHAR(10) NOT NULL,
                note VARCHAR(255),
                created_at TIMESTAMP NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE
            )
        """
    else:  # sqlite
        ddl = """
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                target_price REAL NOT NULL,
                condition VARCHAR(10) NOT NULL,
                note VARCHAR(255),
                created_at TIMESTAMP NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
        """
    with engine.begin() as conn:
        conn.execute(text(ddl))


ensure_alerts_table()


# ==========================================================================
# 10. ROUTES
# ==========================================================================

# --------------------------------------------------------------------
# Root & health check (no auth required -- useful for load balancers /
# uptime monitors / quick sanity checks)
# --------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root():
    """Simple landing endpoint confirming the API is running."""
    return {
        "message": "Stock Market Intelligence Platform API is running.",
        "docs": "/docs",
    }


@app.get("/health", tags=["Meta"])
def health_check():
    """
    Health check endpoint. Verifies the database connection is alive.
    Does NOT require an API key, so monitoring tools can call it freely.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        logger.exception("Health check: database connection failed.")
        db_status = "error"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "ml_model_loaded": ml_model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


# --------------------------------------------------------------------
# GET /stocks - list all available stocks
# --------------------------------------------------------------------

@app.get(
    "/stocks",
    response_model=List[StockInfo],
    tags=["Stocks"],
    summary="List all tracked stocks",
    dependencies=[Depends(verify_api_key)],
)
def list_stocks():
    """Return the static list of all 10 stocks tracked by this platform."""
    return [{"symbol": sym, "name": name} for sym, name in KNOWN_STOCKS.items()]


# --------------------------------------------------------------------
# GET /stocks/{ticker} - latest price + technical indicators
# --------------------------------------------------------------------

@app.get(
    "/stocks/{ticker}",
    response_model=LatestPriceResponse,
    tags=["Stocks"],
    summary="Get latest price and technical indicators for a stock",
    dependencies=[Depends(verify_api_key)],
    responses={404: {"description": "Stock not found or no data available"}},
)
def get_stock_latest(
    ticker: str = Path(..., description="Stock ticker, e.g. TCS.NS", example="TCS.NS"),
):
    """
    Returns the most recent OHLCV values plus all technical indicators
    (SMA20/50, RSI14, MACD, Bollinger Bands) for the given ticker.
    """
    ticker = validate_ticker(ticker)

    try:
        row = fetch_latest_price_and_indicators(ticker)
    except Exception:
        logger.exception(f"Database error fetching latest data for {ticker}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching stock data.",
        )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No price data found for '{ticker}'. "
                f"Has the data pipeline (Phase 1) been run yet?"
            ),
        )

    return row


# --------------------------------------------------------------------
# GET /stocks/{ticker}/history - historical OHLCV data
# --------------------------------------------------------------------

@app.get(
    "/stocks/{ticker}/history",
    response_model=HistoryResponse,
    tags=["Stocks"],
    summary="Get historical OHLCV data for a stock",
    dependencies=[Depends(verify_api_key)],
    responses={404: {"description": "Stock not found or no data available"}},
)
def get_stock_history(
    ticker: str = Path(..., description="Stock ticker, e.g. TCS.NS", example="TCS.NS"),
    days: int = Query(
        30, ge=1, le=365,
        description="Number of most recent trading days to return (1-365). Default 30.",
    ),
):
    """
    Returns up to `days` most recent daily OHLCV records for the given
    ticker, sorted oldest-to-newest (suitable for direct charting).
    """
    ticker = validate_ticker(ticker)

    try:
        df = fetch_history(ticker, days)
    except Exception:
        logger.exception(f"Database error fetching history for {ticker}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching historical data.",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No historical data found for '{ticker}'. "
                f"Has the data pipeline (Phase 1) been run yet?"
            ),
        )

    return {
        "symbol": ticker,
        "name": KNOWN_STOCKS[ticker],
        "days_requested": days,
        "rows_returned": len(df),
        "data": df.to_dict(orient="records"),
    }


# --------------------------------------------------------------------
# GET /stocks/{ticker}/sentiment - latest AI sentiment scores
# --------------------------------------------------------------------

@app.get(
    "/stocks/{ticker}/sentiment",
    response_model=SentimentResponse,
    tags=["Sentiment"],
    summary="Get latest AI-derived news sentiment for a stock",
    dependencies=[Depends(verify_api_key)],
)
def get_stock_sentiment(
    ticker: str = Path(..., description="Stock ticker, e.g. TCS.NS", example="TCS.NS"),
):
    """
    Returns the most recent daily aggregated sentiment score (from
    Claude-analyzed news headlines via Phase 2) plus up to 5 of the most
    recent individual headlines and their sentiment.

    If no sentiment data has been collected yet for this stock, returns
    a response with null/empty sentiment fields rather than a 404 -- this
    is treated as "no news sentiment available yet", not an error.
    """
    ticker = validate_ticker(ticker)

    try:
        sentiment = fetch_latest_sentiment(ticker)
    except Exception:
        logger.exception(f"Database error fetching sentiment for {ticker}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching sentiment data.",
        )

    return {
        "symbol": ticker,
        "name": KNOWN_STOCKS[ticker],
        **sentiment,
    }


# --------------------------------------------------------------------
# GET /stocks/{ticker}/predict - ML model prediction
# --------------------------------------------------------------------

@app.get(
    "/stocks/{ticker}/predict",
    response_model=PredictionResponse,
    tags=["ML Predictions"],
    summary="Get next-day price direction prediction (Up/Down)",
    dependencies=[Depends(verify_api_key)],
    responses={
        404: {"description": "Stock not found, or no feature data available for prediction"},
        503: {"description": "ML model not loaded / not yet trained"},
    },
)
def get_stock_prediction(
    ticker: str = Path(..., description="Stock ticker, e.g. TCS.NS", example="TCS.NS"),
):
    """
    Returns the XGBoost model's prediction for whether `ticker`'s closing
    price will go Up or Down on the next trading day, along with the
    model's confidence (probability of the predicted class).

    Requires that `python ml_model.py train` has been run at least once
    (so that models/xgb_direction_model.joblib and the latest feature
    snapshot exist).
    """
    ticker = validate_ticker(ticker)

    if ml_model is None or feature_columns is None or symbol_encoder is None or latest_features_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ML model is not available. Run 'python ml_model.py train' "
                "to train and save the model first."
            ),
        )

    if ticker not in symbol_encoder.classes_:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trained model data available for '{ticker}'.",
        )

    row = latest_features_df[latest_features_df["symbol"] == ticker]
    if row.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No feature data available for '{ticker}' to make a "
                f"prediction. It may not have enough historical data yet."
            ),
        )

    row = row.iloc[[-1]]
    X = row[feature_columns]

    if X.isnull().any(axis=None):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Feature data for '{ticker}' contains missing values "
                f"(insufficient history for rolling/lag indicators)."
            ),
        )

    try:
        pred_class = int(ml_model.predict(X)[0])
        pred_proba = ml_model.predict_proba(X)[0]
        confidence = float(pred_proba[pred_class])
    except Exception:
        logger.exception(f"ML prediction failed for {ticker}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while generating prediction.",
        )

    return {
        "symbol": ticker,
        "prediction": "Up" if pred_class == 1 else "Down",
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.2f}%",
        "as_of_date": row["date"].iloc[0].date(),
    }


# --------------------------------------------------------------------
# GET /dashboard/summary - aggregate stats for all stocks
# --------------------------------------------------------------------

@app.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
    tags=["Dashboard"],
    summary="Get a one-row-per-stock summary for all tracked stocks",
    dependencies=[Depends(verify_api_key)],
)
def get_dashboard_summary():
    """
    Returns a compact summary across all 10 stocks, suitable for
    populating a dashboard table/overview:
        - latest close price
        - day-over-day % change
        - RSI(14)
        - latest average news sentiment score & label

    Stocks with no data yet are still included, with null fields, so the
    frontend can render a consistent table.
    """
    summary_rows = []

    for symbol, name in KNOWN_STOCKS.items():
        row = {
            "symbol": symbol,
            "name": name,
            "date": None,
            "close": None,
            "daily_change_pct": None,
            "rsi_14": None,
            "avg_sentiment_score": None,
            "sentiment_label_today": None,
        }

        try:
            # --- Latest + previous close, for % change ---
            price_query = text("""
                SELECT date, close
                FROM stock_prices
                WHERE symbol = :symbol
                ORDER BY date DESC
                LIMIT 2
            """)
            with engine.connect() as conn:
                price_rows = conn.execute(price_query, {"symbol": symbol}).mappings().fetchall()

            if price_rows:
                latest = price_rows[0]
                raw_date = latest["date"]
                if isinstance(raw_date, str):
                    row["date"] = pd.to_datetime(raw_date).date()
                elif hasattr(raw_date, "date"):
                    row["date"] = raw_date.date()
                else:
                    row["date"] = raw_date
                row["close"] = latest["close"]

                if len(price_rows) == 2 and price_rows[1]["close"]:
                    prev_close = price_rows[1]["close"]
                    row["daily_change_pct"] = round(
                        ((latest["close"] - prev_close) / prev_close) * 100, 2
                    )

            # --- Latest RSI ---
            rsi_query = text("""
                SELECT rsi_14
                FROM stock_indicators
                WHERE symbol = :symbol
                ORDER BY date DESC
                LIMIT 1
            """)
            with engine.connect() as conn:
                rsi_row = conn.execute(rsi_query, {"symbol": symbol}).mappings().fetchone()
            if rsi_row:
                row["rsi_14"] = rsi_row["rsi_14"]

            # --- Latest sentiment ---
            try:
                sentiment_query = text("""
                    SELECT avg_sentiment_score
                    FROM news_sentiment_daily
                    WHERE symbol = :symbol
                    ORDER BY date DESC
                    LIMIT 1
                """)
                with engine.connect() as conn:
                    sent_row = conn.execute(sentiment_query, {"symbol": symbol}).mappings().fetchone()
                if sent_row:
                    score = sent_row["avg_sentiment_score"]
                    row["avg_sentiment_score"] = score
                    if score is not None:
                        if score > 1:
                            row["sentiment_label_today"] = "Bullish"
                        elif score < -1:
                            row["sentiment_label_today"] = "Bearish"
                        else:
                            row["sentiment_label_today"] = "Neutral"
            except Exception:
                # Sentiment table may not exist yet -- not a fatal error
                pass

        except Exception:
            logger.exception(f"Error building dashboard summary row for {symbol}.")
            # Leave row with null values rather than failing the whole request

        summary_rows.append(row)

    return {
        "as_of": datetime.utcnow(),
        "total_stocks": len(summary_rows),
        "stocks": summary_rows,
    }


# --------------------------------------------------------------------
# POST /alerts - create a price alert
# --------------------------------------------------------------------

@app.post(
    "/alerts",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Alerts"],
    summary="Create a price alert for a stock",
    dependencies=[Depends(verify_api_key)],
)
def create_alert(alert: AlertCreateRequest):
    """
    Create a new price alert. The alert is stored in the `price_alerts`
    table and can later be checked by a separate monitoring job (not part
    of this API) that compares current prices against `target_price` and
    `condition`.

    `condition` must be either:
        - "above": trigger when price >= target_price
        - "below": trigger when price <= target_price
    """
    symbol = validate_ticker(alert.symbol)

    condition = alert.condition.lower().strip()
    if condition not in ("above", "below"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="`condition` must be either 'above' or 'below'.",
        )

    created_at = datetime.utcnow()

    insert_query = text("""
        INSERT INTO price_alerts (symbol, target_price, condition, note, created_at, is_active)
        VALUES (:symbol, :target_price, :condition, :note, :created_at, :is_active)
    """)

    try:
        with engine.begin() as conn:
            result = conn.execute(
                insert_query,
                {
                    "symbol": symbol,
                    "target_price": alert.target_price,
                    "condition": condition,
                    "note": alert.note,
                    "created_at": created_at,
                    "is_active": True,
                },
            )

            # Retrieve the newly created row's ID.
            # SQLite: use cursor.lastrowid via result.inserted_primary_key
            # PostgreSQL: same approach works via SQLAlchemy's RETURNING-less
            # insert when using the default autoincrement/serial PK.
            if DB_KIND == "sqlite":
                new_id = result.lastrowid
            else:
                # For PostgreSQL, fetch the max id for this symbol/created_at
                # as a simple, dependency-free way to retrieve the new ID.
                id_query = text("""
                    SELECT id FROM price_alerts
                    WHERE symbol = :symbol AND created_at = :created_at
                    ORDER BY id DESC LIMIT 1
                """)
                row = conn.execute(id_query, {"symbol": symbol, "created_at": created_at}).mappings().fetchone()
                new_id = row["id"] if row else -1

    except Exception:
        logger.exception(f"Database error creating alert for {symbol}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while creating the alert.",
        )

    return {
        "id": new_id,
        "symbol": symbol,
        "target_price": alert.target_price,
        "condition": condition,
        "note": alert.note,
        "created_at": created_at,
        "is_active": True,
    }


# ==========================================================================
# 11. GLOBAL EXCEPTION HANDLER (catch-all safety net)
# ==========================================================================
# Any unhandled exception not already converted to an HTTPException above
# will be caught here and returned as a clean 500 response instead of
# leaking a stack trace to the client.

from fastapi.requests import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred."},
    )


# ==========================================================================
# 12. LOCAL DEV ENTRY POINT
# ==========================================================================
# Allows running `python api.py` directly for quick local testing, though
# `uvicorn api:app --reload` is recommended during development.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
