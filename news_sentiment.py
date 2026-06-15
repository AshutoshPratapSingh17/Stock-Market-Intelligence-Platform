"""
==========================================================================
 Real-Time Stock Market Intelligence Platform - Phase 2
 Script 1: news_sentiment.py

 What this script does:
   1. Fetches latest financial news headlines for each of the 10 stocks
      using NewsAPI (https://newsapi.org) - free developer tier.
   2. Sends each headline to the Claude API (model: claude-sonnet-4-6)
      with a strict prompt asking for JSON containing:
          - sentiment_score   (integer, -10 to +10)
          - sentiment_label   ("Bullish" / "Bearish" / "Neutral")
          - key_reason        (one-sentence explanation)
   3. Aggregates these per-headline scores into a single
      "average sentiment score" per stock, per day.
   4. Stores both the raw per-headline results AND the daily
      aggregate into the database (SQLite or PostgreSQL, same
      DB_KIND/DATABASE_URL logic as Phase 1).
   5. Includes rate limiting (to respect NewsAPI & Claude API limits)
      and robust error handling so one bad headline/stock doesn't
      crash the whole run.

 Environment variables required:
   NEWSAPI_KEY      -> your free NewsAPI.org API key
   ANTHROPIC_API_KEY -> your Anthropic Claude API key

 Run:
    python news_sentiment.py            -> runs once
    python news_sentiment.py --once     -> same as above (explicit)
==========================================================================
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, date
from logging.handlers import RotatingFileHandler

import requests
import pandas as pd

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Date, Text,
    UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# Anthropic Python SDK
import anthropic


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

# --- Same 10 stocks as Phase 1 ---
# For NewsAPI we search using a human-readable company name (better hit
# rate than ticker symbols), but we keep the ticker as the DB key so it
# joins cleanly with the Phase 1 tables.
STOCKS = {
    "TCS.NS":        {"name": "TCS",           "query": "Tata Consultancy Services TCS"},
    "INFY.NS":       {"name": "INFOSYS",       "query": "Infosys"},
    "RELIANCE.NS":   {"name": "RELIANCE",      "query": "Reliance Industries"},
    "HDFCBANK.NS":   {"name": "HDFC_BANK",     "query": "HDFC Bank"},
    "WIPRO.NS":      {"name": "WIPRO",         "query": "Wipro"},
    "ICICIBANK.NS":  {"name": "ICICI_BANK",    "query": "ICICI Bank"},
    "HCLTECH.NS":    {"name": "HCL_TECH",      "query": "HCL Technologies"},
    "BAJFINANCE.NS": {"name": "BAJAJ_FINANCE", "query": "Bajaj Finance"},
    "ASIANPAINT.NS": {"name": "ASIAN_PAINTS",  "query": "Asian Paints"},
    "MARUTI.NS":     {"name": "MARUTI",        "query": "Maruti Suzuki"},
}

# --- NewsAPI configuration ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
MAX_HEADLINES_PER_STOCK = 5     # keep low to respect free-tier limits (100 req/day)

# --- Claude API configuration ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-6"

# --- Rate limiting ---
# NewsAPI free tier: 100 requests/day -> 1 request per stock per run is fine,
# but we still add a small delay to be polite.
NEWSAPI_DELAY_SECONDS = 1.0
# Claude API: add a small delay between calls to avoid hitting rate limits
# when processing many headlines back-to-back.
CLAUDE_DELAY_SECONDS = 1.0

# --- Database configuration (mirrors Phase 1 logic) ---
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


# ==========================================================================
# 2. LOGGING SETUP
# ==========================================================================

logger = logging.getLogger("news_sentiment")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = RotatingFileHandler(
    "news_sentiment.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ==========================================================================
# 3. DATABASE MODELS
# ==========================================================================
# Two new tables:
#   - news_sentiment_raw   : one row per headline, with Claude's analysis
#   - news_sentiment_daily : one row per stock per day, aggregated score
#
# Both have UNIQUE constraints to support UPSERT (no duplicates on rerun).

Base = declarative_base()


class NewsSentimentRaw(Base):
    """One row per news headline analyzed by Claude."""
    __tablename__ = "news_sentiment_raw"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    name = Column(String(50), nullable=False)
    published_at = Column(DateTime, nullable=False)
    headline = Column(Text, nullable=False)
    source = Column(String(100))
    url = Column(Text)

    sentiment_score = Column(Integer)      # -10 to +10
    sentiment_label = Column(String(20))   # Bullish / Bearish / Neutral
    key_reason = Column(Text)

    fetched_at = Column(DateTime, nullable=False)

    __table_args__ = (
        # Prevent storing the exact same headline+stock twice
        UniqueConstraint("symbol", "url", name="uq_news_symbol_url"),
    )


class NewsSentimentDaily(Base):
    """Aggregated average sentiment per stock per calendar day."""
    __tablename__ = "news_sentiment_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    name = Column(String(50), nullable=False)
    date = Column(Date, nullable=False, index=True)

    avg_sentiment_score = Column(Float)    # average of sentiment_score
    headline_count = Column(Integer)       # how many headlines contributed
    bullish_count = Column(Integer)
    bearish_count = Column(Integer)
    neutral_count = Column(Integer)

    updated_at = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_sentiment_symbol_date"),
    )


# ==========================================================================
# 4. DATABASE ENGINE SETUP
# ==========================================================================

def get_engine():
    """
    Create a SQLAlchemy engine, falling back to SQLite if PostgreSQL
    is configured but unreachable (mirrors Phase 1 behaviour).
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


# ==========================================================================
# 5. NEWSAPI FETCHING
# ==========================================================================

def fetch_headlines(query: str, symbol: str) -> list:
    """
    Fetch the latest news headlines for a given search query using NewsAPI.

    Returns a list of dicts: {title, source, url, published_at}
    Returns an empty list on failure (and logs the issue).
    """
    if not NEWSAPI_KEY:
        logger.error("NEWSAPI_KEY is not set. Cannot fetch headlines.")
        return []

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": MAX_HEADLINES_PER_STOCK,
        "apiKey": NEWSAPI_KEY,
    }

    try:
        logger.info(f"Fetching news headlines for {symbol} (query='{query}')...")
        response = requests.get(NEWSAPI_URL, params=params, timeout=15)

        # Handle common NewsAPI error responses explicitly
        if response.status_code == 401:
            logger.error("NewsAPI returned 401 Unauthorized. Check NEWSAPI_KEY.")
            return []
        if response.status_code == 429:
            logger.warning("NewsAPI rate limit reached (429). Skipping this stock.")
            return []

        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            logger.warning(f"NewsAPI returned non-ok status for {symbol}: {data.get('message')}")
            return []

        articles = data.get("articles", [])
        headlines = []
        for art in articles:
            headlines.append({
                "title": art.get("title", "").strip(),
                "source": (art.get("source") or {}).get("name", "Unknown"),
                "url": art.get("url", ""),
                "published_at": art.get("publishedAt"),
            })

        # Filter out junk entries (NewsAPI sometimes returns "[Removed]")
        headlines = [h for h in headlines if h["title"] and h["title"] != "[Removed]"]

        logger.info(f"Fetched {len(headlines)} headlines for {symbol}.")
        return headlines

    except requests.exceptions.RequestException:
        logger.exception(f"Network error while fetching news for {symbol}.")
        return []
    except Exception:
        logger.exception(f"Unexpected error while fetching news for {symbol}.")
        return []


# ==========================================================================
# 6. CLAUDE API SENTIMENT ANALYSIS
# ==========================================================================

# Initialize the Anthropic client once (reused across calls)
_claude_client = None


def get_claude_client():
    """Lazily initialize and return the Anthropic client."""
    global _claude_client
    if _claude_client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        _claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _claude_client


# The system prompt is fixed and instructs Claude to ALWAYS respond with
# strict JSON only -- no markdown, no extra commentary. This makes parsing
# reliable.
SENTIMENT_SYSTEM_PROMPT = """\
You are a financial news sentiment analysis engine for an Indian stock \
market intelligence platform. You will be given a single news headline \
related to a specific company's stock.

Analyze the headline and respond with ONLY a single valid JSON object \
(no markdown formatting, no code fences, no extra text before or after) \
with EXACTLY these three keys:

{
  "sentiment_score": <integer from -10 to +10, where -10 is extremely \
bearish/negative for the stock price, 0 is neutral, and +10 is extremely \
bullish/positive>,
  "sentiment_label": <one of "Bullish", "Bearish", or "Neutral">,
  "key_reason": <a single concise sentence explaining the score>
}

If the headline is unrelated to the company's stock price outlook, \
return sentiment_score 0 and sentiment_label "Neutral".
"""


def analyze_headline_sentiment(headline: str, company_name: str) -> dict:
    """
    Send a single headline to Claude and parse the strict JSON response.

    Returns a dict: {sentiment_score, sentiment_label, key_reason}
    On failure, returns a "Neutral" / score-0 default so the pipeline
    can continue without crashing.
    """
    default_result = {
        "sentiment_score": 0,
        "sentiment_label": "Neutral",
        "key_reason": "Could not analyze headline due to an error.",
    }

    try:
        client = get_claude_client()

        user_message = (
            f"Company: {company_name}\n"
            f"Headline: {headline}"
        )

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            system=SENTIMENT_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ],
        )

        # Extract the text content from the response
        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        # Defensive parsing: in case Claude wraps JSON in code fences despite
        # instructions, strip them before parsing.
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove a leading "json" language hint if present
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        # Validate and coerce types / ranges
        score = int(parsed.get("sentiment_score", 0))
        score = max(-10, min(10, score))  # clamp to [-10, 10]

        label = str(parsed.get("sentiment_label", "Neutral")).strip()
        if label not in ("Bullish", "Bearish", "Neutral"):
            label = "Neutral"

        reason = str(parsed.get("key_reason", "")).strip() or "No reason provided."

        return {
            "sentiment_score": score,
            "sentiment_label": label,
            "key_reason": reason,
        }

    except json.JSONDecodeError:
        logger.error(f"Claude returned invalid JSON for headline: '{headline[:80]}...'")
        return default_result
    except anthropic.APIError:
        logger.exception(f"Claude API error while analyzing headline: '{headline[:80]}...'")
        return default_result
    except Exception:
        logger.exception(f"Unexpected error analyzing headline: '{headline[:80]}...'")
        return default_result


# ==========================================================================
# 7. PARSE DATES SAFELY
# ==========================================================================

def parse_published_at(value: str) -> datetime:
    """
    NewsAPI returns ISO-8601 timestamps like '2024-06-01T12:34:56Z'.
    Convert to a naive datetime (UTC, no tzinfo) for DB storage.
    Falls back to "now" if parsing fails.
    """
    try:
        # Replace 'Z' with '+00:00' so fromisoformat can parse it
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


# ==========================================================================
# 8. DATABASE WRITE (UPSERT) HELPERS
# ==========================================================================

def upsert_dataframe(engine, df: pd.DataFrame, model, conflict_cols, update_cols):
    """Generic upsert helper (same pattern as Phase 1)."""
    if df.empty:
        logger.warning(f"No rows to write to {model.__tablename__}.")
        return

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    table = model.__table__

    if DB_KIND == "postgresql":
        insert_stmt = pg_insert(table).values(records)
    else:
        insert_stmt = sqlite_insert(table).values(records)

    update_dict = {col: getattr(insert_stmt.excluded, col) for col in update_cols}

    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=conflict_cols,
        set_=update_dict,
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)

    logger.info(f"Upserted {len(records)} rows into {model.__tablename__}.")


# ==========================================================================
# 9. MAIN PIPELINE
# ==========================================================================

def run_sentiment_pipeline():
    """
    Main job:
      1. Connect to DB, create tables if needed.
      2. For each stock:
           a. Fetch headlines from NewsAPI.
           b. For each headline, call Claude for sentiment.
           c. Store each headline's result (raw table).
           d. Aggregate scores -> store daily summary (daily table).
      3. Respect rate limits with small sleeps between API calls.
    """
    logger.info("=" * 70)
    logger.info("Starting news sentiment pipeline run...")

    if not NEWSAPI_KEY:
        logger.error("NEWSAPI_KEY environment variable is not set. Aborting.")
        return
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY environment variable is not set. Aborting.")
        return

    try:
        engine = get_engine()
    except Exception:
        logger.exception("Could not connect to the database. Aborting.")
        return

    Base.metadata.create_all(engine)

    today = date.today()

    for symbol, info in STOCKS.items():
        name = info["name"]
        query = info["query"]

        logger.info("-" * 50)
        logger.info(f"Processing news sentiment for {name} ({symbol})")

        # --- Step 1: Fetch headlines ---
        headlines = fetch_headlines(query, symbol)
        time.sleep(NEWSAPI_DELAY_SECONDS)  # rate limiting for NewsAPI

        if not headlines:
            logger.warning(f"No headlines found for {name}. Skipping sentiment analysis.")
            continue

        # --- Step 2: Analyze each headline with Claude ---
        raw_rows = []
        scores = []
        label_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0}

        for h in headlines:
            try:
                logger.info(f"Analyzing headline: '{h['title'][:80]}...'")
                result = analyze_headline_sentiment(h["title"], name)

                raw_rows.append({
                    "symbol": symbol,
                    "name": name,
                    "published_at": parse_published_at(h["published_at"]),
                    "headline": h["title"],
                    "source": h["source"],
                    "url": h["url"],
                    "sentiment_score": result["sentiment_score"],
                    "sentiment_label": result["sentiment_label"],
                    "key_reason": result["key_reason"],
                    "fetched_at": datetime.utcnow(),
                })

                scores.append(result["sentiment_score"])
                label_counts[result["sentiment_label"]] += 1

            except Exception:
                logger.exception(f"Failed to process headline for {name}: '{h.get('title', '')[:80]}'")
                continue
            finally:
                # Rate limiting between Claude API calls
                time.sleep(CLAUDE_DELAY_SECONDS)

        # --- Step 3: Store raw per-headline results ---
        raw_df = pd.DataFrame(raw_rows)
        upsert_dataframe(
            engine,
            raw_df,
            NewsSentimentRaw,
            conflict_cols=["symbol", "url"],
            update_cols=[
                "name", "published_at", "headline", "source",
                "sentiment_score", "sentiment_label", "key_reason", "fetched_at",
            ],
        )

        # --- Step 4: Aggregate and store daily summary ---
        if scores:
            avg_score = sum(scores) / len(scores)
            daily_row = pd.DataFrame([{
                "symbol": symbol,
                "name": name,
                "date": today,
                "avg_sentiment_score": avg_score,
                "headline_count": len(scores),
                "bullish_count": label_counts["Bullish"],
                "bearish_count": label_counts["Bearish"],
                "neutral_count": label_counts["Neutral"],
                "updated_at": datetime.utcnow(),
            }])

            upsert_dataframe(
                engine,
                daily_row,
                NewsSentimentDaily,
                conflict_cols=["symbol", "date"],
                update_cols=[
                    "name", "avg_sentiment_score", "headline_count",
                    "bullish_count", "bearish_count", "neutral_count", "updated_at",
                ],
            )

            logger.info(
                f"{name}: avg_sentiment_score={avg_score:.2f} "
                f"(Bullish={label_counts['Bullish']}, "
                f"Bearish={label_counts['Bearish']}, "
                f"Neutral={label_counts['Neutral']})"
            )

    logger.info("-" * 50)
    logger.info("News sentiment pipeline run complete.")
    logger.info("=" * 70)


# ==========================================================================
# 10. ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch news headlines, analyze sentiment with Claude, store in DB."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the pipeline a single time (default behaviour).",
    )
    args = parser.parse_args()

    run_sentiment_pipeline()
