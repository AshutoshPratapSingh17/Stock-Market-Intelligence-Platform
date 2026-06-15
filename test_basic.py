"""
==========================================================================
 test_basic.py - Basic smoke tests for CI (GitHub Actions)

 These tests do NOT require a live database or external API keys.
 They verify that:
   - All core modules import successfully (no syntax/dependency errors)
   - Technical indicator calculations produce correct, sane values
   - The FastAPI app can be instantiated and basic routes respond

 Run:
   pytest test_basic.py -v
==========================================================================
"""

import os
import sys
import importlib

import pandas as pd
import pytest

# Ensure SQLite fallback is used during tests (no real DB needed)
os.environ.setdefault("SQLITE_PATH", "test_stocks.db")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.pop("PG_HOST", None)
os.environ.pop("PG_DB", None)
os.environ.pop("PG_USER", None)
os.environ.pop("PG_PASSWORD", None)


# ==========================================================================
# 1. Import tests - make sure every module loads without error
# ==========================================================================

def test_import_stock_pipeline():
    module = importlib.import_module("stock_pipeline")
    assert hasattr(module, "calculate_rsi")
    assert hasattr(module, "calculate_macd")
    assert hasattr(module, "calculate_bollinger_bands")


def test_import_ml_model():
    module = importlib.import_module("ml_model")
    assert hasattr(module, "engineer_features")
    assert hasattr(module, "FEATURE_COLUMNS")


def test_import_api():
    module = importlib.import_module("api")
    assert hasattr(module, "app")


# ==========================================================================
# 2. Technical indicator correctness tests
# ==========================================================================

def test_rsi_bounds():
    """RSI must always be between 0 and 100."""
    from stock_pipeline import calculate_rsi

    prices = pd.Series([100, 102, 101, 105, 110, 108, 107, 111, 115, 113,
                         112, 116, 120, 119, 121, 125, 123, 122, 126, 130])
    rsi = calculate_rsi(prices, period=14)

    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_macd_shapes():
    """MACD line, signal line, and histogram must all align in length."""
    from stock_pipeline import calculate_macd

    prices = pd.Series(range(100, 150))
    macd_line, signal_line, hist = calculate_macd(prices)

    assert len(macd_line) == len(prices)
    assert len(signal_line) == len(prices)
    assert len(hist) == len(prices)

    # Histogram should equal macd_line - signal_line
    pd.testing.assert_series_equal(hist, macd_line - signal_line, check_names=False)


def test_bollinger_bands_ordering():
    """Upper band must always be >= middle band >= lower band."""
    from stock_pipeline import calculate_bollinger_bands

    prices = pd.Series([100, 102, 101, 105, 110, 108, 107, 111, 115, 113,
                         112, 116, 120, 119, 121, 125, 123, 122, 126, 130])
    upper, middle, lower = calculate_bollinger_bands(prices, window=5)

    valid_idx = upper.dropna().index
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (middle[valid_idx] >= lower[valid_idx]).all()


# ==========================================================================
# 3. FastAPI app tests (no DB required for these basic checks)
# ==========================================================================

def test_api_root_endpoint():
    """The root endpoint should respond without requiring auth."""
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_api_health_endpoint():
    """The /health endpoint should respond without requiring auth."""
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "database" in body


def test_api_requires_key_for_protected_route():
    """/stocks should return 401 if no API key is provided."""
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    response = client.get("/stocks")

    assert response.status_code == 401


def test_api_stocks_with_valid_key():
    """/stocks should return 200 and a list of 10 stocks with a valid key."""
    from fastapi.testclient import TestClient
    from api import app, API_KEY

    client = TestClient(app)
    response = client.get("/stocks", headers={"X-API-Key": API_KEY})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 10


# ==========================================================================
# Cleanup: remove the test SQLite file after the test session
# ==========================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db():
    yield
    if os.path.exists("test_stocks.db"):
        os.remove("test_stocks.db")
