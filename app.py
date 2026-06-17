"""
==========================================================================
 Real-Time Stock Market Intelligence Platform - Phase 4 (v3)
 app.py - Premium Streamlit Dashboard (simplified + fixed)

 CHANGES IN THIS VERSION:
   - REMOVED the animated ticker tape banner (was unreliable/buggy)
   - REMOVED the AI Sentiment KPI card and the entire News & Sentiment tab
     (these depended on Phase 2 sentiment data which is not populated)
   - Hardcoded a fixed API_KEY shared between this file and api.py, so
     there's no more env-var mismatch between terminals. You can still
     override via the API_KEY environment variable if set.
   - Added company logo images next to each stock's name (via Clearbit's
     free logo API, using each company's domain)
   - Added a hero banner image at the top of the page for visual polish
   - Kept: Overview tab (candlestick + MA + RSI + MACD), ML Insights tab
     (prediction card + feature importance), Compare tab (2-stock compare
     + correlation heatmap)

 IMPORTANT: api.py must use the SAME API_KEY value. See the matching
 patch instructions provided alongside this file.

 Run:
   streamlit run app.py
==========================================================================
"""

import os
import time
from datetime import date, datetime, timedelta

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Fixed shared API key -- MUST match the value in api.py's FIXED_API_KEY.
# Can still be overridden by setting the API_KEY environment variable.
FIXED_API_KEY = "stockplatform-shared-key-2026"
API_KEY = os.getenv("API_KEY", FIXED_API_KEY)

HEADERS = {"X-API-Key": API_KEY}
REQUEST_TIMEOUT = 10  # seconds

# The 10 stocks tracked by the platform.
# Each entry: display name + company domain (for logo lookup via Clearbit).
STOCKS = {
    "TCS.NS":        {"name": "TCS",                  "domain": "tcs.com"},
    "INFY.NS":       {"name": "Infosys",              "domain": "infosys.com"},
    "RELIANCE.NS":   {"name": "Reliance Industries",  "domain": "ril.com"},
    "HDFCBANK.NS":   {"name": "HDFC Bank",            "domain": "hdfcbank.com"},
    "WIPRO.NS":      {"name": "Wipro",                "domain": "wipro.com"},
    "ICICIBANK.NS":  {"name": "ICICI Bank",           "domain": "icicibank.com"},
    "HCLTECH.NS":    {"name": "HCL Technologies",     "domain": "hcltech.com"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance",        "domain": "bajajfinserv.in"},
    "ASIANPAINT.NS": {"name": "Asian Paints",         "domain": "asianpaints.com"},
    "MARUTI.NS":     {"name": "Maruti Suzuki",        "domain": "marutisuzuki.com"},
}

AUTO_REFRESH_SECONDS = 30


def logo_url(symbol: str) -> str:
    """Return a logo image URL for the given stock symbol using Clearbit's
    free logo API (https://logo.clearbit.com/<domain>)."""
    domain = STOCKS[symbol]["domain"]
    return f"https://logo.clearbit.com/{domain}?size=128"


# ==========================================================================
# 2. PAGE CONFIG + PREMIUM DARK THEME (CSS)
# ==========================================================================

st.set_page_config(
    page_title="Stock Market Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #0b0e14;
        color: #e9ebf1;
    }

    /* Hide the default white Streamlit header bar */
    header[data-testid="stHeader"] {
        background-color: #0b0e14;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    section[data-testid="stSidebar"] {
        background-color: #11141c;
        border-right: 1px solid #1f2430;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stDateInput label {
        font-weight: 600;
        font-size: 0.8rem;
        color: #8b93a7;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ====================================================================
       HERO BANNER
    ==================================================================== */
    .hero-banner {
        position: relative;
        border-radius: 18px;
        overflow: hidden;
        margin-bottom: 22px;
        height: 180px;
        background-image:
            linear-gradient(120deg, rgba(11,14,20,0.55) 0%, rgba(11,14,20,0.85) 60%, rgba(11,14,20,1) 100%),
            url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=1600&q=60');
        background-size: cover;
        background-position: center;
        display: flex;
        align-items: center;
        padding: 0 32px;
    }
    .hero-content h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.01em;
    }
    .hero-content p {
        color: #c3c8d6;
        font-size: 0.95rem;
        margin: 0;
        max-width: 560px;
    }

    /* ====================================================================
       FADE-IN ANIMATION
    ==================================================================== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.5s ease-out; }

    /* ====================================================================
       BRAND / PAGE HEADER (with logo)
    ==================================================================== */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .brand-logo {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        object-fit: contain;
        background: #ffffff;
        padding: 6px;
        border: 1px solid #232838;
    }
    .brand-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .brand-ticker {
        font-size: 1.05rem;
        font-weight: 700;
        color: #8a93ff;
        background: rgba(138, 147, 255, 0.12);
        border: 1px solid rgba(138, 147, 255, 0.25);
        border-radius: 8px;
        padding: 3px 12px;
    }
    .brand-sub {
        color: #6b7388;
        font-size: 0.95rem;
        margin-top: 2px;
        margin-bottom: 0;
    }

    /* ====================================================================
       KPI CARDS
    ==================================================================== */
    .kpi-card {
        position: relative;
        background: #12151f;
        border: 1px solid #232838;
        border-radius: 16px;
        padding: 20px 22px;
        overflow: hidden;
        transition: border-color 0.15s ease, transform 0.15s ease;
        height: 100%;
    }
    .kpi-card:hover {
        border-color: #3a4264;
        transform: translateY(-2px);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6c63ff, #5dd0ff);
    }
    .kpi-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #8b93a7;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-icon { font-size: 1.1rem; opacity: 0.7; }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f3f5fa;
        line-height: 1.1;
        letter-spacing: -0.01em;
    }
    .kpi-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-top: 10px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .pill-positive {
        background: rgba(53, 208, 127, 0.14);
        color: #35d07f;
        border: 1px solid rgba(53, 208, 127, 0.30);
    }
    .pill-negative {
        background: rgba(255, 92, 122, 0.14);
        color: #ff5c7a;
        border: 1px solid rgba(255, 92, 122, 0.30);
    }
    .pill-neutral {
        background: rgba(140, 148, 170, 0.14);
        color: #aab2c8;
        border: 1px solid rgba(140, 148, 170, 0.30);
    }

    .positive { color: #35d07f; }
    .negative { color: #ff5c7a; }
    .neutral  { color: #aab2c8; }
    .accent   { color: #8a93ff; }

    /* ====================================================================
       BIG PREDICTION CARD
    ==================================================================== */
    .prediction-box {
        position: relative;
        border-radius: 18px;
        padding: 32px;
        text-align: center;
        border: 1px solid #232838;
        background: radial-gradient(circle at 50% 0%, #1a1f2e 0%, #12151f 70%);
        overflow: hidden;
        height: 100%;
    }
    .prediction-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6c63ff, #5dd0ff);
    }
    .prediction-arrow {
        font-size: 4.2rem;
        font-weight: 900;
        line-height: 1;
        margin: 10px 0;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    /* ====================================================================
       STOCK PICKER GRID (logos)
    ==================================================================== */
    .stock-grid-card {
        background: #12151f;
        border: 1px solid #232838;
        border-radius: 14px;
        padding: 14px;
        text-align: center;
        transition: border-color 0.15s ease, transform 0.15s ease;
        height: 100%;
    }
    .stock-grid-card:hover {
        border-color: #6c63ff;
        transform: translateY(-2px);
    }
    .stock-grid-logo {
        width: 40px;
        height: 40px;
        object-fit: contain;
        background: #ffffff;
        border-radius: 8px;
        padding: 4px;
        margin-bottom: 8px;
    }
    .stock-grid-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: #e9ebf1;
    }
    .stock-grid-symbol {
        font-size: 0.72rem;
        color: #6b7388;
    }

    /* ====================================================================
       STOCK PICKER BUTTONS
       Style Streamlit's native st.button() to look like stock cards.
       Use broad selectors + !important since Streamlit's internal class
       names vary between versions.
    ==================================================================== */
    .stButton button,
    .stButton button p,
    .stButton button span,
    .stButton button div {
        color: #e9ebf1 !important;
    }
    /* ====================================================================
       MOBILE RESPONSIVE
    ==================================================================== */
    @media (max-width: 768px) {
        .hero-banner {
            height: 100px;
        }
        .hero-content h1 {
            font-size: 1.3rem;
        }
        .kpi-value {
            font-size: 1.4rem;
        }
    }
    .stButton button {
        width: 100% !important;
        background-color: #12151f !important;
        border: 1px solid #232838 !important;
        border-radius: 14px !important;
        padding: 10px 8px !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        line-height: 1.4 !important;
        white-space: pre-line !important;
        transition: border-color 0.15s ease, transform 0.15s ease !important;
        box-shadow: none !important;
    }
    .stButton button:hover {
        border-color: #6c63ff !important;
        transform: translateY(-2px);
    }
    .stButton button:hover p,
    .stButton button:hover span,
    .stButton button:hover div {
        color: #f3f5fa !important;
    }
    .stButton button:focus {
        box-shadow: none !important;
        border-color: #6c63ff !important;
    }
    .picker-active .stButton button {
        border-color: #6c63ff !important;
        background-color: #181c2c !important;
    }

    /* ====================================================================
       SIDEBAR BANNER
       A dark abstract image with a gradient fade-out at the bottom so it
       blends seamlessly into the sidebar's solid background color
       (#11141c), rather than showing a hard rectangular edge.
    ==================================================================== */
    .sidebar-banner {
        position: relative;
        width: calc(100% + 0px);
        height: 130px;
        margin: -1rem -1rem 14px -1rem;
        border-radius: 0;
        overflow: hidden;
        background-image:
            linear-gradient(180deg, rgba(17,20,28,0.15) 0%, rgba(17,20,28,1) 100%),
            url('https://images.unsplash.com/photo-1639762681485-074b7f938ba0?auto=format&fit=crop&w=600&q=60');
        background-size: cover;
        background-position: center;
    }
    .sidebar-banner::after {
        content: "";
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 60px;
        background: linear-gradient(180deg, rgba(17,20,28,0) 0%, rgba(17,20,28,1) 100%);
    }

    /* ====================================================================
       HEADINGS + TABS
    ==================================================================== */
    h1, h2, h3 { color: #f3f5fa; letter-spacing: -0.01em; }
    h3 { font-size: 1.15rem; font-weight: 700; margin-top: 0.4rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #232838;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        font-weight: 600;
        color: #8b93a7;
    }
    .stTabs [aria-selected="true"] {
        color: #f3f5fa !important;
        border-bottom: 2px solid #6c63ff !important;
    }

    .section-spacer { margin-top: 0.25rem; margin-bottom: 0.25rem; }
    hr { border-color: #232838 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0b0e14",
    plot_bgcolor="#0b0e14",
    font=dict(color="#e9ebf1", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=30),
)


# ==========================================================================
# 3. UI HELPER FUNCTIONS
# ==========================================================================

def kpi_card(label, value, sub_text=None, sub_class="neutral", icon=""):
    """Build the HTML for a single premium KPI card."""
    pill_class = {
        "positive": "pill-positive",
        "negative": "pill-negative",
        "neutral": "pill-neutral",
    }.get(sub_class, "pill-neutral")

    pill_html = f'<div class="kpi-pill {pill_class}">{sub_text}</div>' if sub_text else ""
    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ""

    return f"""
    <div class="kpi-card fade-in">
        <div class="kpi-top-row">
            <span class="kpi-label">{label}</span>
            {icon_html}
        </div>
        <div class="kpi-value">{value}</div>
        {pill_html}
    </div>
    """


# ==========================================================================
# 4. API HELPER FUNCTIONS
# ==========================================================================

def api_get(path, params=None):
    """Generic GET helper for the FastAPI backend."""
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 401:
            return None, "Unauthorized: check your API_KEY."
        if resp.status_code == 404:
            return None, "No data found (404)."
        if resp.status_code == 503:
            try:
                detail = resp.json().get("detail", "")
            except ValueError:
                detail = ""
            return None, f"Service unavailable: {detail}"

        resp.raise_for_status()
        return resp.json(), None

    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to the API at {API_BASE_URL}. Is the backend running?"
    except requests.exceptions.Timeout:
        return None, "The API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"API request failed: {e}"
    except ValueError:
        return None, "API returned an invalid (non-JSON) response."


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def get_latest_price(ticker):
    return api_get(f"/stocks/{ticker}")


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def get_history(ticker, days):
    return api_get(f"/stocks/{ticker}/history", params={"days": days})


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def get_prediction(ticker):
    return api_get(f"/stocks/{ticker}/predict")


# ==========================================================================
# 5. SIDEBAR
# ==========================================================================

# Initialize the selected stock in session_state -- the Quick Stock Picker
# grid (rendered later, in the main area) is the only way to change this.
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = list(STOCKS.keys())[0]

with st.sidebar:
    st.markdown(
        '<div class="sidebar-banner"></div>',
        unsafe_allow_html=True,
    )

    st.markdown("## 📈 Stock Intelligence")
    st.markdown(
        "<span class='neutral'>Real-time data & ML predictions "
        "for top Indian stocks.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Show which stock is currently selected (read-only here -- pick a
    # different one via the Quick Stock Picker grid below).
    current_info = STOCKS[st.session_state.selected_ticker]
    st.markdown(
        f"**Currently viewing:**  \n{current_info['name']} "
        f"({st.session_state.selected_ticker})"
    )

    selected_ticker = st.session_state.selected_ticker

    st.markdown("---")

    today = date.today()
    default_start = today - timedelta(days=90)

    date_range = st.date_input(
        "Date Range (for charts)",
        value=(default_start, today),
        max_value=today,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, today

    days_requested = max((end_date - start_date).days, 1)
    days_requested = min(days_requested, 365)

    st.markdown("---")

    auto_refresh = st.toggle(
        f"🔄 Auto-refresh every {AUTO_REFRESH_SECONDS}s",
        value=False,
        help="When enabled, the dashboard automatically refetches data "
             "every 30 seconds.",
    )

    st.markdown("---")
    st.caption(f"Backend: `{API_BASE_URL}`")
    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")


if auto_refresh:
    time.sleep(AUTO_REFRESH_SECONDS)
    st.rerun()


# ==========================================================================
# 6. HERO BANNER
# ==========================================================================

st.markdown(
    """
    <div class="hero-banner fade-in">
        <div class="hero-content">
            <h1>📈 Stock Market Intelligence Platform</h1>
            <p>Real-time prices, technical indicators, and machine-learning
            powered next-day direction predictions for 10 of India's
            largest companies.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# 7. STOCK PICKER GRID (with logos)
# ==========================================================================

st.markdown("#### Quick Stock Picker")
st.caption("Click a company to view its dashboard below.")

grid_cols = st.columns(5)
for i, (sym, info) in enumerate(STOCKS.items()):
    with grid_cols[i % 5]:
        is_active = (sym == st.session_state.selected_ticker)
        wrapper_class = "picker-active" if is_active else ""

        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        clicked = st.button(
            f"{info['name']}\n{sym}",
            key=f"picker_{sym}",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            st.session_state.selected_ticker = sym
            st.rerun()

# Re-read selected_ticker in case a picker button just changed it
selected_ticker = st.session_state.selected_ticker

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
st.markdown("---")


# ==========================================================================
# 8. PAGE HEADER (selected stock, with logo)
# ==========================================================================

selected_info = STOCKS[selected_ticker]

st.markdown(
    f"""
    <div class="fade-in">
        <div class="brand-header">
            <img class="brand-logo" src="{logo_url(selected_ticker)}"
                 onerror="this.style.display='none'" />
            <div>
                <div style="display:flex; align-items:baseline; gap:14px;">
                    <h1>{selected_info['name']}</h1>
                    <span class="brand-ticker">{selected_ticker}</span>
                </div>
                <p class="brand-sub">Live price, technical indicators & ML predictions</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# 9. KPI CARDS ROW (Price, ML Prediction, Volume)
# ==========================================================================

kpi_cols = st.columns(3)

# --- KPI 1: Current Price + % change ---
with kpi_cols[0]:
    with st.spinner("Loading price..."):
        latest, err = get_latest_price(selected_ticker)

    if err or latest is None:
        st.markdown(
            kpi_card("Current Price", "N/A", err or "No data", "neutral", "💰"),
            unsafe_allow_html=True,
        )
    else:
        hist_small, hist_err = get_history(selected_ticker, days=2)
        pct_change = None
        if not hist_err and hist_small and len(hist_small.get("data", [])) == 2:
            prev_close = hist_small["data"][0]["close"]
            curr_close = hist_small["data"][1]["close"]
            if prev_close:
                pct_change = ((curr_close - prev_close) / prev_close) * 100

        price = latest.get("close")
        sub_class = "neutral"
        sub_text = "—"
        if pct_change is not None:
            sub_class = "positive" if pct_change >= 0 else "negative"
            arrow = "▲" if pct_change >= 0 else "▼"
            sub_text = f"{arrow} {pct_change:+.2f}%"

        st.markdown(
            kpi_card("Current Price (₹)", f"₹{price:,.2f}", sub_text, sub_class, "💰"),
            unsafe_allow_html=True,
        )


# --- KPI 2: ML Prediction (Up/Down arrow + confidence) ---
with kpi_cols[1]:
    with st.spinner("Loading prediction..."):
        prediction, pred_err = get_prediction(selected_ticker)

    if pred_err or prediction is None:
        st.markdown(
            kpi_card("ML Prediction (Next Day)", "N/A", pred_err or "Model not available", "neutral", "🤖"),
            unsafe_allow_html=True,
        )
    else:
        direction = prediction.get("prediction", "—")
        confidence_pct = prediction.get("confidence_pct", "—")
        if direction == "Up":
            arrow, sub_class = "▲", "positive"
        else:
            arrow, sub_class = "▼", "negative"

        st.markdown(
            kpi_card("ML Prediction (Next Day)", f"{arrow} {direction}",
                     f"Confidence: {confidence_pct}", sub_class, "🤖"),
            unsafe_allow_html=True,
        )


# --- KPI 3: Trading Volume ---
with kpi_cols[2]:
    if err or latest is None:
        st.markdown(
            kpi_card("Volume", "N/A", "No data", "neutral", "📊"),
            unsafe_allow_html=True,
        )
    else:
        volume = latest.get("volume") or 0
        if volume >= 1_000_000:
            vol_display = f"{volume / 1_000_000:.2f}M"
        elif volume >= 1_000:
            vol_display = f"{volume / 1_000:.1f}K"
        else:
            vol_display = str(volume)

        st.markdown(
            kpi_card("Volume", vol_display, "Shares traded", "neutral", "📊"),
            unsafe_allow_html=True,
        )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)


# ==========================================================================
# 10. MAIN TABS: Overview | ML Insights | Compare Stocks
# ==========================================================================

tab_overview, tab_ml, tab_compare = st.tabs(
    ["📊 Overview", "🤖 ML Insights", "⚖️ Compare Stocks"]
)


# --------------------------------------------------------------------
# TAB 1: OVERVIEW -- candlestick, moving averages, RSI, MACD
# --------------------------------------------------------------------
with tab_overview:

    with st.spinner("Loading historical data..."):
        history, hist_err = get_history(selected_ticker, days=days_requested)

    if hist_err or history is None or not history.get("data"):
        st.error(f"⚠️ Could not load historical data: {hist_err or 'No data returned.'}")
    else:
        df = pd.DataFrame(history["data"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()

        st.subheader("Price Chart (Candlestick + Volume)")

        fig_candle = go.Figure()

        fig_candle.add_trace(go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="Price",
            increasing_line_color="#35d07f",
            decreasing_line_color="#ff5c7a",
        ))

        fig_candle.add_trace(go.Scatter(
            x=df["date"], y=df["sma_20"], name="SMA 20",
            line=dict(color="#8a93ff", width=1.5),
        ))
        fig_candle.add_trace(go.Scatter(
            x=df["date"], y=df["sma_50"], name="SMA 50",
            line=dict(color="#5dd0ff", width=1.5),
        ))

        fig_candle.add_trace(go.Bar(
            x=df["date"], y=df["volume"], name="Volume",
            marker_color="rgba(138, 147, 255, 0.25)",
            yaxis="y2",
        ))

        fig_candle.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_rangeslider_visible=False,
            yaxis=dict(title="Price (₹)"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )

        st.plotly_chart(fig_candle, use_container_width=True)

        # ------------------------------------------------------------
        # RSI Chart
        # ------------------------------------------------------------
        st.subheader("RSI (Relative Strength Index)")

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].where(avg_loss != 0, 100)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df["date"], y=df["rsi_14"], name="RSI (14)",
            line=dict(color="#8a93ff", width=2),
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff5c7a",
                          annotation_text="Overbought (70)", annotation_position="top left")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#35d07f",
                          annotation_text="Oversold (30)", annotation_position="bottom left")

        fig_rsi.update_layout(
            **PLOTLY_LAYOUT,
            yaxis=dict(title="RSI", range=[0, 100]),
            height=300,
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # ------------------------------------------------------------
        # MACD Chart
        # ------------------------------------------------------------
        st.subheader("MACD (Moving Average Convergence Divergence)")

        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Bar(
            x=df["date"], y=df["macd_hist"], name="Histogram",
            marker_color=np.where(df["macd_hist"] >= 0, "#35d07f", "#ff5c7a"),
        ))
        fig_macd.add_trace(go.Scatter(
            x=df["date"], y=df["macd"], name="MACD",
            line=dict(color="#5dd0ff", width=2),
        ))
        fig_macd.add_trace(go.Scatter(
            x=df["date"], y=df["macd_signal"], name="Signal",
            line=dict(color="#8a93ff", width=2),
        ))

        fig_macd.update_layout(
            **PLOTLY_LAYOUT,
            yaxis=dict(title="MACD"),
            height=300,
        )
        st.plotly_chart(fig_macd, use_container_width=True)

        st.caption(
            "ℹ️ RSI and MACD shown here are computed live from the loaded "
            "price history for the selected date range."
        )


# --------------------------------------------------------------------
# TAB 2: ML INSIGHTS -- prediction card + feature importance chart
# --------------------------------------------------------------------
with tab_ml:

    st.subheader("ML Price Direction Prediction")

    with st.spinner("Loading ML prediction..."):
        prediction, pred_err = get_prediction(selected_ticker)

    if pred_err or prediction is None:
        st.warning(
            f"⚠️ ML prediction unavailable: {pred_err or 'Unknown error.'}\n\n"
            f"Make sure `ml_model.py train` has been run on the backend."
        )
    else:
        direction = prediction.get("prediction", "—")
        confidence = prediction.get("confidence", 0)
        confidence_pct = prediction.get("confidence_pct", "—")
        as_of = prediction.get("as_of_date", "—")

        if direction == "Up":
            arrow, css_class = "▲", "positive"
        else:
            arrow, css_class = "▼", "negative"

        pred_col, gauge_col = st.columns([1, 1])

        with pred_col:
            pill_variant = "pill-positive" if css_class == "positive" else "pill-negative"
            st.markdown(
                f"""
                <div class="prediction-box fade-in">
                    <div class="kpi-label">Next-Day Direction Prediction</div>
                    <div class="prediction-arrow {css_class}">{arrow}</div>
                    <div class="kpi-value {css_class}" style="margin-top:10px;">{direction}</div>
                    <div class="kpi-pill {pill_variant}" style="margin-top:14px;">
                        Confidence: {confidence_pct}
                    </div>
                    <div class="neutral" style="font-size:0.8rem; margin-top:14px;">
                        As of {as_of}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={"text": "Model Confidence (%)"},
                number={"suffix": "%", "font": {"color": "#e9ebf1"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b93a7"},
                    "bar": {"color": "#8a93ff"},
                    "bgcolor": "#12151f",
                    "borderwidth": 1,
                    "bordercolor": "#232838",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(255, 92, 122, 0.15)"},
                        {"range": [50, 75], "color": "rgba(170, 178, 200, 0.10)"},
                        {"range": [75, 100], "color": "rgba(53, 208, 127, 0.15)"},
                    ],
                },
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Feature Importance (Illustrative)")
    st.caption(
        "ℹ️ This chart shows the relative importance of each feature in the "
        "trained XGBoost model, based on typical importances from "
        "ml_model.py."
    )

    illustrative_importance = pd.DataFrame({
        "feature": [
            "rsi_14", "macd_hist", "close_vs_sma20", "avg_sentiment_score",
            "lag_return_1", "roll_std_5", "close_vs_bb_lower", "macd",
            "volume_change", "lag_return_2", "sma_20", "close_vs_sma50",
            "roll_mean_5", "lag_close_1", "bb_upper",
        ],
        "importance": [
            0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06,
            0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01,
        ],
    }).sort_values("importance", ascending=True)

    fig_importance = go.Figure(go.Bar(
        x=illustrative_importance["importance"],
        y=illustrative_importance["feature"],
        orientation="h",
        marker_color="#8a93ff",
    ))
    fig_importance.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(title="Relative Importance"),
        yaxis=dict(title=""),
        height=450,
    )
    st.plotly_chart(fig_importance, use_container_width=True)


# --------------------------------------------------------------------
# TAB 3: COMPARE STOCKS -- side-by-side comparison + correlation heatmap
# --------------------------------------------------------------------
with tab_compare:

    st.subheader("Compare Two Stocks")

    compare_cols = st.columns(2)
    symbol_options = list(STOCKS.keys())

    with compare_cols[0]:
        stock_a = st.selectbox(
            "Stock A",
            options=symbol_options,
            format_func=lambda s: f"{STOCKS[s]['name']} ({s})",
            index=symbol_options.index(selected_ticker),
            key="compare_stock_a",
        )
    with compare_cols[1]:
        default_b_index = 1 if symbol_options[0] == stock_a else 0
        stock_b = st.selectbox(
            "Stock B",
            options=symbol_options,
            format_func=lambda s: f"{STOCKS[s]['name']} ({s})",
            index=default_b_index,
            key="compare_stock_b",
        )

    with st.spinner("Loading comparison data..."):
        hist_a, err_a = get_history(stock_a, days=days_requested)
        hist_b, err_b = get_history(stock_b, days=days_requested)
        pred_a, perr_a = get_prediction(stock_a)
        pred_b, perr_b = get_prediction(stock_b)

    if err_a or err_b or not hist_a.get("data") or not hist_b.get("data"):
        st.error("⚠️ Could not load historical data for one or both stocks.")
    else:
        df_a = pd.DataFrame(hist_a["data"])
        df_b = pd.DataFrame(hist_b["data"])
        df_a["date"] = pd.to_datetime(df_a["date"])
        df_b["date"] = pd.to_datetime(df_b["date"])

        st.markdown("#### Normalized Price Performance (Rebased to 100)")

        norm_a = df_a["close"] / df_a["close"].iloc[0] * 100
        norm_b = df_b["close"] / df_b["close"].iloc[0] * 100

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=df_a["date"], y=norm_a, name=f"{STOCKS[stock_a]['name']} ({stock_a})",
            line=dict(color="#5dd0ff", width=2),
        ))
        fig_compare.add_trace(go.Scatter(
            x=df_b["date"], y=norm_b, name=f"{STOCKS[stock_b]['name']} ({stock_b})",
            line=dict(color="#8a93ff", width=2),
        ))
        fig_compare.update_layout(
            **PLOTLY_LAYOUT,
            yaxis=dict(title="Normalized Price (Base = 100)"),
            height=400,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("#### Side-by-Side Snapshot")
        col_a, col_b = st.columns(2)

        def render_snapshot(col, symbol, df, pred, pred_err_):
            with col:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <img src="{logo_url(symbol)}" style="width:32px; height:32px; object-fit:contain; background:#fff; border-radius:6px; padding:3px;" onerror="this.style.display='none'" />
                        <strong>{STOCKS[symbol]['name']} ({symbol})</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                latest_close = df["close"].iloc[-1]
                first_close = df["close"].iloc[0]
                period_change = ((latest_close - first_close) / first_close) * 100

                change_class = "positive" if period_change >= 0 else "negative"
                st.markdown(
                    f"**Price:** ₹{latest_close:,.2f} &nbsp; "
                    f"<span class='{change_class}'>({period_change:+.2f}% over period)</span>",
                    unsafe_allow_html=True,
                )

                if not pred_err_ and pred:
                    direction = pred.get("prediction", "—")
                    conf = pred.get("confidence_pct", "—")
                    css_class = "positive" if direction == "Up" else "negative"
                    arrow = "▲" if direction == "Up" else "▼"
                    st.markdown(
                        f"**ML Prediction:** <span class='{css_class}'>{arrow} {direction}</span> "
                        f"(Confidence: {conf})",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("**ML Prediction:** _Unavailable_")

        render_snapshot(col_a, stock_a, df_a, pred_a, perr_a)
        render_snapshot(col_b, stock_b, df_b, pred_b, perr_b)

    st.markdown("---")
    st.markdown("#### Correlation Heatmap (All 10 Stocks)")
    st.caption(
        "ℹ️ Correlation of daily returns over the selected date range. "
        "Values closer to +1 (purple) mean two stocks tend to move "
        "together; values closer to -1 mean they move oppositely."
    )

    with st.spinner("Loading data for all stocks (this may take a moment)..."):
        close_series = {}
        load_errors = []

        for sym in STOCKS.keys():
            hist, h_err = get_history(sym, days=days_requested)
            if h_err or not hist or not hist.get("data"):
                load_errors.append(sym)
                continue

            df_sym = pd.DataFrame(hist["data"])
            df_sym["date"] = pd.to_datetime(df_sym["date"])
            df_sym = df_sym.set_index("date")["close"]
            close_series[sym] = df_sym

    if len(close_series) < 2:
        st.warning("⚠️ Not enough data available across stocks to build a correlation heatmap.")
    else:
        price_df = pd.DataFrame(close_series)
        returns_df = price_df.pct_change().dropna()

        corr_matrix = returns_df.corr()

        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale=["#ff5c7a", "#12151f", "#8a93ff"],
            zmin=-1, zmax=1,
            aspect="auto",
        )
        fig_heatmap.update_layout(
            **PLOTLY_LAYOUT,
            height=550,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        if load_errors:
            st.caption(f"⚠️ Could not load data for: {', '.join(load_errors)}")


# ==========================================================================
# 11. FOOTER
# ==========================================================================
st.markdown("---")
st.caption(
    "📈 Real-Time Stock Market Intelligence Platform — "
    "Data via FastAPI backend | Charts powered by Plotly | "
    "Built with Streamlit"
)
