import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


st.set_page_config(
    page_title="Cryptocurrency Market Trend Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#5A8AC9"
SECONDARY_COLOR = "#1D4ED8"
ACCENT_COLOR = "#0EA5A4"
HIGHLIGHT_COLOR = "#08111F"
SURFACE_COLOR = "#F4F7FB"
PLOT_BACKGROUND = "#FFFFFF"
PAPER_BACKGROUND = "#F8FBFF"
API_BASE_URL = "https://api.coingecko.com/api/v3"
DEFAULT_DAYS = 180
SUPPORTED_COINS = {
    "bitcoin": {"label": "Bitcoin", "symbol": "BTC"},
    "ethereum": {"label": "Ethereum", "symbol": "ETH"},
    "litecoin": {"label": "Litecoin", "symbol": "LTC"},
    "ripple": {"label": "Ripple (XRP)", "symbol": "XRP"},
    "cardano": {"label": "Cardano", "symbol": "ADA"},
    "solana": {"label": "Solana", "symbol": "SOL"},
    "dogecoin": {"label": "Dogecoin", "symbol": "DOGE"},
    "polkadot": {"label": "Polkadot", "symbol": "DOT"},
    "chainlink": {"label": "Chainlink", "symbol": "LINK"},
    "stellar": {"label": "Stellar", "symbol": "XLM"},
}


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #0F2747;
    --secondary: #1D4ED8;
    --accent: #0EA5A4;
    --accent-soft: #D9FBFA;
    --background: #EEF3F9;
    --surface: rgba(255, 255, 255, 0.88);
    --surface-strong: #FFFFFF;
    --surface-soft: #F6F9FD;
    --panel-dark: #091426;
    --panel-muted: #13223A;
    --text-primary: #0B1220;
    --text-secondary: #324256;
    --text-muted: #61748B;
    --text-soft: #7B8CA3;
    --border-soft: rgba(148, 163, 184, 0.22);
    --border-strong: rgba(100, 116, 139, 0.24);
    --shadow-soft: 0 18px 40px rgba(15, 23, 42, 0.08);
    --shadow-card: 0 28px 60px rgba(15, 23, 42, 0.12);
    --shadow-hero: 0 30px 80px rgba(8, 17, 31, 0.24);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(14, 165, 164, 0.14), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, var(--background) 40%, #edf2f8 100%);
    color: var(--text-primary);
    font-family: 'Manrope', sans-serif;
}

body {
    background-color: var(--background);
    color: var(--text-primary);
    font-family: 'Manrope', sans-serif;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

.main .block-container,
.stApp,
.stApp div,
.stApp p,
.stApp span,
.stApp label,
.stMarkdown,
.stText,
.stCaption,
.st-emotion-cache-16txtl3,
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary);
    font-family: 'Manrope', sans-serif;
}

.stApp .stMarkdown p,
.stApp .stMarkdown li,
.stApp .stMarkdown div,
.stApp .stCaption,
.stApp small,
.stApp ul,
.stApp ol {
    color: var(--text-secondary);
}

.stApp a {
    color: var(--secondary) !important;
}

[data-testid="stVerticalBlock"] > div:has(> .info-card),
[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
    color: var(--text-primary);
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(7, 17, 33, 0.98) 0%, rgba(16, 36, 66, 0.98) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown div,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

.sidebar-brand {
    padding: 1rem 1rem 1.15rem 1rem;
    border-radius: 24px;
    background:
        linear-gradient(160deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04)),
        radial-gradient(circle at top right, rgba(14,165,164,0.28), transparent 32%);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}

.sidebar-eyebrow {
    display: inline-block;
    padding: 0.28rem 0.62rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.1);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.68rem;
    font-weight: 700;
    color: #D7E3F4 !important;
    margin-bottom: 0.7rem;
}

.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 0.35rem 0;
}

.sidebar-copy {
    margin: 0;
    color: #CAD7E6 !important;
    font-size: 0.92rem;
    line-height: 1.65;
}

section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.6rem;
}

section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 16px;
    padding: 0.7rem 0.85rem;
    border: 1px solid rgba(255,255,255,0.08) !important;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(255,255,255,0.16) !important;
}

section[data-testid="stSidebar"] .stRadio input:checked + div {
    background: linear-gradient(135deg, #1D4ED8, #0EA5A4) !important;
    color: white !important;
    border-radius: 12px;
    padding: 0.15rem 0.35rem;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption p {
    color: #BFD0E3 !important;
}

.hero-card {
    position: relative;
    overflow: hidden;
    padding: 2rem;
    border-radius: 30px;
    background:
        radial-gradient(circle at top right, rgba(14, 165, 164, 0.28), transparent 26%),
        linear-gradient(135deg, rgba(8,17,31,0.98) 0%, rgba(15,39,71,0.97) 58%, rgba(29,78,216,0.92) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: var(--shadow-hero);
    margin-bottom: 1rem;
}

.hero-card::before {
    content: "";
    position: absolute;
    inset: auto -4rem -4rem auto;
    width: 18rem;
    height: 18rem;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(255,255,255,0.14), transparent 68%);
}

.hero-shell {
    position: relative;
    display: flex;
    justify-content: space-between;
    gap: 1.5rem;
    align-items: flex-start;
}

.hero-copy {
    max-width: 54rem;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    color: #DCE8F7 !important;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.95rem;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1.02;
    letter-spacing: -0.03em;
    color: #FFFFFF !important;
    margin: 0 0 0.85rem 0;
}

.hero-description {
    margin: 0;
    max-width: 46rem;
    color: #D6E4F5 !important;
    font-size: 1rem;
    line-height: 1.8;
}

.hero-panel {
    min-width: 15.5rem;
    padding: 1rem 1.1rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}

.hero-panel-label {
    color: #AFC5DD !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.hero-panel-value {
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
}

.hero-panel-copy {
    color: #CFE0F1 !important;
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0;
}

.info-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,249,253,0.98));
    border: 1px solid var(--border-soft);
    border-radius: 24px;
    padding: 1.15rem 1.2rem;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(10px);
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-bottom: 0.55rem;
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,248,252,0.96));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 22px;
    padding: 1rem 1rem;
    box-shadow: var(--shadow-soft);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.03em;
}

.stSubheader,
[data-testid="stHeading"] {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif;
}

.stSelectbox label,
.stMultiSelect label,
.stDateInput label,
.stSlider label,
.stRadio label {
    color: var(--text-primary) !important;
    font-weight: 600;
}

.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
.stDateInput [data-baseweb="input"] > div,
.stDateInput input,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: rgba(255,255,255,0.95) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.03) !important;
}

.stMultiSelect [data-baseweb="tag"] {
    background: #E7FBFA !important;
    color: #0B1220 !important;
    border: 1px solid #B7F0EE !important;
}

[data-baseweb="popover"] *,
[role="listbox"] *,
[data-baseweb="menu"] * {
    color: var(--text-primary) !important;
    background-color: #ffffff !important;
}

.stSelectbox [data-baseweb="select"] span,
.stMultiSelect [data-baseweb="select"] span,
.stDateInput input,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    color: var(--text-primary) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, var(--secondary), var(--accent)) !important;
    border: 2px solid #ffffff !important;
}

.stSlider [data-baseweb="slider"] > div > div {
    color: var(--text-primary) !important;
}

.stRadio [role="radiogroup"] label {
    background: rgba(255,255,255,0.88) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 16px;
    margin-bottom: 0.35rem;
    padding: 0.5rem 0.7rem;
}

.stRadio [role="radiogroup"] label:hover {
    background: #FFFFFF !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 0.4rem;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--secondary), #2563EB);
    color: #ffffff !important;
    border: 1px solid rgba(29, 78, 216, 0.28);
    border-radius: 16px;
    font-weight: 700;
    min-height: 2.95rem;
    transition: all 0.24s ease;
    box-shadow: 0 16px 32px rgba(29, 78, 216, 0.18);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 36px rgba(29, 78, 216, 0.22);
}

.stDataFrame, .stTable {
    background: rgba(255,255,255,0.92);
    border: 1px solid var(--border-soft);
    border-radius: 22px;
    box-shadow: var(--shadow-soft);
}

.stAlert,
[data-testid="stNotificationContentInfo"],
[data-testid="stNotificationContentSuccess"],
[data-testid="stNotificationContentWarning"],
[data-testid="stNotificationContentError"] {
    color: var(--text-primary) !important;
}

.stAlert p,
.stAlert div,
.stAlert span {
    color: var(--text-primary) !important;
}

[data-testid="stDataFrame"] *,
.stTable * {
    color: var(--text-primary) !important;
}

[data-testid="stMarkdownContainer"] strong {
    color: var(--text-primary) !important;
}

.small-note {
    color: var(--text-secondary) !important;
    font-size: 0.93rem;
    line-height: 1.7;
}

[data-testid="stPlotlyChart"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,249,253,0.96));
    border: 1px solid var(--border-soft);
    border-radius: 24px;
    box-shadow: var(--shadow-card);
    padding: 0.55rem 0.65rem 0.3rem 0.65rem;
}

[data-testid="stDataFrame"] {
    border-radius: 24px;
    overflow: hidden;
}

.stAlert {
    border-radius: 18px;
    border: 1px solid var(--border-soft);
    background: rgba(255,255,255,0.94);
    box-shadow: var(--shadow-soft);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

.dashboard-note {
    padding: 1rem 1.05rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.08);
}

.dashboard-note strong {
    color: #FFFFFF !important;
}

@media (max-width: 980px) {
    .hero-shell {
        flex-direction: column;
    }

    .hero-panel {
        min-width: 100%;
    }
}

</style>
"""


def inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=900)
def get_crypto_data(coin: str = "bitcoin", days: int = 180, refresh_token: int = 0) -> pd.DataFrame:
    """Fetch live cryptocurrency market data from CoinGecko."""
    if coin not in SUPPORTED_COINS:
        raise ValueError(f"Unsupported cryptocurrency: {coin}")

    endpoint = f"{API_BASE_URL}/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": int(days), "interval": "daily"}

    try:
        time.sleep(1)
        response = requests.get(endpoint, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"CoinGecko request failed for {coin}: {exc}") from exc

    prices = pd.DataFrame(payload.get("prices", []), columns=["timestamp", "price"])
    market_caps = pd.DataFrame(payload.get("market_caps", []), columns=["timestamp", "market_cap"])
    volumes = pd.DataFrame(payload.get("total_volumes", []), columns=["timestamp", "volume"])

    if prices.empty:
        raise ValueError(f"No market data returned for {coin}")

    data = prices.merge(market_caps, on="timestamp", how="left").merge(volumes, on="timestamp", how="left")
    data["date"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    data["coin_id"] = coin
    data["cryptocurrency"] = SUPPORTED_COINS[coin]["label"]
    data["symbol"] = SUPPORTED_COINS[coin]["symbol"]
    data = data[["date", "price", "market_cap", "volume", "cryptocurrency", "symbol"]]
    data = data.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["market_cap"] = pd.to_numeric(data["market_cap"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    data["percent_change_24h"] = data["price"].pct_change() * 100
    data = data.ffill().bfill()
    return data


@st.cache_data(show_spinner=False, ttl=900)
def get_multi_coin_data(coins: tuple[str, ...], days: int = 180) -> pd.DataFrame:
    frames = [get_crypto_data(coin=coin, days=days) for coin in coins]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["cryptocurrency", "date"]).reset_index(drop=True)
    return data


def live_data_error_message() -> None:
    st.error("API limit reached. Try again later.")


def fetch_live_data_with_fallback(coin: str, days: int, refresh_token: int = 0) -> pd.DataFrame:
    cache_key = f"{coin}_{days}_{refresh_token}"
    session_cache = st.session_state.setdefault("live_data_cache", {})

    try:
        data = get_crypto_data(coin=coin, days=days, refresh_token=refresh_token)
        session_cache[cache_key] = data.copy()
        return data
    except Exception:
        if cache_key in session_cache:
            st.warning("CoinGecko rate limit reached. Showing the most recently cached live data for this selection.")
            return session_cache[cache_key].copy()
        raise


def create_sequences(data: np.ndarray, seq_length: int = 60) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


@st.cache_resource(show_spinner=False)
def train_lstm_model(prices: tuple[float, ...], seq_length: int = 60, epochs: int = 4, batch_size: int = 32):
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow/Keras is not installed. Install tensorflow and keras to use the LSTM prediction module."
        ) from exc

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    price_array = np.array(prices, dtype=np.float32).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_array)

    X, y = create_sequences(scaled_data, seq_length=seq_length)
    if len(X) == 0:
        raise ValueError(f"At least {seq_length + 1} price records are required to train the LSTM model.")

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(25),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=epochs, batch_size=min(batch_size, len(X)), verbose=0)

    predicted_scaled = model.predict(X, verbose=0)
    predicted_prices = scaler.inverse_transform(predicted_scaled).flatten()
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    predicted_prices = np.maximum(predicted_prices, 0)
    actual_prices = np.maximum(actual_prices, 0)

    return {
        "model": model,
        "scaler": scaler,
        "scaled_data": scaled_data,
        "predicted_prices": predicted_prices,
        "actual_prices": actual_prices,
    }


def forecast_future_prices(model, scaler, scaled_data: np.ndarray, seq_length: int, forecast_days: int) -> np.ndarray:
    rolling_window = scaled_data.flatten().tolist()
    future_scaled_predictions = []

    for _ in range(forecast_days):
        input_window = np.array(rolling_window[-seq_length:], dtype=np.float32).reshape(1, seq_length, 1)
        next_scaled = float(model.predict(input_window, verbose=0)[0][0])
        future_scaled_predictions.append(next_scaled)
        rolling_window.append(next_scaled)

    forecast = scaler.inverse_transform(np.array(future_scaled_predictions).reshape(-1, 1)).flatten()
    return np.maximum(forecast, 0)


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"${value:,.2f}"


def render_hero(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-shell">
                <div class="hero-copy">
                    <div class="hero-eyebrow">Live Intelligence Suite</div>
                    <h1 class="hero-title">{title}</h1>
                    <p class="hero-description">{description}</p>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-label">Platform Focus</div>
                    <div class="hero-panel-value">Market Clarity</div>
                    <p class="hero-panel-copy">
                        Analytics, forecasting, and live monitoring presented in a cleaner executive-dashboard format.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_date_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["date"].min().normalize(), df["date"].max().normalize()


def filter_by_date(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()


def build_price_figure(data: pd.DataFrame, title: str, color=None) -> go.Figure:
    fig = px.line(
        data,
        x="date",
        y="price",
        color=color,
        title=title,
    )
    fig.update_layout(legend_title_text="Cryptocurrency" if color else "")
    return style_figure(fig, height=420, xaxis_title="Date", yaxis_title="Price (USD)")


def style_figure(fig: go.Figure, height: int = 420, xaxis_title: str = "Date", yaxis_title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        margin=dict(l=20, r=20, t=72, b=24),
        font=dict(color="#213247", size=14, family="Manrope, sans-serif"),
        title_font=dict(size=22, color="#0B1220", family="Space Grotesk, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(148, 163, 184, 0.18)",
            borderwidth=1,
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hoverlabel=dict(
            bgcolor="#091426",
            font_color="#F8FBFF",
            bordercolor="rgba(255,255,255,0.16)",
            font_size=13,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(100, 116, 139, 0.26)",
        tickfont=dict(color="#5B6B81"),
        title_font=dict(color="#425368"),
    )
    fig.update_yaxes(
        gridcolor="rgba(29, 78, 216, 0.10)",
        zeroline=False,
        linecolor="rgba(100, 116, 139, 0.26)",
        tickfont=dict(color="#5B6B81"),
        title_font=dict(color="#425368"),
    )
    return fig


def render_overview_page() -> None:
    overview_days = 30

    control_col1, control_col2 = st.columns([1.2, 1])
    with control_col1:
        selected_coin = st.selectbox(
            "Overview Cryptocurrency",
            list(SUPPORTED_COINS.keys()),
            index=0,
            format_func=lambda coin_id: SUPPORTED_COINS[coin_id]["label"],
            key="overview_coin",
        )
    with control_col2:
        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        st.caption("Overview now uses a single live request to avoid CoinGecko rate limits.")

    try:
        with st.spinner("Loading live market overview..."):
            df = fetch_live_data_with_fallback(selected_coin, overview_days)
    except Exception:
        live_data_error_message()
        return

    latest_point = df.iloc[-1]
    previous_point = df.iloc[-2] if len(df) > 1 else latest_point
    latest_date = latest_point["date"]
    price_change = latest_point["price"] - previous_point["price"]
    selected_label = SUPPORTED_COINS[selected_coin]["label"]

    render_hero(
        "Cryptocurrency Market Trend Analysis Across Digital Assets",
        "A professional cryptocurrency analytics dashboard using live CoinGecko market data for exploring price behavior, market capitalization, trading activity, and model-driven forecasting across major digital assets.",
    )

    st.success("Dashboard loaded successfully with live market data.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Assets Supported", f"{len(SUPPORTED_COINS)}")
    metric_cols[1].metric("Latest Market Date", latest_date.strftime("%d %b %Y"))
    metric_cols[2].metric("Latest Price", format_currency(latest_point["price"]), f"{price_change:,.2f}")
    metric_cols[3].metric("Trading Volume", format_currency(latest_point["volume"]))

    st.divider()

    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.subheader("Market Snapshot")
        snapshot_view = df.tail(10)[
            ["date", "cryptocurrency", "symbol", "price", "market_cap", "volume", "percent_change_24h"]
        ].sort_values("date", ascending=False)
        st.dataframe(
            snapshot_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "market_cap": st.column_config.NumberColumn("Market Cap", format="$%.0f"),
                "volume": st.column_config.NumberColumn("Volume", format="$%.0f"),
                "percent_change_24h": st.column_config.NumberColumn("24h Change (%)", format="%.2f"),
            },
        )

    with right_col:
        st.subheader("Summary Insights")
        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-title">Key Highlights</div>
                <p class="small-note">
                    <strong>{selected_label}</strong> is currently trading at
                    <strong>{format_currency(latest_point['price'])}</strong>.
                </p>
                <p class="small-note">
                    Latest market capitalization stands at
                    <strong>{format_currency(latest_point['market_cap'])}</strong>.
                </p>
                <p class="small-note">
                    The latest 24-hour momentum is
                    <strong>{latest_point['percent_change_24h']:.2f}%</strong>,
                    with daily volume at <strong>{format_currency(latest_point['volume'])}</strong>.
                </p>
                <p class="small-note">
                    The dashboard combines descriptive analytics with predictive modeling to support market monitoring,
                    trend interpretation, and final-year project presentation use cases.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Live Price Trend")
        market_cap_fig = px.line(
            df,
            x="date",
            y="price",
            title=f"{selected_label} Price Trend",
        )
        market_cap_fig.update_traces(line=dict(color=ACCENT_COLOR, width=3))
        style_figure(market_cap_fig, height=420, xaxis_title="Date", yaxis_title="Price (USD)")
        market_cap_fig.update_layout(showlegend=False)
        st.plotly_chart(market_cap_fig, use_container_width=True)

    with chart_col2:
        st.subheader("Volume Trend")
        change_fig = px.area(
            df,
            x="date",
            y="volume",
            title=f"{selected_label} Trading Volume Trend",
        )
        change_fig.update_traces(line=dict(color=PRIMARY_COLOR, width=3), fillcolor="rgba(15, 139, 141, 0.24)")
        style_figure(change_fig, height=420, xaxis_title="Date", yaxis_title="Volume (USD)")
        change_fig.update_layout(showlegend=False)
        st.plotly_chart(change_fig, use_container_width=True)


def render_data_analysis_page() -> None:
    render_hero(
        "Data Analysis",
        "Explore live historical price behavior, moving averages, volume movement, and cross-asset correlation with interactive controls.",
    )

    coin_ids = list(SUPPORTED_COINS.keys())

    filter_col1, filter_col2 = st.columns([1, 1.2])
    with filter_col1:
        selected_coin_id = st.selectbox(
            "Select Cryptocurrency",
            coin_ids,
            index=0,
            format_func=lambda coin_id: SUPPORTED_COINS[coin_id]["label"],
        )
    with filter_col2:
        selected_days = st.slider("Select Historical Days", min_value=30, max_value=365, value=180, step=30)

    try:
        with st.spinner("Fetching live analysis data..."):
            crypto_df = fetch_live_data_with_fallback(selected_coin_id, selected_days)
    except Exception:
        live_data_error_message()
        return

    if crypto_df.empty:
        st.error("No records are available for the selected filters.")
        return

    selected_crypto = SUPPORTED_COINS[selected_coin_id]["label"]
    crypto_df["moving_average_30"] = crypto_df["price"].rolling(window=30, min_periods=1).mean()

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Filtered Records", f"{len(crypto_df):,}")
    stats_col2.metric("Average Price", format_currency(crypto_df["price"].mean()))
    stats_col3.metric("Average Volume", format_currency(crypto_df["volume"].mean()))

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Price Trend")
        price_fig = px.line(
            crypto_df,
            x="date",
            y="price",
            title=f"{selected_crypto} Historical Price Trend",
        )
        price_fig.update_traces(line=dict(color=ACCENT_COLOR, width=3))
        style_figure(price_fig, height=420, xaxis_title="Date", yaxis_title="Price (USD)")
        price_fig.update_layout(showlegend=False)
        st.plotly_chart(price_fig, use_container_width=True)

    with chart_col2:
        st.subheader("Moving Average")
        ma_fig = go.Figure()
        ma_fig.add_trace(
            go.Scatter(
                x=crypto_df["date"],
                y=crypto_df["price"],
                mode="lines",
                name="Actual Price",
                line=dict(color=ACCENT_COLOR, width=2),
            )
        )
        ma_fig.add_trace(
            go.Scatter(
                x=crypto_df["date"],
                y=crypto_df["moving_average_30"],
                mode="lines",
                name="30-Day Moving Average",
                line=dict(color=SECONDARY_COLOR, width=3),
            )
        )
        ma_fig.update_layout(
            title=f"{selected_crypto} Price vs 30-Day Moving Average",
        )
        style_figure(ma_fig, height=420, xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(ma_fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    feature_corr_df = crypto_df[["price", "market_cap", "volume", "percent_change_24h"]].corr().fillna(0)
    heatmap_fig = px.imshow(
        feature_corr_df,
        text_auto=".2f",
        color_continuous_scale=[
            [0.0, "#132347"],
            [0.25, "#2563eb"],
            [0.5, "#f8fbff"],
            [0.75, "#0f8b8d"],
            [1.0, "#ff7a59"],
        ],
        aspect="auto",
        title=f"{selected_crypto} Feature Correlation",
    )
    style_figure(heatmap_fig, height=500, xaxis_title="", yaxis_title="")
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.download_button(
        label="Download Filtered Dataset",
        data=crypto_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_crypto_analysis.csv",
        mime="text/csv",
    )


def render_prediction_page() -> None:
    render_hero(
        "Prediction Dashboard",
        "Review historical fit, retrain an LSTM on live cryptocurrency data, evaluate model performance, and generate future price projections.",
    )

    control_col1, control_col2 = st.columns([1, 1.2])
    with control_col1:
        selected_coin_id = st.selectbox(
            "Select Cryptocurrency for Prediction",
            list(SUPPORTED_COINS.keys()),
            index=0,
            format_func=lambda coin_id: SUPPORTED_COINS[coin_id]["label"],
        )
    with control_col2:
        history_days = st.slider(
            "Select Historical Window for Prediction", min_value=30, max_value=365, value=180, step=30
        )

    selected_coin_label = SUPPORTED_COINS[selected_coin_id]["label"]

    try:
        with st.spinner(f"Fetching live {selected_coin_label} data for prediction..."):
            prediction_df = fetch_live_data_with_fallback(selected_coin_id, history_days)
    except Exception:
        live_data_error_message()
        return

    prediction_df = prediction_df.sort_values("date").reset_index(drop=True)
    seq_length = 60
    if len(prediction_df) <= seq_length:
        st.warning("More live history is needed before the LSTM model can train. Increase the historical window.")
        return

    control_col1, control_col2 = st.columns([1, 1.2])
    with control_col1:
        forecast_days = st.slider("Select Number of Days to Predict", min_value=1, max_value=90, value=30)
    with control_col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-title">Prediction Module</div>
                <p class="small-note">
                    This module trains an LSTM model directly on the latest live selected-coin price series using
                    MinMax scaling, 60-timestep sequences, inverse scaling, and recursive forecasting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        with st.spinner("Training LSTM model and generating predictions..."):
            training_result = train_lstm_model(
                tuple(prediction_df["price"].astype(float)), seq_length=seq_length, epochs=4
            )
            predicted_prices = training_result["predicted_prices"]
            actual_prices = training_result["actual_prices"]
            predicted_prices = np.maximum(predicted_prices, 0)
            actual_prices = np.maximum(actual_prices, 0)

            prediction_dates = prediction_df["date"].iloc[seq_length:].reset_index(drop=True)
            prediction_df = prediction_df.iloc[seq_length:].copy().reset_index(drop=True)
            prediction_df["predicted_price"] = predicted_prices
            prediction_df["date"] = prediction_dates

            future_predictions = forecast_future_prices(
                training_result["model"],
                training_result["scaler"],
                training_result["scaled_data"],
                seq_length=seq_length,
                forecast_days=forecast_days,
            )
            last_date = prediction_df["date"].max()
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
            future_df = pd.DataFrame({"date": future_dates, "predicted_price": future_predictions})
    except ModuleNotFoundError as exc:
        st.error(str(exc))
        st.info("Install `tensorflow` and `keras`, then restart the Streamlit app to enable live LSTM forecasting.")
        return
    except Exception as exc:
        st.error(f"LSTM training failed: {exc}")
        return

    rmse = float(np.sqrt(mean_squared_error(actual_prices, predicted_prices)))
    safe_actual_prices = np.where(actual_prices == 0, np.nan, actual_prices)
    mape = float(np.nanmean(np.abs((actual_prices - predicted_prices) / safe_actual_prices)) * 100)

    st.success("Prediction outputs updated successfully.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Latest Actual Price", format_currency(actual_prices[-1]))
    metric_cols[1].metric("Latest Predicted Price", format_currency(prediction_df["predicted_price"].iloc[-1]))
    metric_cols[2].metric("RMSE", f"{rmse:,.2f}")
    metric_cols[3].metric("MAPE", f"{mape:,.2f}%")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Actual vs Predicted")
        fit_fig = go.Figure()
        fit_fig.add_trace(
            go.Scatter(
                x=prediction_df["date"],
                y=actual_prices,
                mode="lines",
                name="Actual Price",
                line=dict(color=ACCENT_COLOR, width=2.5),
            )
        )
        fit_fig.add_trace(
            go.Scatter(
                x=prediction_df["date"],
                y=prediction_df["predicted_price"],
                mode="lines",
                name="Predicted Price",
                line=dict(color=SECONDARY_COLOR, width=2.5, dash="dash"),
            )
        )
        fit_fig.update_layout(
            title=f"{selected_coin_label} Historical Actual vs Predicted Prices",
        )
        style_figure(fit_fig, height=430, xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fit_fig, use_container_width=True)

    with chart_col2:
        st.subheader("Future Prediction Output")
        forecast_fig = go.Figure()
        forecast_fig.add_trace(
            go.Scatter(
                x=prediction_df["date"].tail(120),
                y=actual_prices[-120:],
                mode="lines",
                name="Recent Actual Price",
                line=dict(color=PRIMARY_COLOR, width=2.5),
            )
        )
        forecast_fig.add_trace(
            go.Scatter(
                x=future_df["date"],
                y=future_df["predicted_price"],
                mode="lines+markers",
                name="Future Forecast",
                line=dict(color=SECONDARY_COLOR, width=3),
                marker=dict(size=7, color=SECONDARY_COLOR, line=dict(color="#ffffff", width=1)),
            )
        )
        forecast_fig.update_layout(
            title=f"Next {forecast_days} Days {selected_coin_label} Forecast",
        )
        style_figure(forecast_fig, height=430, xaxis_title="Date", yaxis_title="Predicted Price (USD)")
        st.plotly_chart(forecast_fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(
        future_df.assign(predicted_price=future_df["predicted_price"].round(2)),
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DatetimeColumn("Forecast Date", format="DD MMM YYYY"),
            "predicted_price": st.column_config.NumberColumn("Predicted Price", format="$%.2f"),
        },
    )


def render_live_data_page() -> None:
    render_hero(
        "Live Data",
        "Fetch the latest cryptocurrency pricing data from CoinGecko on demand without affecting the existing analysis and prediction modules.",
    )

    if "live_data_refresh_token" not in st.session_state:
        st.session_state["live_data_refresh_token"] = 0

    control_col1, control_col2, control_col3 = st.columns([1.2, 1, 0.8])
    with control_col1:
        selected_coin = st.selectbox(
            "Select Cryptocurrency",
            list(SUPPORTED_COINS.keys()),
            index=0,
            format_func=lambda coin_id: SUPPORTED_COINS[coin_id]["label"],
        )
    with control_col2:
        selected_days = st.slider("Select Historical Days", min_value=30, max_value=365, value=DEFAULT_DAYS, step=30)
    with control_col3:
        st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
        if st.button("Refresh Live Data", use_container_width=True):
            st.session_state["live_data_refresh_token"] += 1
            st.success("Fresh live data will be fetched for this page selection.")

    try:
        with st.spinner("Fetching live data from CoinGecko..."):
            live_df = fetch_live_data_with_fallback(
                selected_coin,
                selected_days,
                refresh_token=st.session_state["live_data_refresh_token"],
            )
    except Exception:
        live_data_error_message()
        return

    selected_coin_label = SUPPORTED_COINS[selected_coin]["label"]

    latest_price = live_df["price"].iloc[-1]
    previous_price = live_df["price"].iloc[-2] if len(live_df) > 1 else latest_price
    latest_market_cap = live_df["market_cap"].iloc[-1]
    latest_volume = live_df["volume"].iloc[-1]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Latest Price", format_currency(latest_price), f"{latest_price - previous_price:,.2f}")
    metric_cols[1].metric("Market Cap", format_currency(latest_market_cap))
    metric_cols[2].metric("Trading Volume", format_currency(latest_volume))

    st.divider()
    st.subheader(f"{selected_coin_label} Live Price Trend")
    live_fig = px.line(
        live_df,
        x="date",
        y="price",
        title=f"{selected_coin_label} Price Trend for the Last {selected_days} Days",
    )
    live_fig.update_traces(line=dict(color=ACCENT_COLOR, width=3))
    style_figure(live_fig, height=460, xaxis_title="Date", yaxis_title="Price (USD)")
    live_fig.update_layout(showlegend=False)
    st.plotly_chart(live_fig, use_container_width=True)

    st.subheader("Live Dataset Preview")
    st.dataframe(
        live_df[["date", "price", "market_cap", "volume", "percent_change_24h"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY"),
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_cap": st.column_config.NumberColumn("Market Cap", format="$%.0f"),
            "volume": st.column_config.NumberColumn("Volume", format="$%.0f"),
            "percent_change_24h": st.column_config.NumberColumn("24h Change (%)", format="%.2f"),
        },
    )

    st.download_button(
        label="Download Live Data",
        data=live_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_coin}_live_data.csv",
        mime="text/csv",
    )


def render_about_page() -> None:
    render_hero(
        "About Project",
        "This project demonstrates how data analytics and machine learning can be combined in Streamlit to create an academic-grade financial dashboard for cryptocurrency trend analysis using live CoinGecko data.",
    )

    tech_col, scope_col = st.columns(2)

    with tech_col:
        st.subheader("Project Description")
        st.markdown(
            """
            The dashboard studies historical cryptocurrency price movement, market capitalization, trading volume,
            and percentage change patterns across multiple digital assets. It is designed for final-year project
            demonstration, viva presentation, and applied analytics storytelling.
            """
        )
        st.subheader("Technologies Used")
        st.markdown(
            """
            - Python
            - Streamlit
            - Pandas and NumPy
            - Plotly for interactive visualization
            - Requests for live API integration
            - CoinGecko Market Chart API
            - Scikit-learn based prediction model
            - Time-series inspired market trend evaluation
            """
        )

    with scope_col:
        st.subheader("Project Scope")
        st.markdown(
            f"""
            - Data source: **CoinGecko live market API**
            - Assets supported: **{len(SUPPORTED_COINS)}**
            - Forecasting model: **TensorFlow LSTM**
            - Dashboard mode: **Real-time analytics**
            """
        )
        st.subheader("Author Details")
        st.markdown(
            """
            Developed as a final-year project dashboard for cryptocurrency market trend analysis,
            with emphasis on interactive analytics, predictive insights, and professional presentation quality.
            """
        )

    st.divider()
    st.info(
        "This application is structured for direct execution with `streamlit run app.py` and keeps the original dataset and model workflow intact."
    )


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-eyebrow">Digital Asset Desk</div>
            <div class="sidebar-title">Crypto Market Trend Analysis</div>
            <p class="sidebar-copy">
                A refined analytics workspace for tracking live prices, historical movement, and predictive signals.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Live Data", "Data Analysis", "Prediction", "About"],
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <div class="dashboard-note">
            <div class="small-note">
                <strong>Data Source:</strong> CoinGecko API<br>
                <strong>Cache Window:</strong> 300 seconds<br>
                <strong>Mode:</strong> Live analytics only
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def main() -> None:
    inject_styles()

    page = render_sidebar()

    try:
        if page == "Overview":
            render_overview_page()
        elif page == "Live Data":
            render_live_data_page()
        elif page == "Data Analysis":
            render_data_analysis_page()
        elif page == "Prediction":
            render_prediction_page()
        elif page == "About":
            render_about_page()
    except Exception as exc:
        st.error(f"An unexpected error occurred while rendering the page: {exc}")


if __name__ == "__main__":
    main()
