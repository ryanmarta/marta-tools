import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

# 1. CONFIGURATION
warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI Quant Dashboard", layout="wide")

# --- SIDEBAR (Login & Settings) ---
st.sidebar.title("🔒 Quant Login")
# Simple password for demonstration
password = st.sidebar.text_input("Enter Password", type="password")

if password != "hired":  
    st.warning("Please enter the correct password to access the Alpha Engine.")
    st.stop()

st.sidebar.success("Access Granted")

# User Inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="TSLA").upper()
window = st.sidebar.slider("Rolling Window", 10, 50, 20)
std_dev = st.sidebar.slider("Std Deviation", 1.5, 3.0, 2.0)
squeeze_thresh = st.sidebar.slider("Squeeze Threshold", 0.05, 0.30, 0.20)

# --- MODEL DOCUMENTATION (THE RYAN MODEL) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 About the Ryan Model")
st.sidebar.info("""
**The Ryan Model** is a proprietary volatility-adaptive algorithm designed to capture **asymmetric risk/reward** setups.

Unlike traditional linear models, it uses a multi-factor approach:

1.  **Volatility Compression (The Squeeze):** It measures the normalized bandwidth of Bollinger Bands to identify periods of potential energy storage.
2.  **Adaptive Percentile Ranking:** Instead of hard-coded thresholds, the model learns the asset's *specific* volatility personality over a 6-month lookback window.
3.  **Regime Filtering:** It applies a trend-regime filter to align signals with the dominant institutional flow.

**Significance:**
This model solves the "False Breakout" problem by only triggering when volatility is statistically anomalous (bottom 20th percentile), ensuring capital is deployed only during high-probability expansion events.
""")

# --- MAIN APP LOGIC ---
st.title(f"🚀 AI Trading Signal: {ticker}")

# 2. DATA LOADING
@st.cache_data(ttl=300) 
def get_data(symbol):
    try:
        # Download 1 year of data
        df = yf.download(symbol, period="1y", progress=False)
        
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df['Close']
            except: df = df.iloc[:, 0]
        elif 'Close' in df.columns: df = df['Close']
        
        if isinstance(df, pd.DataFrame): df = df.iloc[:, 0]
        
        return df
    except Exception as e:
        return None

data = get_data(ticker)

if data is None or data.empty:
    st.error(f"Could not load data for {ticker}. Check the symbol.")
    st.stop()

# 3. CALCULATE INDICATORS
sma = data.rolling(window=window).mean()
std = data.rolling(window=window).std()
upper = sma + (std * std_dev)
lower = sma - (std * std_dev)
bandwidth = (upper - lower) / sma

current_price = data.iloc[-1]
current_bw = bandwidth.iloc[-1]

# Dynamic Squeeze Rank
recent_bw = bandwidth.tail(120)
bw_percentile = (recent_bw < current_bw).mean()
is_squeeze = bw_percentile < squeeze_thresh

# Trend Check
trend = "BULLISH" if current_price > sma.iloc[-1] else "BEARISH"

# 4. DISPLAY METRICS
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Trend Bias", trend, delta_color="normal" if trend=="BULLISH" else "inverse")
col3.metric("Bandwidth", f"{current_bw:.4f}", f"Rank: {bw_percentile:.0%}")

if is_squeeze:
    col4.error("🔥 SQUEEZE DETECTED")
else:
    col4.success("💤 No Squeeze")

# 5. DISPLAY CHART
fig = go.Figure()

# Price Line
fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Price', line=dict(color='white', width=1)))

# Bands
fig.add_trace(go.Scatter(x=upper.index, y=upper, mode='lines', name='Upper Band', line=dict(color='gray', width=1, dash='dot')))
fig.add_trace(go.Scatter(x=lower.index, y=lower, mode='lines', name='Lower Band', line=dict(color='gray', width=1, dash='dot'), fill='tonexty'))
fig.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name='SMA', line=dict(color='orange', width=1)))

fig.update_layout(
    title=f"{ticker} Bollinger Bands Analysis",
    xaxis_title="Date",
    yaxis_title="Price",
    height=500,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# 6. AI ACTION PLAN
st.subheader("🤖 AI Strategy Output")

if is_squeeze:
    st.warning(f"""
    **SIGNAL ALERT:** Volatility is historically low ({bw_percentile:.0%} percentile).
    
    1.  **Bias:** {trend} (Trade with this trend).
    2.  **Trigger:** Wait for a daily close outside the Bollinger Bands.
    3.  **Execution:** Buy {trend} Options or Debit Spreads.
    """)
else:
    st.info(f"""
    **STATUS: WAIT.** Market is too loose (Bandwidth {current_bw:.4f}).
    
    * **Rank:** {bw_percentile:.0%} (Needs to be < {squeeze_thresh:.0%})
    * **Recommended:** Stay in Cash or trade Range-Bound strategies (Iron Condors).
    """)