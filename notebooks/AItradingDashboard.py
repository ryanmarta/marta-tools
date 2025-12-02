import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
from scipy.stats import norm
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. CONFIGURATION
warnings.filterwarnings("ignore")
# Changed page_icon from DNA to Brain
st.set_page_config(page_title="AI Project Nexus", layout="wide", page_icon="🧠")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC; color: #334155; }
        section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
        header[data-testid="stHeader"] { background-color: #F8FAFC; }
        h1, h2, h3 { color: #0F172A !important; font-family: 'Inter', sans-serif; font-weight: 700; }
        
        /* Metric Styling */
        div.stMetric {
            background-color: #FFFFFF;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.9rem; font-weight: 600; }
        div[data-testid="stMetricValue"] { font-family: 'Courier New', monospace; color: #0F172A !important; font-weight: 700; font-size: 1.6rem; }
        
        /* Inputs */
        div.stButton > button { background-color: #2563EB; color: white; border-radius: 6px; border: none; }
        div.stButton > button:hover { background-color: #1D4ED8; }
        
        /* Educational Footer */
        .edu-footer {
            background-color: #F1F5F9;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2563EB;
            margin-top: 30px;
            color: #334155;
        }
        .highlight-box {
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'ticker' not in st.session_state: st.session_state['ticker'] = "TSLA"
if 'window' not in st.session_state: st.session_state['window'] = 20
if 'std_dev' not in st.session_state: st.session_state['std_dev'] = 2.0
if 'thresh' not in st.session_state: st.session_state['thresh'] = 0.25

def reset_defaults():
    st.session_state['ticker'] = "TSLA"
    st.session_state['window'] = 20
    st.session_state['std_dev'] = 2.0
    st.session_state['thresh'] = 0.25

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Project Nexus")
    # Updated Radio Options
    project_selection = st.radio("Available Projects", ["📈 AI Quant Trading", "🧠 ML Volatility (Regime Detection)", "🤖 Future Module"])
    st.markdown("---")
    if project_selection == "📈 AI Quant Trading":
        st.button("🔄 Reset Defaults", on_click=reset_defaults)
    st.caption("v9.1 | Volatility Regime Intelligence")

# ==========================================
# PROJECT 1: AI QUANT TRADING
# ==========================================
if project_selection == "📈 AI Quant Trading":
    
    st.title("📈 AI Quant Dashboard")
    st.caption("Multi-Factor Quant Model: Volatility Squeeze + Volume Velocity + Relative Strength")
    
    # --- INPUTS ---
    with st.expander("⚙️ Strategy Configuration", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: 
            ticker = st.text_input("Ticker", value=st.session_state['ticker'], key="_t").upper()
            st.session_state['ticker'] = ticker
        with c2: 
            window = st.slider("Lookback Window", 10, 60, value=st.session_state['window'], key="_w")
            st.session_state['window'] = window
        with c3: 
            std_dev = st.slider("Volatility (Sigma)", 1.5, 3.0, value=st.session_state['std_dev'], key="_s")
            st.session_state['std_dev'] = std_dev
        with c4: 
            squeeze_thresh = st.slider("Squeeze Threshold", 0.10, 0.60, value=st.session_state['thresh'], step=0.01, key="_th")
            st.session_state['thresh'] = squeeze_thresh

    # --- DATA ENGINE ---
    @st.cache_data(ttl=300) 
    def get_quant_data(symbol):
        try:
            tickers = f"{symbol} SPY"
            df = yf.download(tickers, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                closes = df['Close']
                volumes = df['Volume']
            else: return None, None, None
            return closes[symbol].dropna(), volumes[symbol].dropna(), closes['SPY'].dropna()
        except: return None, None, None

    data, volume, spy_data = get_quant_data(ticker)
    if data is None or data.empty: st.error("Data Error. Check Ticker."); st.stop()

    # --- TECHNICAL CALCULATIONS ---
    # Bollinger Bands
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bandwidth = (upper - lower) / sma 
    
    current_price = data.iloc[-1]
    current_bw = bandwidth.iloc[-1]
    
    # Trend Logic
    trend_bias = "BULLISH" if current_price > sma.iloc[-1] else "BEARISH"
    ema_9 = data.ewm(span=9, adjust=False).mean()
    ema_21 = data.ewm(span=21, adjust=False).mean()
    ema_signal = "BULLISH" if ema_9.iloc[-1] > ema_21.iloc[-1] else "BEARISH"

    # --- QUANT FACTOR ENGINE ---
    
    # 1. Volatility Squeeze (Weighted Score)
    is_squeeze = current_bw < squeeze_thresh
    squeeze_depth = 0
    if is_squeeze:
        squeeze_depth = (squeeze_thresh - current_bw) / squeeze_thresh
        squeeze_depth = min(max(squeeze_depth, 0), 1) # Clamp
    
    # 2. Volume Velocity
    vol_3d_ma = volume.rolling(3).mean().iloc[-1]
    vol_30_ma = volume.rolling(30).mean().iloc[-1]
    vol_60_ma = volume.rolling(60).mean().iloc[-1]
    vol_90_ma = volume.rolling(90).mean().iloc[-1]
    
    velocity_score = 0
    if vol_3d_ma > vol_30_ma: velocity_score += 1
    if vol_3d_ma > vol_60_ma: velocity_score += 1
    if vol_3d_ma > vol_90_ma: velocity_score += 1
    
    if velocity_score >= 2:
        vol_status = "Speeding Up"
        vol_color = "normal" 
    elif velocity_score <= 0:
        vol_status = "Slowing Down"
        vol_color = "inverse"
    else:
        vol_status = "Neutral"
        vol_color = "off"

    # 3. Relative Strength
    df_rs = pd.DataFrame({'Stock': data, 'SPY': spy_data}).dropna()
    df_rs['RS_Ratio'] = df_rs['Stock'].pct_change(60) - df_rs['SPY'].pct_change(60)
    current_rs = df_rs['RS_Ratio'].iloc[-1]
    
    # 4. Risk (Beta)
    returns = df_rs.pct_change().dropna()
    cov = returns['Stock'].cov(returns['SPY'])
    var = returns['SPY'].var()
    beta = cov / var

    # --- CONFIDENCE ALGORITHM ---
    confidence = 0
    
    # A. Trend Alignment (35%)
    if trend_bias == "BULLISH" and ema_signal == "BULLISH": confidence += 35
    elif trend_bias == "BEARISH" and ema_signal == "BEARISH": confidence += 35
    else: confidence += 15 
    
    # B. Squeeze Weight (25%)
    if is_squeeze:
        confidence += 15 + (squeeze_depth * 10) 
    
    # C. Volume Confirmation (20%)
    if vol_status == "Speeding Up": confidence += 20
    elif vol_status == "Neutral": confidence += 10
    
    # D. Relative Strength (20%)
    if trend_bias == "BULLISH" and current_rs > 0: confidence += 20
    elif trend_bias == "BEARISH" and current_rs < 0: confidence += 20
    
    confidence = min(int(confidence), 99)

    # --- DISPLAY METRICS ---
    st.subheader("🤖 Quant Model Output")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${current_price:.2f}")
    m2.metric("Trend Bias (SMA)", trend_bias, delta="Long" if trend_bias=="BULLISH" else "Short", delta_color="normal" if trend_bias=="BULLISH" else "inverse")
    m3.metric("Momentum (EMA)", ema_signal, delta="Strong" if ema_signal==trend_bias else "Weak", help="9 EMA vs 21 EMA crossover.")
    m4.metric("Squeeze Status", "COILED" if is_squeeze else "LOOSE", delta=f"BW: {current_bw:.3f}", delta_color="inverse")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Confidence Factor", f"{confidence}%", help="Weighted score of Trend, Squeeze Depth, Volume Velocity, and RS.")
        st.progress(confidence/100)
    with q2:
        st.metric("Volume Velocity", vol_status, delta="vs 30/60/90d Avg", delta_color=vol_color, help="Compares current volume flow against 30, 60, and 90-day baselines.")
    with q3:
        st.metric("Relative Strength", f"{current_rs:.1%}", delta="vs SPY (60d)", help="Performance differential vs SPY over last 60 days.")
    with q4:
        st.metric("Risk (Beta)", f"{beta:.2f}", help="Volatility relative to S&P 500. >1.0 is aggressive, <1.0 is defensive.")

    # --- CHARTS ---
    st.write("") 
    tab1, tab2 = st.tabs(["💰 Price Action", "📉 Bandwidth Analyzer"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Price', line=dict(color='#0F172A', width=1.5)))
        fig.add_trace(go.Scatter(x=upper.index, y=upper, mode='lines', name='Upper', line=dict(color='#10B981', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=lower.index, y=lower, mode='lines', name='Lower', line=dict(color='#10B981', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(16, 185, 129, 0.05)'))
        fig.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name='SMA', line=dict(color='#F59E0B', width=1.5)))
        fig.update_layout(height=450, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        bw_fig = go.Figure()
        bw_fig.add_trace(go.Scatter(x=bandwidth.index[-180:], y=bandwidth.tail(180), mode='lines', name='Bandwidth', line=dict(color='#3B82F6', width=2)))
        bw_fig.add_hline(y=squeeze_thresh, line_dash="dash", line_color="#EF4444", annotation_text=f"Threshold ({squeeze_thresh})")
        bw_fig.update_layout(height=400, template="plotly_white", title="Historical Bandwidth vs Threshold", hovermode="x unified")
        st.plotly_chart(bw_fig, use_container_width=True)

    # --- MONTE CARLO ---
    st.markdown("---")
    st.subheader("🎲 Monte Carlo Scenario (30 Days)")
    st.caption("Simulating **1,000 future paths** based on historical volatility.")
    
    daily_vol = np.log(data / data.shift(1)).std()
    annual_vol = daily_vol * np.sqrt(252)
    
    SIMULATIONS = 1000
    DAYS = 30
    random_shocks = np.random.normal(0, daily_vol, (DAYS, SIMULATIONS))
    price_paths = current_price * (1 + random_shocks).cumprod(axis=0)
    
    final_prices = price_paths[-1]
    profitability = (final_prices > current_price).mean()
    
    mc_fig = go.Figure()
    mc_fig.add_trace(go.Scatter(x=np.tile(np.arange(DAYS), SIMULATIONS), y=price_paths.flatten(order='F'), 
                                mode='lines', line=dict(color='#10B981', width=0.5), opacity=0.15, showlegend=False, hoverinfo='skip'))
    mc_fig.add_trace(go.Scatter(x=np.arange(DAYS), y=price_paths.mean(axis=1), mode='lines', name='Mean Path', line=dict(color='black', width=2)))
    mc_fig.add_hline(y=current_price, line_dash="dash", line_color="black")
    mc_fig.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0))
    
    c_mc1, c_mc2 = st.columns([3, 1])
    with c_mc1: st.plotly_chart(mc_fig, use_container_width=True)
    with c_mc2:
        st.metric("Probability of Profit", f"{profitability:.1%}", help="Percentage of 1000 scenarios that end positive.")
        st.metric("Projected Volatility", f"{annual_vol:.1%}")

    # --- EDUCATIONAL FOOTER ---
    st.markdown("""
    <div class="edu-footer">
        <h3>🎓 Model Architecture & Logic</h3>
        <p><strong>1. Volatility Squeeze:</strong> Identifies potential energy. BW < Threshold ({squeeze_thresh}) = Squeeze.</p>
        <p><strong>2. Volume Velocity:</strong> Analyzes speed of participation vs 30/60/90d baselines.</p>
        <p><strong>3. Relative Strength:</strong> Performance differential vs SPY.</p>
        <p><strong>4. Risk Parity:</strong> Beta {beta:.2f} indicates volatility relative to market.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PROJECT 2: ML VOLATILITY (Regime Detection)
# ==========================================
elif project_selection == "🧠 ML Volatility (Regime Detection)":
    st.title("🧠 Regime Intelligence (ML)")
    st.caption("Predicting transitions in uncertainty regimes, not price direction.")

    # --- CONCEPTUAL INTRO ---
    st.markdown("""
    <div class="highlight-box">
        <strong>The "CFO-Grade" Premise:</strong> Decisions fail when uncertainty shifts, not when price forecasts are slightly wrong.
        This model answers: <em>"Are we entering a fragile period?"</em>
        <br><br>
        We use Machine Learning to predict the <strong>probability of a high-volatility regime expansion</strong> in the next 5 days.
    </div>
    """, unsafe_allow_html=True)

    # --- CONFIG ---
    col1, col2 = st.columns([1, 3])
    with col1:
        ml_ticker = st.text_input("Asset Ticker", value="TSLA", key="ml_t").upper()
    
    # --- ML DATA ENGINE ---
    @st.cache_data(ttl=600)
    def prepare_ml_data(symbol):
        try:
            # 1. Fetch deep history for training
            df = yf.download(symbol, period="3y", progress=False)
            if df.empty: return None
            
            # 2. Feature Engineering (Structural Signals)
            data = pd.DataFrame()
            data['Close'] = df['Close']
            data['Returns'] = data['Close'].pct_change()
            
            # Realized Volatility (20-day rolling std dev of returns)
            data['Realized_Vol'] = data['Returns'].rolling(20).std() * np.sqrt(252)
            
            # Bandwidth (Squeeze metric)
            sma = data['Close'].rolling(20).mean()
            std = data['Close'].rolling(20).std()
            data['Bandwidth'] = ((sma + 2*std) - (sma - 2*std)) / sma
            
            # Compression Rate (Slope of bandwidth)
            data['BW_Slope'] = data['Bandwidth'].diff(5)
            
            # Volume Z-Score (Participation context)
            vol_mean = df['Volume'].rolling(20).mean()
            vol_std = df['Volume'].rolling(20).std()
            data['Vol_Z'] = (df['Volume'] - vol_mean) / vol_std
            
            # Distance from Mean (Stress proxy)
            data['Dist_Mean'] = (data['Close'] - sma) / sma
            
            # 3. Label Generation (The Target)
            # Y = 1 if Future Volatility (t+5) > 75th Percentile of Historical Volatility
            future_window = 5
            vol_threshold = data['Realized_Vol'].quantile(0.75)
            data['Future_Vol'] = data['Realized_Vol'].shift(-future_window)
            data['Target'] = (data['Future_Vol'] > vol_threshold).astype(int)
            
            data = data.dropna()
            return data, vol_threshold
        except Exception as e:
            return None, None

    df_ml, threshold_val = prepare_ml_data(ml_ticker)

    if df_ml is None:
        st.error("Error fetching data. Try a valid US ticker.")
        st.stop()

    # --- TRAINING THE MODEL (ON THE FLY) ---
    features = ['Realized_Vol', 'Bandwidth', 'BW_Slope', 'Vol_Z', 'Dist_Mean']
    X = df_ml[features]
    y = df_ml['Target']

    # Train/Test Split (Time Series sensitive - actually we just train on past to predict current state)
    # For this demo, we train on all historical data available up to today to predict "Now"
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X.iloc[:-1], y.iloc[:-1]) # Train on everything except the very last row (today)

    # Predict for Today (The most recent data point)
    current_features = X.iloc[[-1]] 
    probability_expansion = model.predict_proba(current_features)[0][1] # Probability of Class 1
    
    # Current State Analysis
    current_vol = df_ml['Realized_Vol'].iloc[-1]
    
    # Regime Definition for Display (Simplified for better UI)
    if current_vol < df_ml['Realized_Vol'].quantile(0.25):
        regime = "COMPLACENCY"
        regime_color = "#10B981" # Green
    elif current_vol < df_ml['Realized_Vol'].quantile(0.75):
        regime = "TRANSITION"
        regime_color = "#F59E0B" # Orange
    else:
        regime = "HIGH STRESS"
        regime_color = "#EF4444" # Red

    # --- DISPLAY DASHBOARD ---
    
    # Row 1: The Prediction
    st.subheader("🔮 Uncertainty Forecast")
    
    k1, k2, k3 = st.columns(3)
    
    with k1:
        st.metric("Current Regime State", regime, delta="Realized Vol", delta_color="off")
    
    with k2:
        prob_color = "inverse" if probability_expansion > 0.5 else "normal"
        st.metric("P(Regime Expansion)", f"{probability_expansion:.1%}", 
                 delta="Probability of Shift", delta_color=prob_color,
                 help="Probability that volatility will spike into the top 25th percentile within 5 days.")
    
    with k3:
        st.metric("Critical Threshold", f"{threshold_val:.2%}", help="The level of volatility defined as 'High Stress' for this specific asset.")

    # Row 2: Visualizing the Regimes
    st.write("---")
    # Removed DNA icon here
    st.subheader("📊 Regime Map & Feature Importance")
    
    t1, t2 = st.tabs(["Regime History", "Model Logic (Features)"])
    
    with t1:
        # Create a colored scatter plot for regimes
        fig_reg = go.Figure()
        
        # Plot Price
        fig_reg.add_trace(go.Scatter(x=df_ml.index, y=df_ml['Close'], mode='lines', 
                                   name='Price', line=dict(color='#334155', width=1)))
        
        # Color background based on Volatility Regime (Simplified for visual)
        # We overlay a heatmap-like bar at the bottom
        # Switched to RdYlGn_r (Red-Yellow-Green Reversed) so Green = Low Vol, Red = High Vol
        fig_reg.add_trace(go.Bar(x=df_ml.index, y=df_ml['Realized_Vol']*100, 
                               name='Realized Vol', yaxis='y2', marker=dict(color=df_ml['Realized_Vol'], colorscale='RdYlGn_r')))
        
        fig_reg.update_layout(
            height=400, 
            template="plotly_white", 
            title=f"Price vs Volatility Intensity ({ml_ticker})",
            yaxis2=dict(title="Volatility", overlaying='y', side='right', showgrid=False)
        )
        st.plotly_chart(fig_reg, use_container_width=True)
        
    with t2:
        # Feature Importance
        importances = model.feature_importances_
        feature_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(feature_df, x='Importance', y='Feature', orientation='h', title="What drives the prediction?")
        fig_imp.update_traces(marker_color='#2563EB')
        fig_imp.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Feature importance derived from the Random Forest classifier trained on historical data.")

    # --- EDUCATIONAL FOOTER ---
    st.markdown("""
    <div class="edu-footer">
        <h3>🎓 Decision Intelligence Context</h3>
        <p><strong>1. The Target ($y$):</strong> We are not predicting price up/down. We are predicting if Realized Volatility at $t+5$ will exceed the historical 75th percentile.</p>
        <p><strong>2. The Probability:</strong> A score of >50% suggests the system is "fragile" and prone to a shock.</p>
        <p><strong>3. Use Case:</strong> 
        <ul>
            <li><strong>CFO:</strong> Delay M&A or financing if P(Expansion) is high.</li>
            <li><strong>Trader:</strong> Switch from directional bets to long-volatility (straddles/vix).</li>
            <li><strong>Supply Chain:</strong> Increase inventory buffers if currency volatility is predicted to expand.</li>
        </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PROJECT 3: FUTURE MODULE
# ==========================================
# ==========================================
# PROJECT 3: FUTURE MODULE (Market Scanner)
# ==========================================
elif project_selection == "🤖 Future Module":
    st.title("🤖 Nexus Scanner: S&P 100 Elite")
    st.caption("Automated Opportunity Hunter | Criteria: The Ryan Model (Strict 80%+ Confidence)")

    # --- 1. SCANNER CONFIGURATION ---
    with st.expander("🛠️ Scanner Settings", expanded=False):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            scan_window = st.slider("Lookback Window", 10, 50, 20)
        with sc2:
            scan_sqz_thresh = st.slider("Squeeze Threshold", 0.10, 0.40, 0.25)
        with sc3:
            # Default set to 80 per "Ryan Model" specs
            min_confidence = st.slider("Min Confidence %", 50, 95, 80)
    
    # --- 2. THE WATCHLIST (Top 100 + High Beta) ---
    TICKER_LIST = [
        # Mag 7 & Tech
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "QCOM", "TXN", "INTC", "IBM", "MU", "NOW", "UBER", "PANW",
        # Financials
        "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "AXP", "BLK", "C", "PYPL", "HOOD", "COIN", "SOFI",
        # Consumer
        "WMT", "COST", "PG", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX", "TGT", "LOW", "TJX", 
        # Healthcare
        "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "PFE", "AMGN", "ISRG", "BMY", "GILD", "CVS",
        # Industrial & Energy
        "CAT", "DE", "HON", "GE", "UNP", "UPS", "BA", "LMT", "RTX", "XOM", "CVX", "COP", "SLB", "EOG",
        # High Beta / Day Trader Pack
        "MSTR", "MARA", "PLTR", "DKNG", "ROKU", "SQ", "AFRM", "RIOT", "CLSK", "CVNA", "UPST", "AI", "GME", "AMC", 
        # ETFs
        "SPY", "QQQ", "IWM", "DIA", "TLT"
    ]

    # --- 3. ANALYSIS ENGINE ---
    @st.cache_data(ttl=600)
    def batch_process_tickers(tickers):
        results = []
        
        # Download batch data for speed
        # We download slightly more history to ensure rolling windows are full
        data_batch = yf.download(tickers + ["SPY"], period="6mo", progress=False)
        
        # Helper to get series safely
        def get_series(df, symbol, col):
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    return df[col][symbol]
                else:
                    return df[col]
            except: return None

        # Iterate and Calculate
        for symbol in tickers:
            try:
                # Extract Data
                closes = get_series(data_batch, symbol, 'Close').dropna()
                volumes = get_series(data_batch, symbol, 'Volume').dropna()
                spy_closes = get_series(data_batch, "SPY", 'Close').dropna()
                
                if len(closes) < 60: continue

                # --- RYAN MODEL METRICS ---
                
                # 1. BB & Squeeze
                sma = closes.rolling(window=scan_window).mean()
                std = closes.rolling(window=scan_window).std()
                upper = sma + (std * 2.0)
                lower = sma - (std * 2.0)
                bandwidth = (upper - lower) / sma
                
                current_price = closes.iloc[-1]
                current_bw = bandwidth.iloc[-1]
                is_squeeze = current_bw < scan_sqz_thresh
                
                squeeze_depth = 0
                if is_squeeze:
                    squeeze_depth = (scan_sqz_thresh - current_bw) / scan_sqz_thresh
                    squeeze_depth = min(max(squeeze_depth, 0), 1)

                # 2. Trend & Momentum
                trend_bias = "BULLISH" if current_price > sma.iloc[-1] else "BEARISH"
                ema_9 = closes.ewm(span=9, adjust=False).mean()
                ema_21 = closes.ewm(span=21, adjust=False).mean()
                ema_signal = "BULLISH" if ema_9.iloc[-1] > ema_21.iloc[-1] else "BEARISH"
                
                # 3. Volume Velocity
                vol_3d = volumes.rolling(3).mean().iloc[-1]
                vol_30 = volumes.rolling(30).mean().iloc[-1]
                vol_60 = volumes.rolling(60).mean().iloc[-1]
                vol_90 = volumes.rolling(90).mean().iloc[-1]
                
                velocity_score = 0
                if vol_3d > vol_30: velocity_score += 1
                if vol_3d > vol_60: velocity_score += 1
                if vol_3d > vol_90: velocity_score += 1
                
                vol_status = "Speeding Up" if velocity_score >= 2 else ("Slowing Down" if velocity_score <= 0 else "Neutral")

                # 4. Relative Strength
                stock_pct = closes.pct_change(60).iloc[-1]
                spy_pct = spy_closes.pct_change(60).iloc[-1]
                rs_ratio = stock_pct - spy_pct

                # --- CONFIDENCE MATH ---
                confidence = 0
                
                # A. Trend Alignment (35%)
                if trend_bias == "BULLISH" and ema_signal == "BULLISH": confidence += 35
                elif trend_bias == "BEARISH" and ema_signal == "BEARISH": confidence += 35
                else: confidence += 15 
                
                # B. Squeeze Weight (25%)
                if is_squeeze: confidence += 15 + (squeeze_depth * 10) 
                
                # C. Volume (20%)
                if vol_status == "Speeding Up": confidence += 20
                elif vol_status == "Neutral": confidence += 10
                
                # D. RS (20%)
                if trend_bias == "BULLISH" and rs_ratio > 0: confidence += 20
                elif trend_bias == "BEARISH" and rs_ratio < 0: confidence += 20
                
                confidence = min(int(confidence), 99)

                results.append({
                    "Ticker": symbol,
                    "Price": current_price,
                    "Trend": trend_bias,
                    "Momentum": ema_signal,
                    "Squeeze": "COILED" if is_squeeze else "LOOSE",
                    "Bandwidth": current_bw,
                    "Confidence": confidence,
                    "RS_vs_SPY": rs_ratio,
                    "Vol_State": vol_status
                })

            except Exception as e:
                continue
                
        return pd.DataFrame(results)

    # --- 4. EXECUTION ---
    if st.button("🚀 Initialize Scan Sequence"):
        # Progress Bar Logic
        progress_text = "Connecting to Neural Lattice... Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        with st.spinner(f"Scanning {len(TICKER_LIST)} Assets..."):
            # Simulate progress for UX
            my_bar.progress(30, text="Downloading Batch Data...")
            df_results = batch_process_tickers(TICKER_LIST)
            my_bar.progress(90, text="Applying Ryan Model Logic...")
            
            if df_results is not None and not df_results.empty:
                my_bar.progress(100, text="Scan Complete.")
                my_bar.empty()

                # --- FILTERING LOGIC (THE RYAN MODEL) ---
                # Buy Long: Bullish Trend + Bullish Mom + Coiled + High Conf
                longs = df_results[
                    (df_results['Trend'] == 'BULLISH') & 
                    (df_results['Momentum'] == 'BULLISH') & 
                    (df_results['Squeeze'] == 'COILED') & 
                    (df_results['Confidence'] >= min_confidence)
                ]

                # Buy Short: Bearish Trend + Bearish Mom + Coiled + High Conf
                shorts = df_results[
                    (df_results['Trend'] == 'BEARISH') & 
                    (df_results['Momentum'] == 'BEARISH') & 
                    (df_results['Squeeze'] == 'COILED') & 
                    (df_results['Confidence'] >= min_confidence)
                ]

                # --- DISPLAY TABLES ---
                
                # 🟢 LONG CANDIDATES TABLE
                st.subheader(f"🟢 Long Setups ({len(longs)})")
                if not longs.empty:
                    st.dataframe(
                        longs.style.format({
                            "Price": "${:.2f}", 
                            "Bandwidth": "{:.4f}", 
                            "RS_vs_SPY": "{:.2%}",
                            "Confidence": "{:.0f}%"
                        }).background_gradient(subset=['Confidence'], cmap='Greens', vmin=50, vmax=100),
                        use_container_width=True,
                        column_order=("Ticker", "Confidence", "Price", "Trend", "Momentum", "Squeeze", "RS_vs_SPY"),
                        hide_index=True
                    )
                else:
                    st.info(f"No Long setups found matching strict Ryan Model criteria (Conf > {min_confidence}% + Coiled).")

                st.markdown("---")

                # 🔴 SHORT CANDIDATES TABLE
                st.subheader(f"🔴 Short Setups ({len(shorts)})")
                if not shorts.empty:
                    st.dataframe(
                        shorts.style.format({
                            "Price": "${:.2f}", 
                            "Bandwidth": "{:.4f}", 
                            "RS_vs_SPY": "{:.2%}",
                            "Confidence": "{:.0f}%"
                        }).background_gradient(subset=['Confidence'], cmap='Reds', vmin=50, vmax=100),
                        use_container_width=True,
                        column_order=("Ticker", "Confidence", "Price", "Trend", "Momentum", "Squeeze", "RS_vs_SPY"),
                        hide_index=True
                    )
                else:
                    st.info(f"No Short setups found matching strict Ryan Model criteria (Conf > {min_confidence}% + Coiled).")
                
                # 📋 FULL DATA EXPANDER
                with st.expander("📂 View Full Scan Results (All 100 Assets)"):
                    st.dataframe(
                        df_results.sort_values(by="Confidence", ascending=False)
                        .style.format({"Price": "${:.2f}", "Confidence": "{:.0f}%"}), 
                        use_container_width=True
                    )
            
            else:
                st.error("Scan returned no data. Please check API connection.")
    else:
        st.info("👋 Ready to Scan. This will analyze the Top 100 S&P stocks using the Ryan Model logic.")