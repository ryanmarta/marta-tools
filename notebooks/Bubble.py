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
from sklearn.metrics import roc_auc_score, confusion_matrix

# 1. CONFIGURATION
warnings.filterwarnings("ignore")
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
            box-shadow: 0 1px 2px rgba(0,0,0,0.15);
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
    project_selection = st.radio(
        "Available Projects",
        ["📈 AI Quant Trading", "🧠 ML Volatility (Regime Detection)", "🤖 Future Module"]
    )
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
            else:
                return None, None, None
            return closes[symbol].dropna(), volumes[symbol].dropna(), closes['SPY'].dropna()
        except:
            return None, None, None

    data, volume, spy_data = get_quant_data(ticker)
    if data is None or data.empty:
        st.error("Data Error. Check Ticker.")
        st.stop()

    # --- TECHNICAL CALCULATIONS ---
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
        squeeze_depth = min(max(squeeze_depth, 0), 1)
    
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
    if trend_bias == "BULLISH" and ema_signal == "BULLISH":
        confidence += 35
    elif trend_bias == "BEARISH" and ema_signal == "BEARISH":
        confidence += 35
    else:
        confidence += 15 
    
    # B. Squeeze Weight (25%)
    if is_squeeze:
        confidence += 15 + (squeeze_depth * 10) 
    
    # C. Volume Confirmation (20%)
    if vol_status == "Speeding Up":
        confidence += 20
    elif vol_status == "Neutral":
        confidence += 10
    
    # D. Relative Strength (20%)
    if trend_bias == "BULLISH" and current_rs > 0:
        confidence += 20
    elif trend_bias == "BEARISH" and current_rs < 0:
        confidence += 20
    
    confidence = min(int(confidence), 99)

    # --- DISPLAY METRICS ---
    st.subheader("🤖 Quant Model Output")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${current_price:.2f}")
    m2.metric("Trend Bias (SMA)", trend_bias, delta="Long" if trend_bias=="BULLISH" else "Short",
              delta_color="normal" if trend_bias=="BULLISH" else "inverse")
    m3.metric("Momentum (EMA)", ema_signal,
              delta="Strong" if ema_signal==trend_bias else "Weak",
              help="9 EMA vs 21 EMA crossover.")
    m4.metric("Squeeze Status", "COILED" if is_squeeze else "LOOSE",
              delta=f"BW: {current_bw:.3f}", delta_color="inverse")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Confidence Factor", f"{confidence}%", help="Weighted score of Trend, Squeeze Depth, Volume Velocity, and RS.")
        st.progress(confidence/100)
    with q2:
        st.metric("Volume Velocity", vol_status, delta="vs 30/60/90d Avg",
                  delta_color=vol_color,
                  help="Compares current volume flow against 30, 60, and 90-day baselines.")
    with q3:
        st.metric("Relative Strength", f"{current_rs:.1%}", delta="vs SPY (60d)",
                  help="Performance differential vs SPY over last 60 days.")
    with q4:
        st.metric("Risk (Beta)", f"{beta:.2f}", help="Volatility relative to S&P 500. >1.0 is aggressive, <1.0 is defensive.")

    # --- CHARTS ---
    st.write("") 
    tab1, tab2 = st.tabs(["💰 Price Action", "📉 Bandwidth Analyzer"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Price',
                                 line=dict(color='#0F172A', width=1.5)))
        fig.add_trace(go.Scatter(x=upper.index, y=upper, mode='lines', name='Upper',
                                 line=dict(color='#10B981', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=lower.index, y=lower, mode='lines', name='Lower',
                                 line=dict(color='#10B981', width=1, dash='dot'),
                                 fill='tonexty', fillcolor='rgba(16, 185, 129, 0.05)'))
        fig.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name='SMA',
                                 line=dict(color='#F59E0B', width=1.5)))
        fig.update_layout(height=450, template="plotly_white",
                          margin=dict(l=0, r=0, t=10, b=0),
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        bw_fig = go.Figure()
        bw_fig.add_trace(go.Scatter(x=bandwidth.index[-180:], y=bandwidth.tail(180),
                                    mode='lines', name='Bandwidth',
                                    line=dict(color='#3B82F6', width=2)))
        bw_fig.add_hline(y=squeeze_thresh, line_dash="dash", line_color="#EF4444",
                         annotation_text=f"Threshold ({squeeze_thresh})")
        bw_fig.update_layout(height=400, template="plotly_white",
                             title="Historical Bandwidth vs Threshold",
                             hovermode="x unified")
        st.plotly_chart(bw_fig, use_container_width=True)

    # --- MONTE CARLO ---
    st.markdown("---")
    st.subheader("🎲 Monte Carlo Scenario (30 Days)")
    st.caption("Simulating 1,000 future paths based on historical volatility.")
    
    daily_vol = np.log(data / data.shift(1)).std()
    annual_vol = daily_vol * np.sqrt(252)
    
    SIMULATIONS = 1000
    DAYS = 30
    random_shocks = np.random.normal(0, daily_vol, (DAYS, SIMULATIONS))
    price_paths = current_price * (1 + random_shocks).cumprod(axis=0)
    
    final_prices = price_paths[-1]
    profitability = (final_prices > current_price).mean()
    
    mc_fig = go.Figure()
    mc_fig.add_trace(go.Scatter(
        x=np.tile(np.arange(DAYS), SIMULATIONS),
        y=price_paths.flatten(order='F'),
        mode='lines',
        line=dict(color='#10B981', width=0.5),
        opacity=0.15,
        showlegend=False,
        hoverinfo='skip'
    ))
    mc_fig.add_trace(go.Scatter(
        x=np.arange(DAYS),
        y=price_paths.mean(axis=1),
        mode='lines',
        name='Mean Path',
        line=dict(color='black', width=2)
    ))
    mc_fig.add_hline(y=current_price, line_dash="dash", line_color="black")
    mc_fig.update_layout(height=350, template="plotly_white",
                         margin=dict(l=0, r=0, t=10, b=0))
    
    c_mc1, c_mc2 = st.columns([3, 1])
    with c_mc1:
        st.plotly_chart(mc_fig, use_container_width=True)
    with c_mc2:
        st.metric("Probability of Profit", f"{profitability:.1%}", help="Percentage of 1000 scenarios that end positive.")
        st.metric("Projected Volatility", f"{annual_vol:.1%}")

    # --- EDUCATIONAL FOOTER ---
    st.markdown(f"""
    <div class="edu-footer">
        <h3>🎓 Model Architecture & Logic</h3>
        <p><strong>1. Volatility Squeeze:</strong> Identifies potential energy. BW &lt; Threshold ({squeeze_thresh}) = Squeeze.</p>
        <p><strong>2. Volume Velocity:</strong> Analyzes speed of participation vs 30/60/90d baselines.</p>
        <p><strong>3. Relative Strength:</strong> Performance differential vs SPY.</p>
        <p><strong>4. Risk Parity:</strong> Beta {beta:.2f} indicates volatility relative to market.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PROJECT 2: ML VOLATILITY (Regime Detection) - SPY ONLY, 30-YEAR BACKTEST
# ==========================================
elif project_selection == "🧠 ML Volatility (Regime Detection)":
    st.title("🧠 SPY Volatility Regime Lab")
    st.caption("Train on 30 years of SPY, validate on 2019-2023, and simulate bubble risk through 2026.")

    st.markdown("""
    <div class="highlight-box">
        <strong>Research Design:</strong><br>
        • Asset: SPY only (market proxy)<br>
        • History: ~1993 through today<br>
        • Train: up to 2018-12-31<br>
        • Test: 2019-01-01 to 2023-12-31<br>
        • Forecast: Monte Carlo paths from today to 2026-12-31 to estimate probability of entering a high-stress volatility regime ("bubble risk").
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=86400)
    def prepare_spy_vol_data():
        df = yf.download("SPY", start="1993-01-01", progress=False)
        if df.empty:
            return None, None

        data = pd.DataFrame()
        data["Close"] = df["Close"]
        data["Volume"] = df["Volume"]
        data["Return"] = np.log(data["Close"] / data["Close"].shift(1))

        # Realized volatility windows
        data["RealizedVol20"] = data["Return"].rolling(20).std() * np.sqrt(252)
        data["RealizedVol60"] = data["Return"].rolling(60).std() * np.sqrt(252)

        # Bollinger style bandwidth
        sma20 = data["Close"].rolling(20).mean()
        std20 = data["Close"].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        data["Bandwidth"] = (upper - lower) / sma20
        data["BW_Slope"] = data["Bandwidth"].diff(5)

        # Distance from moving average
        data["Dist_Mean20"] = (data["Close"] - sma20) / sma20

        # Forward volatility target (20 trading days)
        horizon = 20
        data["FutureVol20"] = data["RealizedVol20"].shift(-horizon)

        # Define training cut and threshold using training set only
        train_cutoff = pd.to_datetime("2018-12-31")
        train_mask_for_thresh = data.index <= train_cutoff
        vol_threshold = data.loc[train_mask_for_thresh, "FutureVol20"].quantile(0.90)

        # Classification label: high-stress future vol
        data["Target"] = (data["FutureVol20"] > vol_threshold).astype(int)

        data = data.dropna()
        return data, vol_threshold

    df_spy, vol_threshold = prepare_spy_vol_data()
    if df_spy is None:
        st.error("Failed to load SPY history.")
        st.stop()

    # --- TRAIN / TEST SPLIT BY CALENDAR ---
    train_cut = pd.to_datetime("2018-12-31")
    test_end = pd.to_datetime("2023-12-31")

    features = ["RealizedVol20", "RealizedVol60", "Bandwidth", "BW_Slope", "Dist_Mean20"]
    X = df_spy[features]
    y = df_spy["Target"]

    train_mask = df_spy.index <= train_cut
    test_mask = (df_spy.index > train_cut) & (df_spy.index <= test_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on 2019-2023
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    test_acc = (test_pred == y_test).mean()
    test_auc = roc_auc_score(y_test, test_proba)

    # Current regime probabilities (today)
    current_features = X.iloc[[-1]]
    prob_high_stress_next_20d = model.predict_proba(current_features)[0, 1]
    current_vol = df_spy["RealizedVol20"].iloc[-1]

    # Regime classification based on realized vol distribution
    q25 = df_spy["RealizedVol20"].quantile(0.25)
    q75 = df_spy["RealizedVol20"].quantile(0.75)
    if current_vol < q25:
        regime = "COMPLACENCY"
        regime_color = "#10B981"
    elif current_vol < q75:
        regime = "NEUTRAL / TRANSITION"
        regime_color = "#F59E0B"
    else:
        regime = "HIGH STRESS"
        regime_color = "#EF4444"

    # --- TOP ROW: MODEL OUTPUT ---
    st.subheader("🔮 Regime Forecast and Backtest")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Current Regime (SPY)", regime,
                  delta=f"Realized Vol: {current_vol:.2%}",
                  delta_color="off")
    with k2:
        st.metric("P(High-Stress Vol in 20d)", f"{prob_high_stress_next_20d:.1%}",
                  help="Model probability that 20-day forward realized volatility exceeds the 90th percentile threshold.")
    with k3:
        st.metric("Backtest Accuracy (2019-23)", f"{test_acc:.1%}",
                  help="Directionally, how often the model classified regime correctly on the out-of-sample period.")
    with k4:
        st.metric("Backtest ROC AUC (2019-23)", f"{test_auc:.2f}",
                  help="Discrimination power between calm vs high-stress future volatility.")

    # --- REGIME HISTORY AND FEATURE IMPORTANCE ---
    st.write("---")
    st.subheader("📊 Regime History and Model Logic")

    t1, t2 = st.tabs(["Regime Map (30 Years)", "Feature Importance"])

    with t1:
        fig_reg = go.Figure()
        fig_reg.add_trace(
            go.Scatter(
                x=df_spy.index,
                y=df_spy["Close"],
                mode="lines",
                name="SPY Price",
                line=dict(color="#334155", width=1)
            )
        )

        fig_reg.add_trace(
            go.Bar(
                x=df_spy.index,
                y=df_spy["RealizedVol20"] * 100,
                name="Realized Vol (20d)",
                yaxis="y2",
                marker=dict(
                    color=df_spy["RealizedVol20"],
                    colorscale="Inferno"
                ),
                opacity=0.8
            )
        )

        fig_reg.update_layout(
            height=420,
            template="plotly_white",
            title="SPY Price vs Volatility Intensity (1993 - Present)",
            yaxis=dict(title="Price"),
            yaxis2=dict(
                title="Realized Vol (20d, %)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with t2:
        importances = model.feature_importances_
        feature_df = pd.DataFrame(
            {"Feature": features, "Importance": importances}
        ).sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(
            feature_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="What drives the high-vol forecast?"
        )
        fig_imp.update_traces(marker_color="#2563EB")
        fig_imp.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Random Forest feature importance on the training set.")

    # --- MONTE CARLO TO 2026: BUBBLE RISK ---
    st.write("---")
    st.subheader("🎲 Monte Carlo Bubble Risk Through 2026")

    last_date = df_spy.index[-1]
    today = last_date.date()
    mc_end = datetime(2026, 12, 31)

    # Approximate future trading days using business days
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), mc_end)
    n_days = len(future_dates)

    if n_days <= 0:
        st.warning("End date for Monte Carlo is in the past. Adjust the end date if you change the code.")
    else:
        # Calibrate drift and volatility from last 5 years of daily log returns
        recent_window_days = 252 * 5
        recent_returns = df_spy["Return"].dropna().iloc[-recent_window_days:]
        mu = recent_returns.mean()
        sigma = recent_returns.std()
        dt = 1.0 / 252

        N_SIM = 2000
        rand = np.random.normal(0, 1, size=(n_days, N_SIM))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * rand

        log_returns_paths = drift + diffusion
        start_price = df_spy["Close"].iloc[-1]
        log_price_paths = np.log(start_price) + np.cumsum(log_returns_paths, axis=0)
        price_paths = np.exp(log_price_paths)

        # Rolling 20-day realized vol along each path
        window = 20
        path_vol = np.zeros_like(price_paths)
        for t in range(window, n_days):
            window_rets = log_returns_paths[t-window:t, :]
            path_vol[t, :] = window_rets.std(axis=0) * np.sqrt(252)

        # Bubble event: any 20-day realized vol on the path crosses training high-stress threshold
        bubble_event = (path_vol >= vol_threshold).any(axis=0)
        bubble_prob = bubble_event.mean()

        terminal_prices = price_paths[-1, :]

        c_mc1, c_mc2 = st.columns([3, 1])

        with c_mc1:
            # Plot a subset of paths for clarity
            show_paths = 100
            fig_mc = go.Figure()
            for i in range(show_paths):
                fig_mc.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=price_paths[:, i],
                        mode="lines",
                        line=dict(width=0.6, color="rgba(37, 99, 235, 0.5)"),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )
            fig_mc.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=price_paths.mean(axis=1),
                    mode="lines",
                    name="Mean Path",
                    line=dict(color="#111827", width=2)
                )
            )
            fig_mc.add_hline(
                y=start_price,
                line_dash="dash",
                line_color="#6B7280",
                annotation_text="Today"
            )
            fig_mc.update_layout(
                height=420,
                template="plotly_white",
                title="Simulated SPY Paths to 2026-12-31",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with c_mc2:
            st.metric(
                "Bubble Risk to 2026-12-31",
                f"{bubble_prob:.1%}",
                help="Fraction of Monte Carlo paths where 20-day realized volatility ever exceeds the high-stress threshold derived from 1993-2018 training data."
            )
            st.metric(
                "High-Stress Vol Threshold",
                f"{vol_threshold:.2%}",
                help="Training-set 90th percentile of forward 20-day realized volatility."
            )
            st.metric(
                "Median Terminal SPY",
                f"${np.median(terminal_prices):.0f}",
                help="Median SPY level at 2026-12-31 across all simulations."
            )

        st.markdown("""
        <div class="edu-footer">
            <h3>🎓 Interpretation for a Future CFO</h3>
            <p>
            • The classifier answers: <strong>"Is the system likely to enter a high-stress volatility regime in the next month?"</strong><br>
            • The backtest on 2019-2023 shows how stable that signal is through COVID, rate hikes, and post-2020 dynamics.<br>
            • The Monte Carlo overlay converts that into a <strong>forward probability of stress</strong> through 2026 under a GBM-style market model.
            </p>
            <p>
            In a statement on a resume or interview: you built a regime model on 30 years of SPY data,
            validated it on an out-of-sample window, and linked Monte Carlo stress probabilities
            to board-level decisions like capital raises, buybacks timing, or leverage.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PROJECT 3: FUTURE MODULE
# ==========================================
elif project_selection == "🤖 Future Module":
    st.title("🚧 Under Construction")
