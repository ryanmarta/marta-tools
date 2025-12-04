"""
NEXUS v13.1: QUANTITATIVE STRATEGIST ENGINE (Clean Theme)
---------------------------------------------------------
Author: Nexus AI
Version: 13.1 (Stable Serialization + Pro Grey UI)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import warnings

# --- 1. QUANT CONFIGURATION ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Nexus Quant | Inst. Grade", 
    layout="wide", 
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (PRO-GREY THEME) ---
# High visibility, professional contrast, no pitch black backgrounds
st.markdown("""
    <style>
        /* Global Background - Neutral Light Grey/Blue Tint */
        .stApp { background-color: #F3F4F6; color: #1F2937; font-family: 'Inter', sans-serif; }
        
        /* Sidebar - Slight Darker Grey for separation */
        section[data-testid="stSidebar"] { background-color: #E5E7EB; border-right: 1px solid #D1D5DB; }
        
        /* Metrics - White Cards with Shadows */
        div.stMetric {
            background-color: #FFFFFF;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        div.stMetric > div { color: #111827 !important; } /* Dark Text for readability */
        div[data-testid="stMetricLabel"] { color: #6B7280 !important; }
        
        /* Tables - Clean White Background */
        .dataframe { 
            background-color: #FFFFFF !important; 
            color: #374151 !important;
            font-family: 'Courier New', monospace; 
        }
        
        /* Custom Boxes */
        .quant-box {
            background-color: #FFFFFF;
            border-left: 5px solid #10B981;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            color: #1F2937;
        }
        .risk-box {
            background-color: #FEF2F2;
            border-left: 5px solid #EF4444;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 6px;
            color: #991B1B;
        }
        
        /* Headings */
        h1, h2, h3 { color: #111827 !important; }
        
        /* Inputs */
        div.stTextInput > div > div > input { color: #1F2937; background-color: #FFFFFF; }
        div.stSelectbox > div > div { color: #1F2937; background-color: #FFFFFF; }
        
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CLASS: MARKET DATA HANDLER (The Data Feed)
# ==============================================================================
class MarketData:
    """
    Handles robust data fetching, cleaning, and advanced volatility estimation.
    Implements Garman-Klass Volatility for superior variance estimation.
    """
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.obj = yf.Ticker(self.ticker)
        
    @st.cache_data(ttl=300)
    def fetch_history(_self, period="1y"):
        """Fetches OHLC data."""
        df = _self.obj.history(period=period)
        if df.empty: return None
        return df
    
    @st.cache_data(ttl=300)
    def fetch_option_chain(_self, expiry_date):
        """
        Fetches and structures the option chain.
        CRITICAL FIX: Returns clean DataFrames only, avoiding serialization errors.
        """
        try:
            # We access the internal object but convert immediately to DF
            chain = _self.obj.option_chain(expiry_date)
            calls = chain.calls
            puts = chain.puts
            
            # Tag them before merging
            calls['type'] = 'call'
            puts['type'] = 'put'
            
            # Clean Data
            full_chain = pd.concat([calls, puts])
            full_chain['mid'] = (full_chain['bid'] + full_chain['ask']) / 2
            
            # Filter noise
            full_chain = full_chain[full_chain['mid'] > 0.01] 
            
            return full_chain
        except Exception as e:
            return None

    def get_spot(self):
        """Get live spot price."""
        try:
            hist = self.obj.history(period='1d')
            if not hist.empty:
                return hist['Close'].iloc[-1]
            return 0.0
        except: return 0.0

    def get_expirations(self):
        """Get available expiry dates."""
        return self.obj.options

    @staticmethod
    def garman_klass_volatility(df):
        """
        Calculates Garman-Klass Volatility.
        Considers Open, High, Low, Close for better estimation efficiency than Std Dev.
        Ref: Garman, M. B., & Klass, M. J. (1980).
        """
        if df is None or len(df) < 2: return 0.0
        
        # Avoid divide by zero logs
        df = df.replace(0, np.nan).dropna()
        
        log_hl = np.log(df['High'] / df['Low']) ** 2
        log_co = np.log(df['Close'] / df['Open']) ** 2
        
        # GK Formula: 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # Annualize
        return np.sqrt(gk_var.mean()) * np.sqrt(252)

# ==============================================================================
# CLASS: BLACK-SCHOLES MATH KERNEL (The Pricing Engine)
# ==============================================================================
class BlackScholesEngine:
    """
     Vectorized Black-Scholes-Merton Model.
     Includes First Order (Delta, Vega) AND Second Order Greeks (Vanna, Volga).
    """
    @staticmethod
    def d1(S, K, T, r, sigma):
        # Guard against zero vol or time
        if sigma <= 0 or T <= 0: return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        if sigma <= 0 or T <= 0: return 0
        return BlackScholesEngine.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def price(S, K, T, r, sigma, type="call"):
        if T <= 0 or sigma <= 0: return max(S-K, 0) if type == "call" else max(K-S, 0)
        
        d1 = BlackScholesEngine.d1(S, K, T, r, sigma)
        d2 = BlackScholesEngine.d2(S, K, T, r, sigma)
        
        if type == "call":
            return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    @staticmethod
    def get_greeks(S, K, T, r, sigma, type="call"):
        """Calculates comprehensive Greek suite."""
        if T <= 0 or sigma <= 0: 
            return {k: 0.0 for k in ["Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga", "Charm"]}

        d1 = BlackScholesEngine.d1(S, K, T, r, sigma)
        d2 = BlackScholesEngine.d2(S, K, T, r, sigma)
        norm_pdf_d1 = stats.norm.pdf(d1)
        norm_cdf_d1 = stats.norm.cdf(d1)
        
        # 1. First Order
        if type == "call":
            delta = norm_cdf_d1
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
            theta = (-S * norm_pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365.0
        else:
            delta = norm_cdf_d1 - 1
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)
            theta = (-S * norm_pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365.0
            
        gamma = norm_pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * norm_pdf_d1 * np.sqrt(T) / 100.0 # Scaled for % vol change
        
        # 2. Second Order (The Quant Edge)
        vanna = -norm_pdf_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma
        
        # Charm approximation
        charm = -norm_pdf_d1 * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
            
        return {
            "Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho,
            "Vanna": vanna, "Volga": volga, "Charm": charm
        }

# ==============================================================================
# CLASS: VOLATILITY SURFACE MODELER (The Smile)
# ==============================================================================
class VolatilitySurface:
    """
    Constructs a smooth volatility surface from discrete market data using
    Cubic Spline Interpolation. Fixes 'broken' chains and visualizes Skew.
    """
    def __init__(self, spot, chain):
        self.spot = spot
        self.chain = chain
        self.model = None
        
    def fit_smile(self):
        """Fits a cubic spline to the Implied Volatility vs Strike curve."""
        # 1. Filter Data
        if self.chain is None or self.chain.empty: return None
        
        df = self.chain.copy()
        # Focus on reasonable IVs
        df = df[(df['impliedVolatility'] > 0.05) & (df['impliedVolatility'] < 3.0)]
        df = df[df['volume'] > 0] # Liquidity filter
        
        if len(df) < 5: return None
        
        # Sort by strike
        df = df.sort_values('strike')
        
        # 2. Interpolation (Cubic Spline)
        try:
            # We average IVs if multiple options exist at same strike
            df_unique = df.groupby('strike')['impliedVolatility'].mean().reset_index()
            if len(df_unique) < 4: return None
            
            self.model = CubicSpline(df_unique['strike'], df_unique['impliedVolatility'], bc_type='natural')
            return self.model
        except:
            return None

# ==============================================================================
# CLASS: MONTE CARLO PRICER (The Simulation)
# ==============================================================================
class MonteCarloEngine:
    """
    Advanced Monte Carlo Pricer with Variance Reduction (Antithetic Variates).
    """
    def __init__(self, S, K, T, r, sigma, simulations=10000):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = simulations
        
    def price_european(self, type="call"):
        if self.T <= 0 or self.sigma <= 0: return 0.0, 0.0
        
        dt = self.T 
        # Antithetic Variates for variance reduction
        z = np.random.standard_normal(int(self.N / 2)) 
        z = np.concatenate((z, -z))
        
        # Terminal Price Calculation
        ST = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + 
                             self.sigma * np.sqrt(self.T) * z)
                             
        if type == "call":
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)
            
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.N)
        return price, std_error

# ==============================================================================
# MAIN APP LOGIC (STREAMLIT)
# ==============================================================================

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("🎛️ Quant Controls")
    ticker = st.text_input("Ticker Symbol", value="TSLA").upper()
    
    st.subheader("Market Parameters")
    risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.25, 0.05) / 100.0
    
    st.subheader("Execution")
    capital = st.number_input("Portfolio Capital ($)", 1000, 1000000, 50000, 1000)
    kelly_frac = st.slider("Kelly Fraction (Risk)", 0.1, 1.0, 0.5, 0.1)

# --- INIT CLASSES ---
market = MarketData(ticker)
bs_engine = BlackScholesEngine()

# --- 1. MARKET OVERVIEW & VOLATILITY REGIME ---
st.title(f"🏛️ Quant Desk: {ticker}")

# A. Fetch History & Calc Volatility
hist_data = market.fetch_history(period="6mo")
if hist_data is not None:
    spot_price = hist_data['Close'].iloc[-1]
    
    # Calc Garman-Klass Vol (The Pro Metric)
    gk_vol = market.garman_klass_volatility(hist_data)
    
    # Calc Simple Close-to-Close (The Retail Metric)
    simple_vol = hist_data['Close'].pct_change().std() * np.sqrt(252)
    
    # Display Regime
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot Price", f"${spot_price:.2f}", f"{hist_data['Close'].pct_change().iloc[-1]:.2%}")
    c2.metric("Garman-Klass Vol", f"{gk_vol:.2%}", delta=f"{gk_vol - simple_vol:.2%} vs StdDev", help="Intraday Volatility Estimator")
    c3.metric("Daily Range (ATR)", f"${(hist_data['High']-hist_data['Low']).rolling(14).mean().iloc[-1]:.2f}")
    c4.metric("Market Regime", "HIGH VOL" if gk_vol > 0.25 else "NORMAL")

    # Volatility Cone (Visualizing Vol vs Time)
    with st.expander("📉 Volatility Analysis (Garman-Klass vs Realized)", expanded=False):
        # Calculate Rolling GK Vol
        rolling_gk = []
        # Safe rolling calculation
        if len(hist_data) > 30:
            for i in range(30, len(hist_data)):
                window = hist_data.iloc[i-30:i]
                rolling_gk.append(market.garman_klass_volatility(window))
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(y=rolling_gk, x=hist_data.index[30:], name="30D Garman-Klass Vol", line=dict(color='#2563EB')))
            fig_vol.update_layout(title="Historical Volatility Regime", height=300, template="plotly_white")
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Not enough data for volatility chart.")

# --- 2. OPTION CHAIN & SURFACE ---
st.markdown("---")
st.subheader("🧬 Option Chain & Volatility Surface")

# Select Expiry
expirations = market.get_expirations()
if expirations:
    selected_exp = st.selectbox("Select Expiration", expirations, index=0)
    
    # Fetch Chain
    chain = market.fetch_option_chain(selected_exp)
    
    if chain is not None and not chain.empty:
        # Time to Expiry
        exp_dt = datetime.strptime(selected_exp, "%Y-%m-%d")
        days_to_exp = (exp_dt.date() - date.today()).days
        T_years = max(days_to_exp / 365.0, 0.001)

        # Build Volatility Surface
        vol_model = VolatilitySurface(spot_price, chain)
        spline = vol_model.fit_smile()
        
        # Layout: Table | Smile Chart
        col_tbl, col_chart = st.columns([1, 2])
        
        with col_tbl:
            # Liquidity Filter for Display
            liq_chain = chain[(chain['volume'] > 10) & (chain['strike'] > spot_price * 0.8) & (chain['strike'] < spot_price * 1.2)]
            # Clean formatting for table
            disp_df = liq_chain[['contractSymbol', 'type', 'strike', 'lastPrice', 'impliedVolatility', 'volume']].copy()
            disp_df['impliedVolatility'] = disp_df['impliedVolatility'].map('{:.2%}'.format)
            disp_df['lastPrice'] = disp_df['lastPrice'].map('${:.2f}'.format)
            
            st.dataframe(disp_df, height=400, hide_index=True)
            
        with col_chart:
            # Plot Smile
            if chain['strike'].nunique() > 1:
                strikes_plot = np.linspace(chain['strike'].min(), chain['strike'].max(), 100)
                fig_smile = go.Figure()
                
                # Raw Data
                fig_smile.add_trace(go.Scatter(
                    x=chain['strike'], 
                    y=chain['impliedVolatility'], 
                    mode='markers', 
                    name='Market IV', 
                    marker=dict(color='#9CA3AF', opacity=0.6, size=6)
                ))
                
                # Smoothed Curve
                if spline:
                    smooth_vols = spline(strikes_plot)
                    fig_smile.add_trace(go.Scatter(
                        x=strikes_plot, 
                        y=smooth_vols, 
                        mode='lines', 
                        name='Cubic Spline Model', 
                        line=dict(color='#10B981', width=3)
                    ))
                
                fig_smile.add_vline(x=spot_price, line_dash="dash", line_color="#EF4444", annotation_text="SPOT")
                fig_smile.update_layout(
                    title=f"Volatility Skew ({selected_exp})", 
                    xaxis_title="Strike", 
                    yaxis_title="Implied Vol", 
                    template="plotly_white", 
                    height=400
                )
                st.plotly_chart(fig_smile, use_container_width=True)
            else:
                st.info("Insufficient strikes to plot smile.")
                
        # --- 3. STRATEGIST ENGINE ---
        st.markdown("---")
        st.subheader("🧠 Trade Strategist & Risk Engine")
        
        # Inputs for Specific Trade
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            target_strike = st.selectbox("Strike to Trade", sorted(chain['strike'].unique()), index=len(chain['strike'].unique())//2)
        with sc2:
            opt_type = st.radio("Type", ["Call", "Put"], horizontal=True)
        with sc3:
            iv_override = st.checkbox("Use Model IV (Smoothed)", value=True)
            
        # Get Contract Data
        try:
            # Safe Filtering
            filtered_contract = chain[(chain['strike'] == target_strike) & (chain['type'] == opt_type.lower())]
            
            if not filtered_contract.empty:
                contract_row = filtered_contract.iloc[0]
                market_price = contract_row['lastPrice']
                market_iv = contract_row['impliedVolatility']
                
                # Decide which Vol to use
                sigma_calc = float(spline(target_strike)) if (iv_override and spline) else market_iv
                # Safety clamp for vol
                if sigma_calc <= 0: sigma_calc = market_iv
                
                # --- A. PRICING MODELS ---
                # 1. Black Scholes
                bs_price = bs_engine.price(spot_price, target_strike, T_years, risk_free, sigma_calc, opt_type.lower())
                greeks = bs_engine.get_greeks(spot_price, target_strike, T_years, risk_free, sigma_calc, opt_type.lower())
                
                # 2. Monte Carlo (Verification)
                mc_engine = MonteCarloEngine(spot_price, target_strike, T_years, risk_free, sigma_calc, simulations=10000)
                mc_price, mc_error = mc_engine.price_european(opt_type.lower())
                
                # --- B. KELLY CRITERION (Position Sizing) ---
                # Logic: Edge = Theoretical Price - Market Price
                edge = bs_price - market_price
                win_prob = abs(greeks['Delta']) # Crude proxy for Win Rate
                odds = 2.0 # Assuming 2:1 Payoff for simplicity of display
                
                # Kelly logic
                if edge > 0:
                    kelly_pct = (win_prob * odds - (1 - win_prob)) / odds
                    kelly_pct = max(0, kelly_pct) * kelly_frac # Half Kelly
                    capital_alloc = capital * kelly_pct
                    contracts_rec = int(capital_alloc / (market_price * 100)) if market_price > 0 else 0
                else:
                    kelly_pct = 0
                    capital_alloc = 0
                    contracts_rec = 0
                    
                # --- OUTPUT DASHBOARD ---
                
                # Row 1: Valuation
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Market Price", f"${market_price:.2f}")
                k2.metric("Black-Scholes Model", f"${bs_price:.2f}", delta=f"{bs_price - market_price:.2f} Edge")
                k3.metric("Monte Carlo (10k)", f"${mc_price:.2f}", help=f"Std Error: {mc_error:.3f}")
                k4.metric("Implied Vol Used", f"{sigma_calc:.2%}")
                
                # Row 2: Advanced Greeks
                st.markdown("#### 📐 Second-Order Greeks (Risk Sensitivities)")
                g1, g2, g3, g4, g5, g6 = st.columns(6)
                g1.metric("Delta (Δ)", f"{greeks['Delta']:.3f}")
                g2.metric("Gamma (Γ)", f"{greeks['Gamma']:.3f}")
                g3.metric("Vega (ν)", f"{greeks['Vega']:.3f}")
                g4.metric("Vanna (dΔ/dσ)", f"{greeks['Vanna']:.3f}", help="Change in Delta for 1% change in Vol.")
                g5.metric("Volga (dν/dσ)", f"{greeks['Volga']:.3f}", help="Change in Vega for 1% change in Vol.")
                g6.metric("Charm (dΔ/dt)", f"{greeks['Charm']:.3f}", help="Delta decay over time.")

                # Row 3: Strategy & Execution
                st.markdown("#### 🤖 Algorithmic Execution")
                
                if edge > 0.05:
                    # Positive Edge
                    st.markdown(f"""
                    <div class="quant-box">
                        <h3 style="margin:0; color:#065F46;">BUY / LONG SIGNAL</h3>
                        <p style="margin:5px 0;">Model value (${bs_price:.2f}) > Market (${market_price:.2f}). Positive Expectancy.</p>
                        <hr style="border-color: #D1D5DB;">
                        <strong>Kelly Criterion Sizing (f* = {kelly_pct:.1%}):</strong><br>
                        Allocate <strong>${capital_alloc:.2f}</strong> ({contracts_rec} Contracts).<br>
                        <span style="color:#6B7280; font-size:0.9em;">Based on {kelly_frac}x Fractional Kelly with estimated Win Probability of {win_prob:.1%}.</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif edge < -0.05:
                    # Negative Edge
                    st.markdown(f"""
                    <div class="risk-box">
                        <h3 style="margin:0; color:#991B1B;">SELL / AVOID</h3>
                        <p style="margin:5px 0;">Model value (${bs_price:.2f}) < Market (${market_price:.2f}). Premium is overpriced relative to model.</p>
                        <hr style="border-color: #FECACA;">
                        <strong>Recommendation:</strong> Do not buy long. Consider selling premium (Credit Spreads) if holding underlying.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Neutral
                    st.info(f"⚖️ FAIR VALUE: Market price (${market_price:.2f}) aligns with Model (${bs_price:.2f}). No statistical edge found.")
                
                # Greeks Hedge Calculator
                if contracts_rec > 0:
                    with st.expander("🛡️ Dynamic Hedging Calculator (Delta Neutral)"):
                        st.write("To neutralise directional risk on this position:")
                        hedge_shares = int(abs(greeks['Delta']) * 100 * contracts_rec)
                        if opt_type == "Call":
                            st.warning(f"📉 SHORT {hedge_shares} shares of {ticker}.")
                        else:
                            st.warning(f"📈 LONG {hedge_shares} shares of {ticker}.")
                        st.caption(f"Net Gamma: {greeks['Gamma']*100*contracts_rec:.2f} | Net Vega: {greeks['Vega']*100*contracts_rec:.2f}")
            else:
                st.warning(f"No {opt_type} contract found at Strike ${target_strike}.")

        except Exception as e:
            st.error(f"Computation Error: {e}")
            
    else:
        st.warning("No valid option data found for this expiry.")
else:
    st.warning("No option data found. Market may be closed or ticker invalid.")