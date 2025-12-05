"""Streamlit Trader V2 app using Yahoo Finance data."""
import datetime as dt
from functools import lru_cache

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Trader V2", page_icon="📈", layout="wide")


def _default_dates():
    today = dt.date.today()
    start = today - dt.timedelta(days=365)
    return start, today


def _fetch_data(ticker: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    history = yf.Ticker(ticker).history(start=start, end=end + dt.timedelta(days=1), interval=interval)
    history.index = history.index.tz_localize(None)
    return history


@lru_cache(maxsize=32)
def fetch_data_cached(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end)
    return _fetch_data(ticker, start_date, end_date, interval)


def main():
    st.title("Trader V2 – Quick Market Overview")
    st.markdown("Compare tickers with Yahoo Finance data and simple stats.")

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", value="AAPL").upper().strip() or "AAPL"
        comparison = st.text_input("Compare with (optional)", value="").upper().strip()
        interval = st.selectbox("Interval", ["1d", "1h", "15m"], index=0)
        start_default, end_default = _default_dates()
        start_date = st.date_input("Start date", value=start_default)
        end_date = st.date_input("End date", value=end_default)

    def render_chart(label: str, df: pd.DataFrame):
        if df.empty:
            st.warning(f"No data for {label}.")
            return
        fig = px.line(df.reset_index(), x="Date", y="Close", title=f"{label} Close Price")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[["Close", "Volume"]].tail(), use_container_width=True)

    cols = st.columns(2)

    with cols[0]:
        if ticker:
            data = fetch_data_cached(ticker, start_date.isoformat(), end_date.isoformat(), interval)
            render_chart(ticker, data)

    with cols[1]:
        if comparison:
            comp_data = fetch_data_cached(comparison, start_date.isoformat(), end_date.isoformat(), interval)
            render_chart(comparison, comp_data)


if __name__ == "__main__":
    main()
