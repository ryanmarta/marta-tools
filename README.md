# Streamlit AI Trading Dashboard

This repository contains the AI Quant Trading dashboard built with Streamlit.
It is configured for deployment to Streamlit Community Cloud.

## Running locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

The main app logic lives in `notebooks/AItradingDashboard.py` and is executed via
`streamlit_app.py` so Community Cloud can use a simple entrypoint.
