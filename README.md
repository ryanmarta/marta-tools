# Streamlit AI Trading Dashboard

This repository contains the AI Quant Trading dashboard built with Streamlit.
It is configured for deployment to Streamlit Community Cloud.

## Running locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Streamlit (default AI trader):
   ```bash
   streamlit run streamlit_app.py
   ```

To run an alternate version located in `notebooks/trader_v2.py`, set an
environment variable before launching Streamlit:

```bash
export STREAMLIT_APP_VARIANT=trader_v2
streamlit run streamlit_app.py
```

If your Streamlit deployment was previously configured to point directly to
`notebooks/Trader_V2.py`, that path now works as a compatibility shim that
executes the same Trader V2 app.

The main app logic lives in `notebooks/` and is executed via `streamlit_app.py`
so Community Cloud can use a simple entrypoint.
