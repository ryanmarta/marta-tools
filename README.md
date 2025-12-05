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

## Streamlit Community Cloud startup timing

Streamlit Community Cloud may spend a minute or two provisioning a machine and
installing dependencies before your app responds. During that window the logs
show steps like "Provisioning machine", "Processing dependencies", and the
`uv pip install` summary. If the log output stalls for 10+ minutes or ends with
an error, select **Rerun** in the Community Cloud UI to restart the build; no
code changes are required unless the log explicitly reports a failure.

### "ModuleNotFoundError: No module named 'yfinance'"

`yfinance` is listed in `requirements.txt`, so a fresh Streamlit deployment
should install it automatically. Trader V2 will also try to auto-install it at
runtime and surface the pip output if that fails. If you see this error:

- Reboot the app in Community Cloud to trigger a clean reinstall of
  dependencies.
- If you recently merged changes, ensure the deployment is pointing at the
  latest commit on `main` so it sees the updated requirements.

The app now surfaces a clearer message when `yfinance` is missing so you know
the failure is coming from dependency installation rather than the app logic.

## About this Codespace notice

If you see the GitHub message saying everything is contained in this
codespace and you can publish the branch later, it simply means your work is
local to this temporary cloud environment. Nothing is pushed to GitHub until
you choose **Publish Branch** (or delete the codespace to discard it). The
Streamlit app and all changes you make here stay private unless you publish.
The main app logic lives in `notebooks/AItradingDashboard.py` and is executed via
`streamlit_app.py` so Community Cloud can use a simple entrypoint.
