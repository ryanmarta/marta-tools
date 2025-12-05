"""Streamlit entrypoint for the AI Trading Dashboard app.

This small shim keeps the original dashboard code in notebooks/AItradingDashboard.py
but lets Streamlit Community Cloud point at a simple root-level file.
"""
from pathlib import Path
import runpy

APP_PATH = Path(__file__).parent / "notebooks" / "AItradingDashboard.py"

if not APP_PATH.exists():
    raise FileNotFoundError(f"Could not find app script at {APP_PATH}")

# Execute the original Streamlit script in its own __main__ namespace
# so Streamlit picks up all top-level UI definitions.
runpy.run_path(APP_PATH, run_name="__main__")
