"""Compatibility shim for Streamlit deployments that point at Trader_V2.py."""
import runpy
from pathlib import Path

runpy.run_path(Path(__file__).with_name("trader_v2.py"), run_name="__main__")
