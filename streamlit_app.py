"""Streamlit entrypoint for the AI Trading Dashboard apps.

This shim lets Streamlit Community Cloud point at a simple root-level file
while still running the real app code from ``notebooks/``. Use the
``STREAMLIT_APP_VARIANT`` environment variable to load alternate versions
such as ``trader_v2`` when present.
"""
import os
from pathlib import Path
import runpy

APP_VARIANTS = {
    "ai_trader": "AItradingDashboard.py",
    "trader_v2": "trader_v2.py",
    "trader_v2.py": "trader_v2.py",
    "trader_v2_cap": "Trader_V2.py",  # compatibility for legacy Streamlit entrypoint
    "trader_v2_cap.py": "Trader_V2.py",
}

variant_key = os.getenv("STREAMLIT_APP_VARIANT", "ai_trader").lower()
script_name = APP_VARIANTS.get(variant_key, variant_key)

# Allow users to pass the bare variant name ("trader_v2") or the filename
# ("trader_v2.py") via the environment variable.
if not script_name.endswith(".py"):
    script_name = f"{script_name}.py"

app_path = Path(__file__).parent / "notebooks" / script_name

if not app_path.exists():
    known = ", ".join(APP_VARIANTS.values())
    raise FileNotFoundError(
        "Could not find app script at "
        f"{app_path}. Set STREAMLIT_APP_VARIANT to one of: {known}."
    )

# Execute the target Streamlit script in its own __main__ namespace so
# Streamlit picks up all top-level UI definitions.
runpy.run_path(app_path, run_name="__main__")
