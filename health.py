"""Runtime and model health checks for Polar EMS."""
from __future__ import annotations
import importlib.util
import platform
import sys
from typing import Any

import config as C


def runtime_diagnostics() -> list[dict[str, Any]]:
    checks = [
        ("Python runtime", True, platform.python_version()),
        ("NumPy", importlib.util.find_spec("numpy") is not None, "Required for simulation and metrics"),
        ("Pandas", importlib.util.find_spec("pandas") is not None, "Required for time-series data"),
        ("scikit-learn", importlib.util.find_spec("sklearn") is not None, "Required for load forecasting"),
        ("Plotly", importlib.util.find_spec("plotly") is not None, "Required for dashboard visualizations"),
        ("Streamlit", importlib.util.find_spec("streamlit") is not None, "Required to launch dashboard"),
        ("PuLP", importlib.util.find_spec("pulp") is not None, "Optional: LP optimizer; fallback is used when absent"),
    ]
    return [{"component": name, "available": bool(ok), "detail": detail} for name, ok, detail in checks]


def configuration_diagnostics() -> list[dict[str, Any]]:
    checks = [
        ("Installed generation capacity", C.PV_CAPACITY_KW > 0 and C.WIND_CAPACITY_KW > 0 and C.DIESEL_CAPACITY_KW > 0, "All generation capacities must be positive"),
        ("Battery capacity", C.BATTERY_CAPACITY_KWH > 0, "Battery capacity must be positive"),
        ("Battery SOC bounds", 0 <= C.BATTERY_MIN_SOC_FRAC <= C.BATTERY_INIT_SOC_FRAC <= C.BATTERY_MAX_SOC_FRAC <= 1, "Require 0 ≤ minimum ≤ initial ≤ maximum ≤ 1"),
        ("Diesel conversion", C.KWH_PER_LITRE_DIESEL > 0 and C.CO2_KG_PER_LITRE_DIESEL > 0, "Fuel and emission conversion factors must be positive"),
        ("Latitude", -90 <= C.LATITUDE_DEG <= 90, "Latitude must be between −90° and 90°"),
    ]
    return [{"check": name, "passed": bool(ok), "detail": detail} for name, ok, detail in checks]
