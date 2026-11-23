"""Intelligent microgrid decision-support helpers for Polar EMS.

All functions operate on simulated dispatch/data. Battery wear assumptions are
transparent estimates, not manufacturer-grade SOH measurements.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config as C
from optimizer import optimize_dispatch
from ems_core import calculate_metrics

HISTORY_PATH = Path(__file__).resolve().parent / ".polar_ems_history.json"
MAX_HISTORY_RUNS = 50
ASSUMED_BATTERY_CYCLE_LIFE = 4000



def battery_health(dispatch: pd.DataFrame, prior_cycles: float = 0.0) -> dict:
    """Estimate equivalent full cycles and linear cycle-based SOH.

    EFC = discharged energy / nominal capacity. Assumption: 4,000 EFC to 80%
    capacity, represented as a linear 20% fade; this is a simplified estimate.
    """
    capacity = max(float(C.BATTERY_CAPACITY_KWH), 1e-9)
    discharge = float(dispatch.get("discharge_kw", pd.Series(dtype=float)).clip(lower=0).sum())
    charge = float(dispatch.get("charge_kw", pd.Series(dtype=float)).clip(lower=0).sum())
    cycles = discharge / capacity
    total_cycles = max(0.0, float(prior_cycles)) + cycles
    fade = min(0.20, total_cycles / ASSUMED_BATTERY_CYCLE_LIFE * 0.20)
    soh = max(0.80, 1.0 - fade)
    return {
        "discharged_kwh": discharge,
        "charged_kwh": charge,
        "equivalent_full_cycles_this_run": cycles,
        "cumulative_equivalent_cycles": total_cycles,
        "estimated_soh_pct": soh * 100,
        "estimated_capacity_kwh": capacity * soh,
        "assumption": f"Linear cycle-based estimate: {ASSUMED_BATTERY_CYCLE_LIFE:,} equivalent full cycles to 80% SOH; no calendar, temperature, or chemistry model.",
    }


def maintenance_signals(dispatch: pd.DataFrame, battery: dict) -> list[dict[str, str]]:
    """Generate explainable maintenance indicators from available simulation data."""
    signals = []
    diesel_hours = int((dispatch.get("diesel_kw", pd.Series(dtype=float)) > 0.1).sum())
    discharge = float(dispatch.get("discharge_kw", pd.Series(dtype=float)).sum())
    charge = float(dispatch.get("charge_kw", pd.Series(dtype=float)).sum())
    if diesel_hours >= max(1, int(len(dispatch) * 0.60)):
        signals.append({"priority": "HIGH", "asset": "Diesel genset", "signal": "High runtime", "basis": f"Diesel active {diesel_hours}/{len(dispatch)} hours; consider inspection and fuel planning."})
    elif diesel_hours:
        signals.append({"priority": "WATCH", "asset": "Diesel genset", "signal": "Runtime observed", "basis": f"Diesel active {diesel_hours} hours in this simulated window."})
    if battery["estimated_soh_pct"] <= 85:
        signals.append({"priority": "HIGH", "asset": "Battery", "signal": "Estimated capacity fade", "basis": f"Simplified cycle estimate is {battery['estimated_soh_pct']:.1f}% SOH; verify with real battery telemetry."})
    elif battery["equivalent_full_cycles_this_run"] >= 0.5:
        signals.append({"priority": "WATCH", "asset": "Battery", "signal": "Cycling activity", "basis": f"{battery['equivalent_full_cycles_this_run']:.2f} EFC this run; review cycle/load patterns."})
    if len(dispatch) and "solar_avail" in dispatch and float(dispatch.solar_avail.max()) > 0:
        solar_use = float(dispatch.get("solar_used", pd.Series(dtype=float)).sum())
        solar_avail = float(dispatch.solar_avail.sum())
        if solar_avail > 0 and solar_use / solar_avail < 0.65:
            signals.append({"priority": "WATCH", "asset": "PV array", "signal": "Curtailment / utilization review", "basis": f"Solar used {solar_use/solar_avail:.0%} of simulated available energy; inspect dispatch or storage headroom."})
    if not signals:
        signals.append({"priority": "OK", "asset": "System", "signal": "No simulated maintenance flags", "basis": "No threshold-based signal was triggered. This is not a physical inspection or condition-monitoring result."})
    return signals



def run_stress_scenario(data: pd.DataFrame, load_factor: float = 1.0,
                        solar_factor: float = 1.0, wind_factor: float = 1.0,
                        label: str = "Scenario") -> dict:
    """Dispatch a bounded scenario with load and renewable availability multipliers."""
    if data.empty:
        raise ValueError("Scenario data cannot be empty")
    if not (0.5 <= load_factor <= 1.8 and 0.0 <= solar_factor <= 1.5 and 0.0 <= wind_factor <= 1.5):
        raise ValueError("Scenario multipliers outside safe bounds")
    load = np.clip(data.load_kw.to_numpy(dtype=float) * load_factor, 0, C.DIESEL_CAPACITY_KW)
    solar = np.clip(data.solar_kw.to_numpy(dtype=float) * solar_factor, 0, C.PV_CAPACITY_KW)
    wind = np.clip(data.wind_kw.to_numpy(dtype=float) * wind_factor, 0, C.WIND_CAPACITY_KW)
    dispatch, status = optimize_dispatch(load, solar, wind)
    dispatch.index = data.index
    metrics = calculate_metrics(dispatch, status)
    return {
        "scenario": str(label), "solver_status": status, "hours": len(data),
        "load_factor": load_factor, "solar_factor": solar_factor, "wind_factor": wind_factor,
        "diesel_kwh": metrics.diesel_kwh, "diesel_litres": metrics.diesel_litres,
        "renewable_share_pct": metrics.renewable_share_pct,
        "ending_soc_pct": metrics.ending_soc_pct,
        "co2_kg": metrics.diesel_litres * float(C.CO2_KG_PER_LITRE_DIESEL),
    }


def forecast_aware_dispatch(data: pd.DataFrame, reserve_pct: float = 5.0) -> tuple[pd.DataFrame, str]:
    """Dispatch against a conservative load envelope to reserve capacity.

    Uses a user-set forecast margin (bounded 0–25%) rather than claiming a new
    future ML forecast. This is a transparent what-if load envelope.
    """
    if not 0 <= reserve_pct <= 25:
        raise ValueError("Forecast reserve must be between 0 and 25 percent")
    adjusted_load = data.load_kw.to_numpy(dtype=float) * (1 + reserve_pct / 100.0)
    adjusted_load = np.clip(adjusted_load, 0, C.DIESEL_CAPACITY_KW)
    dispatch, status = optimize_dispatch(adjusted_load, data.solar_kw, data.wind_kw)
    dispatch.index = data.index
    return dispatch, status



def load_history(path: Path | str = HISTORY_PATH) -> list[dict]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return raw[-MAX_HISTORY_RUNS:]
    except (OSError, ValueError, TypeError):
        return []


def save_history_run(run: dict, path: Path | str = HISTORY_PATH) -> list[dict]:
    """Append a small summary, retaining only the newest bounded run records."""
    path = Path(path)
    history = load_history(path)
    record = dict(run)
    record.setdefault("saved_at_utc", datetime.now(timezone.utc).isoformat())
    history.append(record)
    history = history[-MAX_HISTORY_RUNS:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    except OSError:
        # The UI can still operate if the deployment filesystem is read-only.
        pass
    return history


def build_run_summary(dispatch: pd.DataFrame, scenario: str, solver_status: str) -> dict:
    metrics = calculate_metrics(dispatch, solver_status)
    battery = battery_health(dispatch)
    return {
        "scenario": scenario,
        "solver_status": solver_status,
        "hours": metrics.hours,
        "load_kwh": metrics.load_kwh,
        "diesel_kwh": metrics.diesel_kwh,
        "diesel_litres": metrics.diesel_litres,
        "renewable_share_pct": metrics.renewable_share_pct,
        "ending_soc_pct": metrics.ending_soc_pct,
        "battery_efc": battery["equivalent_full_cycles_this_run"],
        "battery_soh_pct": battery["estimated_soh_pct"],
    }
