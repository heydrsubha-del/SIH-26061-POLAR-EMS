"""Polar EMS domain services: validated metrics, alerting, and run provenance.

Pure functions only; no Streamlit dependency. This keeps operational calculations
independently testable and makes dashboard behavior deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

import config as C


REQUIRED_DISPATCH_COLUMNS = {
    "load_kw", "solar_used", "wind_used", "discharge_kw", "diesel_kw",
    "charge_kw", "soc_kwh", "solar_avail", "wind_avail",
}


@dataclass(frozen=True)
class DispatchMetrics:
    hours: int
    load_kwh: float
    renewable_kwh: float
    diesel_kwh: float
    battery_discharge_kwh: float
    battery_charge_kwh: float
    renewable_share_pct: float
    diesel_litres: float
    baseline_diesel_kwh: float
    fuel_saved_litres: float
    co2_avoided_kg: float
    average_load_kw: float
    peak_load_kw: float
    ending_soc_kwh: float
    ending_soc_pct: float
    minimum_soc_kwh: float
    diesel_active_hours: int
    solver_status: str


def validate_dispatch(dispatch: pd.DataFrame) -> None:
    """Fail early if a dispatch result does not match the dashboard contract."""
    missing = REQUIRED_DISPATCH_COLUMNS.difference(dispatch.columns)
    if missing:
        raise ValueError(f"Dispatch is missing required columns: {', '.join(sorted(missing))}")
    if not dispatch.empty:
        values = dispatch[list(REQUIRED_DISPATCH_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Dispatch contains non-finite values")
        if (dispatch[["load_kw", "solar_used", "wind_used", "discharge_kw", "diesel_kw", "charge_kw", "soc_kwh"]] < -1e-6).any().any():
            raise ValueError("Dispatch contains negative physical quantities")
        if (dispatch.soc_kwh > C.BATTERY_CAPACITY_KWH + 1e-5).any():
            raise ValueError("Battery state of charge exceeds configured capacity")
        floor = C.BATTERY_MIN_SOC_FRAC * C.BATTERY_CAPACITY_KWH
        if (dispatch.soc_kwh < floor - 1e-5).any():
            raise ValueError("Battery state of charge falls below configured reserve")


def calculate_metrics(dispatch: pd.DataFrame, solver_status: str) -> DispatchMetrics:
    validate_dispatch(dispatch)
    if dispatch.empty:
        raise ValueError("Cannot calculate operational metrics for an empty dispatch")
    load = float(dispatch.load_kw.sum())
    renew = float((dispatch.solar_used + dispatch.wind_used).sum())
    diesel = float(dispatch.diesel_kw.sum())
    base = np.maximum(
        0.0,
        dispatch.load_kw.to_numpy(dtype=float)
        - dispatch.solar_avail.to_numpy(dtype=float)
        - dispatch.wind_avail.to_numpy(dtype=float),
    )
    baseline = float(base.sum())
    saved_kwh = max(0.0, baseline - diesel)
    soc = dispatch.soc_kwh.astype(float)
    ending = float(soc.iloc[-1])
    cap = float(C.BATTERY_CAPACITY_KWH)
    diesel_l = diesel / float(C.KWH_PER_LITRE_DIESEL)
    saved_l = saved_kwh / float(C.KWH_PER_LITRE_DIESEL)
    return DispatchMetrics(
        hours=len(dispatch), load_kwh=load, renewable_kwh=renew, diesel_kwh=diesel,
        battery_discharge_kwh=float(dispatch.discharge_kw.sum()),
        battery_charge_kwh=float(dispatch.charge_kw.sum()),
        renewable_share_pct=(renew / load * 100.0 if load else 0.0),
        diesel_litres=diesel_l, baseline_diesel_kwh=baseline,
        fuel_saved_litres=saved_l,
        co2_avoided_kg=saved_l * float(C.CO2_KG_PER_LITRE_DIESEL),
        average_load_kw=float(dispatch.load_kw.mean()),
        peak_load_kw=float(dispatch.load_kw.max()),
        ending_soc_kwh=ending, ending_soc_pct=(ending / cap * 100.0 if cap else 0.0),
        minimum_soc_kwh=float(soc.min()),
        diesel_active_hours=int((dispatch.diesel_kw > 0.1).sum()),
        solver_status=str(solver_status),
    )


def build_operational_alerts(metrics: DispatchMetrics) -> list[dict[str, str]]:
    """Generate deterministic, scenario-window alerts from actual dispatch values."""
    alerts: list[dict[str, str]] = []
    reserve = C.BATTERY_MIN_SOC_FRAC * C.BATTERY_CAPACITY_KWH
    if metrics.minimum_soc_kwh <= reserve + 1.0:
        alerts.append({"severity": "warning", "code": "BATTERY_RESERVE", "title": "Battery reserve reached", "detail": f"State of charge touched the configured {C.BATTERY_MIN_SOC_FRAC:.0%} reserve floor."})
    if metrics.renewable_share_pct < 30.0:
        alerts.append({"severity": "warning", "code": "LOW_RENEWABLE_SHARE", "title": "Low renewable contribution", "detail": f"Renewables supplied {metrics.renewable_share_pct:.1f}% of demand in this window."})
    if metrics.diesel_active_hours:
        alerts.append({"severity": "info", "code": "DIESEL_DISPATCH", "title": "Backup generation dispatched", "detail": f"Diesel generation was active for {metrics.diesel_active_hours} of {metrics.hours} simulated hours."})
    if metrics.solver_status != "Optimal":
        alerts.append({"severity": "notice", "code": "FALLBACK_SOLVER", "title": "Dispatch fallback used", "detail": f"Optimizer status: {metrics.solver_status}. Verify PuLP/CBC before relying on optimization results."})
    if not alerts:
        alerts.append({"severity": "success", "code": "WINDOW_NOMINAL", "title": "No operating alerts", "detail": "The selected dispatch window remained within configured operating limits."})
    return alerts


def build_run_manifest(dispatch: pd.DataFrame, scenario: str, solver_status: str) -> dict[str, Any]:
    """Create reproducibility metadata without storing personal or external data."""
    frame = dispatch.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.index = frame.index.astype(str)
    csv_bytes = frame.to_csv(index=True).encode("utf-8")
    payload = {
        "project": "Polar EMS",
        "problem_statement": "SIH26061",
        "scenario": scenario,
        "simulated_year": int(C.SIM_YEAR),
        "random_seed": int(C.RANDOM_SEED),
        "hours": int(len(dispatch)),
        "start": str(dispatch.index.min()) if len(dispatch) else None,
        "end": str(dispatch.index.max()) if len(dispatch) else None,
        "solver_status": str(solver_status),
        "dispatch_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_mode": "seeded_simulation_not_live_scada",
    }
    return payload


def export_report(metrics: DispatchMetrics, alerts: list[dict[str, str]], manifest: dict[str, Any]) -> str:
    """Build a concise Markdown operational report from computed run values."""
    lines = [
        "# Polar EMS — Dispatch Operations Report", "",
        f"**Scenario:** {manifest['scenario']}  ",
        f"**Period:** {manifest['start']} → {manifest['end']}  ",
        f"**Solver:** {metrics.solver_status}  ",
        "**Data source:** seeded polar-station simulation (not live SCADA)", "",
        "## Energy performance", "",
        "| Metric | Result |", "|---|---:|",
        f"| Station demand | {metrics.load_kwh:,.1f} kWh |",
        f"| Renewable contribution | {metrics.renewable_share_pct:.1f}% |",
        f"| Diesel consumption | {metrics.diesel_litres:,.1f} L |",
        f"| Fuel avoided vs no-battery baseline | {metrics.fuel_saved_litres:,.1f} L |",
        f"| CO₂ avoided | {metrics.co2_avoided_kg:,.1f} kg |",
        f"| Ending battery reserve | {metrics.ending_soc_pct:.1f}% ({metrics.ending_soc_kwh:.1f} kWh) |",
        "", "## Operating alerts", "",
    ]
    for alert in alerts:
        lines.append(f"- **{alert['severity'].upper()} — {alert['title']}:** {alert['detail']}")
    lines += ["", "## Reproducibility", "", "```json", json.dumps(manifest, indent=2), "```", ""]
    return "\n".join(lines)
