"""Step 3 - Fuel-minimising dispatch (Linear Program with safe fallback).

Uses PuLP/CBC when available. If PuLP's API or solver fails in the deployment
environment, a deterministic renewable-first battery dispatch is used so the demo
dashboard remains usable.
"""
import numpy as np
import pandas as pd

try:
    import pulp
except Exception:
    pulp = None

import config as C


def _fallback_dispatch(load, solar_avail, wind_avail, soc0, soc_min, soc_max):
    """Deterministic dispatch fallback: renewable-first, battery, then diesel."""
    T = len(load)
    pv_used = np.zeros(T)
    wind_used = np.zeros(T)
    diesel = np.zeros(T)
    charge = np.zeros(T)
    discharge = np.zeros(T)
    soc = np.zeros(T)

    current_soc = float(np.clip(soc0, soc_min, soc_max))
    for t in range(T):
        renewable_available = max(0.0, float(solar_avail[t])) + max(0.0, float(wind_avail[t]))
        renewable_used = min(float(load[t]), renewable_available)
        # Allocate solar first, then wind.
        pv_used[t] = min(max(0.0, float(solar_avail[t])), renewable_used)
        wind_used[t] = max(0.0, renewable_used - pv_used[t])
        surplus = max(0.0, renewable_available - float(load[t]))
        deficit = max(0.0, float(load[t]) - renewable_used)

        if surplus > 0:
            charge[t] = min(surplus, C.BATTERY_MAX_CHARGE_KW,
                            max(0.0, (soc_max - current_soc) / C.CHARGE_EFFICIENCY))
            current_soc += charge[t] * C.CHARGE_EFFICIENCY
        elif deficit > 0:
            discharge[t] = min(deficit, C.BATTERY_MAX_DISCHARGE_KW,
                             max(0.0, (current_soc - soc_min) * C.DISCHARGE_EFFICIENCY))
            current_soc -= discharge[t] / C.DISCHARGE_EFFICIENCY
            diesel[t] = max(0.0, deficit - discharge[t])

        current_soc = float(np.clip(current_soc, soc_min, soc_max))
        soc[t] = current_soc

    return pd.DataFrame({
        "load_kw": load,
        "solar_used": pv_used,
        "wind_used": wind_used,
        "discharge_kw": discharge,
        "diesel_kw": diesel,
        "charge_kw": charge,
        "soc_kwh": soc,
        "solar_avail": solar_avail,
        "wind_avail": wind_avail,
    })


def optimize_dispatch(load, solar_avail, wind_avail, soc_init_kwh=None):
    """Inputs are equal-length 1-D arrays (kW, hourly).
    Returns (dispatch_df, solver_status_string)."""
    load = np.asarray(load, dtype=float).reshape(-1)
    solar_avail = np.clip(np.asarray(solar_avail, dtype=float).reshape(-1), 0, None)
    wind_avail = np.clip(np.asarray(wind_avail, dtype=float).reshape(-1), 0, None)
    if not (len(load) == len(solar_avail) == len(wind_avail)):
        raise ValueError("load, solar_avail, and wind_avail must have equal lengths")
    T = len(load)

    soc_min = C.BATTERY_MIN_SOC_FRAC * C.BATTERY_CAPACITY_KWH
    soc_max = C.BATTERY_MAX_SOC_FRAC * C.BATTERY_CAPACITY_KWH
    soc0 = (C.BATTERY_INIT_SOC_FRAC * C.BATTERY_CAPACITY_KWH
            if soc_init_kwh is None else float(soc_init_kwh))
    soc0 = float(np.clip(soc0, soc_min, soc_max))

    if T == 0:
        empty = pd.DataFrame(columns=["load_kw", "solar_used", "wind_used",
            "discharge_kw", "diesel_kw", "charge_kw", "soc_kwh",
            "solar_avail", "wind_avail"])
        return empty, "Empty input"

    # Fall back immediately if PuLP is absent or its expected API is unavailable.
    if pulp is None or not callable(getattr(pulp, "LpVariable", None)):
        return _fallback_dispatch(load, solar_avail, wind_avail, soc0, soc_min, soc_max), "Fallback (PuLP unavailable)"

    try:
        m = pulp.LpProblem("dispatch", pulp.LpMinimize)
        pv = [pulp.LpVariable(f"pv_{t}", lowBound=0, upBound=float(solar_avail[t])) for t in range(T)]
        wnd = [pulp.LpVariable(f"wnd_{t}", lowBound=0, upBound=float(wind_avail[t])) for t in range(T)]
        dg = [pulp.LpVariable(f"dg_{t}", lowBound=0, upBound=float(C.DIESEL_CAPACITY_KW)) for t in range(T)]
        ch = [pulp.LpVariable(f"ch_{t}", lowBound=0, upBound=float(C.BATTERY_MAX_CHARGE_KW)) for t in range(T)]
        dis = [pulp.LpVariable(f"dis_{t}", lowBound=0, upBound=float(C.BATTERY_MAX_DISCHARGE_KW)) for t in range(T)]
        soc = [pulp.LpVariable(f"soc_{t}", lowBound=float(soc_min), upBound=float(soc_max)) for t in range(T)]

        m += pulp.lpSum(dg[t] for t in range(T)) + 1e-3 * pulp.lpSum(ch[t] + dis[t] for t in range(T))
        for t in range(T):
            m += pv[t] + wnd[t] + dg[t] + dis[t] == float(load[t]) + ch[t]
            prev = soc0 if t == 0 else soc[t - 1]
            m += soc[t] == prev + float(C.CHARGE_EFFICIENCY) * ch[t] - dis[t] / float(C.DISCHARGE_EFFICIENCY)

        solver = pulp.PULP_CBC_CMD(msg=0)
        status_code = m.solve(solver)
        status = pulp.LpStatus.get(status_code, "Unknown")
        if status != "Optimal":
            raise RuntimeError(f"PuLP solver returned status {status}")

        def col(variables):
            values = []
            for var in variables:
                val = var.value()
                if val is None:
                    raise RuntimeError("Solver returned an unset variable")
                values.append(float(val))
            return np.asarray(values, dtype=float)

        dispatch = pd.DataFrame({
            "load_kw": load,
            "solar_used": col(pv),
            "wind_used": col(wnd),
            "discharge_kw": col(dis),
            "diesel_kw": col(dg),
            "charge_kw": col(ch),
            "soc_kwh": col(soc),
            "solar_avail": solar_avail,
            "wind_avail": wind_avail,
        })
        return dispatch, status
    except Exception as exc:
        # A broken PuLP install, unsupported API, or solver failure should not
        # crash the public demo. Return a safe deterministic dispatch instead.
        fallback = _fallback_dispatch(load, solar_avail, wind_avail, soc0, soc_min, soc_max)
        return fallback, f"Fallback ({type(exc).__name__})"


def baseline_diesel(load, solar_avail, wind_avail):
    """Naive reference case: use renewables as they arrive, NO battery, cover the
    rest with diesel. The optimiser's saving is measured against this."""
    net = (np.asarray(load, dtype=float)
           - np.clip(np.asarray(solar_avail, dtype=float), 0, None)
           - np.clip(np.asarray(wind_avail, dtype=float), 0, None))
    return np.clip(net, 0, None)


if __name__ == "__main__":
    from data_sim import generate_year

    week = generate_year().iloc[:168]
    disp, status = optimize_dispatch(week.load_kw, week.solar_kw, week.wind_kw)
    opt = disp.diesel_kw.sum()
    base = baseline_diesel(week.load_kw, week.solar_kw, week.wind_kw).sum()
    print("solver status:", status)
    print(f"diesel, optimised: {opt:7.0f} kWh")
    print(f"diesel, baseline : {base:7.0f} kWh")
    print(f"fuel saving      : {100 * (base - opt) / base:5.1f} %")
