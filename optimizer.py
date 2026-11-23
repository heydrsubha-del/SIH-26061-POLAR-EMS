"""Step 3 - Fuel-minimising dispatch (Linear Program).

For every hour we decide how much power to take from solar, wind, the battery,
and the diesel genset so that TOTAL DIESEL is minimised while the battery obeys
its physics (state-of-charge limits, charge/discharge limits, efficiency).

Because diesel capacity exceeds peak load, a feasible answer ALWAYS exists - the
solver can never fail on stage. Solved with PuLP + the bundled CBC solver
(no separate solver install needed).

Sanity-check standalone:   python optimizer.py
"""
import numpy as np
import pandas as pd
import pulp

import config as C


def optimize_dispatch(load, solar_avail, wind_avail, soc_init_kwh=None):
    """Inputs are equal-length 1-D arrays (kW, hourly).
    Returns (dispatch_df, solver_status_string)."""
    load = np.asarray(load, dtype=float)
    solar_avail = np.clip(np.asarray(solar_avail, dtype=float), 0, None)
    wind_avail = np.clip(np.asarray(wind_avail, dtype=float), 0, None)
    T = len(load)

    soc_min = C.BATTERY_MIN_SOC_FRAC * C.BATTERY_CAPACITY_KWH
    soc_max = C.BATTERY_MAX_SOC_FRAC * C.BATTERY_CAPACITY_KWH
    soc0 = (C.BATTERY_INIT_SOC_FRAC * C.BATTERY_CAPACITY_KWH
            if soc_init_kwh is None else soc_init_kwh)

    m = pulp.LpProblem("dispatch", pulp.LpMinimize)
    pv = pulp.LpVariable.dicts("pv", range(T), lowBound=0)
    wnd = pulp.LpVariable.dicts("wnd", range(T), lowBound=0)
    dg = pulp.LpVariable.dicts("dg", range(T), lowBound=0, upBound=C.DIESEL_CAPACITY_KW)
    ch = pulp.LpVariable.dicts("ch", range(T), lowBound=0, upBound=C.BATTERY_MAX_CHARGE_KW)
    dis = pulp.LpVariable.dicts("dis", range(T), lowBound=0, upBound=C.BATTERY_MAX_DISCHARGE_KW)
    soc = pulp.LpVariable.dicts("soc", range(T), lowBound=soc_min, upBound=soc_max)

    # Objective: minimise diesel energy. Tiny penalty discourages pointless
    # simultaneous charge+discharge so the schedule stays clean.
    m += (pulp.lpSum(dg[t] for t in range(T))
          + 1e-3 * pulp.lpSum(ch[t] + dis[t] for t in range(T)))

    for t in range(T):
        pv[t].upBound = float(solar_avail[t])    # cannot use more than available
        wnd[t].upBound = float(wind_avail[t])
        # Power balance: everything supplied == load served + battery charging
        m += pv[t] + wnd[t] + dg[t] + dis[t] == load[t] + ch[t]
        # Battery state of charge (1-hour steps, so kW * 1h == kWh)
        prev = soc0 if t == 0 else soc[t - 1]
        m += soc[t] == prev + C.CHARGE_EFFICIENCY * ch[t] - dis[t] / C.DISCHARGE_EFFICIENCY

    m.solve(pulp.PULP_CBC_CMD(msg=0))

    def col(v):
        return np.array([v[t].value() or 0.0 for t in range(T)])

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
    return dispatch, pulp.LpStatus[m.status]


def baseline_diesel(load, solar_avail, wind_avail):
    """Naive reference case: use renewables as they arrive, NO battery, cover the
    rest with diesel. The optimiser's saving is measured against this."""
    net = (np.asarray(load, dtype=float)
           - np.clip(solar_avail, 0, None)
           - np.clip(wind_avail, 0, None))
    return np.clip(net, 0, None)


if __name__ == "__main__":
    from data_sim import generate_year

    week = generate_year().iloc[:168]     # first week
    disp, status = optimize_dispatch(week.load_kw, week.solar_kw, week.wind_kw)
    opt = disp.diesel_kw.sum()
    base = baseline_diesel(week.load_kw, week.solar_kw, week.wind_kw).sum()
    print("solver status:", status, "(should be 'Optimal')")
    print(f"diesel, optimised: {opt:7.0f} kWh")
    print(f"diesel, baseline : {base:7.0f} kWh")
    print(f"fuel saving      : {100 * (base - opt) / base:5.1f} %")
