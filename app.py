"""Step 4 - The dashboard (this is your live demo).

Run:   streamlit run app.py

Nothing computes live except a tiny, instant optimisation for the window you
pick, so the demo is fast and cannot break on stage.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from data_sim import generate_year
from forecast import forecast_load
from optimizer import optimize_dispatch, baseline_diesel

st.set_page_config(page_title="Polar Smart Energy Management", layout="wide")


@st.cache_data
def load_data():
    return generate_year()


@st.cache_data
def get_forecast(_df):
    return forecast_load(_df)


@st.cache_data
def get_dispatch(start, hours):
    d = load_data().iloc[start:start + hours]
    disp, _ = optimize_dispatch(d.load_kw, d.solar_kw, d.wind_kw)
    disp.index = d.index
    return disp


def litres(kwh):
    return kwh / C.KWH_PER_LITRE_DIESEL


def co2(kwh):
    return litres(kwh) * C.CO2_KG_PER_LITRE_DIESEL


df = load_data()

st.title("❄️  AI-Driven Smart Energy Management — Polar Research Station")
st.caption("SIH 2026 · PS SIH26061 · load forecasting + fuel-minimising battery/diesel dispatch")

# ----------------------------------------------------------------------------
# Section 1 - the AI load forecast (always shown, on its held-out test window)
# ----------------------------------------------------------------------------
st.subheader("1 · AI load forecast (held-out 30 days the model never saw)")
res, mae = get_forecast(df)
mape = (np.abs(res.load_actual - res.load_pred) / res.load_actual).mean() * 100

fig_f = go.Figure()
fig_f.add_scatter(x=res.index, y=res.load_actual, name="Actual", line=dict(color="#1f77b4"))
fig_f.add_scatter(x=res.index, y=res.load_pred, name="Forecast",
                  line=dict(color="#ff7f0e", dash="dash"))
fig_f.update_layout(height=300, yaxis_title="Load (kW)",
                    margin=dict(t=10, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_f, use_container_width=True)
st.info(f"Forecast error:  MAE = {mae:.1f} kW   ·   MAPE = {mape:.1f} %")

st.divider()

# ----------------------------------------------------------------------------
# Section 2 - fuel-minimising dispatch, per scenario
# ----------------------------------------------------------------------------
st.subheader("2 · Optimised energy dispatch & fuel savings")

scenarios = {
    "☀️ Summer (midnight sun, Jan)": pd.Timestamp(f"{C.SIM_YEAR}-01-10"),
    "🌑 Polar night (dark, Jul)": pd.Timestamp(f"{C.SIM_YEAR}-07-10"),
    "⛅ Shoulder season (Apr)": pd.Timestamp(f"{C.SIM_YEAR}-04-10"),
}
c1, c2 = st.columns([2, 1])
scen = c1.selectbox("Scenario", list(scenarios))
days = c2.slider("Days shown", 2, 14, 7)

start = int(df.index.get_loc(scenarios[scen]))
disp = get_dispatch(start, days * 24)

# --- KPI tiles ---
opt_kwh = disp.diesel_kw.sum()
base_kwh = baseline_diesel(disp.load_kw, disp.solar_avail, disp.wind_avail).sum()
saved = base_kwh - opt_kwh
renew_frac = (disp.solar_used + disp.wind_used).sum() / disp.load_kw.sum() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Diesel used", f"{litres(opt_kwh):,.0f} L")
k2.metric("Fuel saved vs baseline", f"{litres(saved):,.0f} L",
          f"{(saved / base_kwh * 100 if base_kwh else 0):.0f}%")
k3.metric("Renewable share", f"{renew_frac:.0f}%")
k4.metric("CO₂ avoided", f"{co2(saved):,.0f} kg")

# --- Energy-mix stacked area ---
fig_m = go.Figure()
for name, colname, color in [
    ("Solar", "solar_used", "#f4d03f"),
    ("Wind", "wind_used", "#5dade2"),
    ("Battery", "discharge_kw", "#58d68d"),
    ("Diesel", "diesel_kw", "#e74c3c"),
]:
    fig_m.add_scatter(x=disp.index, y=disp[colname], name=name,
                      stackgroup="mix", line=dict(width=0.5, color=color))
fig_m.add_scatter(x=disp.index, y=disp.load_kw, name="Load",
                  line=dict(color="black", width=2))
fig_m.update_layout(title="Where every kilowatt comes from",
                    height=340, yaxis_title="kW",
                    margin=dict(t=40), legend=dict(orientation="h"))
st.plotly_chart(fig_m, use_container_width=True)

# --- Battery state of charge ---
fig_b = go.Figure()
fig_b.add_scatter(x=disp.index, y=disp.soc_kwh, name="Battery SoC",
                  fill="tozeroy", line=dict(color="#58d68d"))
fig_b.add_hline(y=C.BATTERY_MIN_SOC_FRAC * C.BATTERY_CAPACITY_KWH,
                line_dash="dot", annotation_text="min")
fig_b.add_hline(y=C.BATTERY_CAPACITY_KWH, line_dash="dot", annotation_text="full")
fig_b.update_layout(title="Battery state of charge", height=260,
                    yaxis_title="kWh", margin=dict(t=40))
st.plotly_chart(fig_b, use_container_width=True)

with st.expander("ℹ️  Assumptions & honesty note (read before demoing)"):
    st.markdown(
        "- No public energy data exists for polar stations, so inputs are "
        "**physically simulated** (seeded, reproducible), not real telemetry.\n"
        "- The solar model uses real solar-geometry for a 70°S latitude, so "
        "**polar night** (near-zero solar in July) and the **midnight sun** "
        "(solar even at night in January) emerge naturally.\n"
        "- The optimiser is a linear program (PuLP + CBC). Diesel capacity "
        "exceeds peak load, so a solution always exists.\n"
        "- 'Fuel saved' compares the optimiser against a no-battery baseline "
        "that uses renewables greedily and covers the rest with diesel."
    )
