"""Polar EMS — professional operations dashboard for the SIH26061 demo.

All displayed values come from the project's seeded simulation and existing
forecast/dispatch modules. This is simulated operational telemetry, not live SCADA.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from data_sim import generate_year
from forecast import forecast_load
from optimizer import optimize_dispatch, baseline_diesel


st.set_page_config(page_title="POLAR EMS | Energy Operations", page_icon="❄️", layout="wide", initial_sidebar_state="expanded")

# ---------- Visual system ----------
BG = "#FFFFFF"
PANEL = "#FFFFFF"
PANEL_2 = "#FAFAFA"
TEXT = "#171717"
MUTED = "#737373"
CYAN = "#0070F3"
BLUE = "#0070F3"
GREEN = "#00A884"
AMBER = "#F5A623"
RED = "#EE0000"

st.markdown(f"""<style>
:root {{ color-scheme: light; }}
html, body, [data-testid="stAppViewContainer"] {{ background:{BG}; color:{TEXT}; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:#FAFAFA; border-right:1px solid #EBEBEB; }}
[data-testid="stSidebar"] * {{ color:{TEXT}; }}
.block-container {{ padding-top:1.35rem; padding-bottom:2.5rem; max-width:1500px; }}
h1,h2,h3 {{ letter-spacing:-.035em; color:{TEXT}; font-weight:600; }}
h1 {{ font-size:2.1rem !important; }}
[data-testid="stMetric"] {{ background:#FFFFFF; border:1px solid #EAEAEA; padding:16px 18px; border-radius:8px; box-shadow:0 1px 2px rgba(0,0,0,.025); }}
[data-testid="stMetricLabel"] {{ color:{MUTED} !important; font-size:.78rem !important; letter-spacing:.02em; }}
[data-testid="stMetricValue"] {{ color:{TEXT} !important; font-weight:650; }}
[data-testid="stMetricDelta"] {{ font-size:.78rem; }}
[data-testid="stTabs"] button {{ color:{MUTED}; font-size:.82rem; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color:{CYAN}; border-bottom-color:{CYAN}; }}
[data-testid="stPlotlyChart"] {{ border:1px solid #EBEBEB; border-radius:8px; overflow:hidden; background:{PANEL}; }}
[data-testid="stExpander"] {{ border:1px solid #EAEAEA; border-radius:8px; background:{PANEL}; }}
[data-testid="stSelectbox"] label,[data-testid="stSlider"] label {{ color:{MUTED}; }}
hr {{ border-color:#EBEBEB; }}
.smallcaps {{ color:{MUTED}; font-size:10px; letter-spacing:1.5px; font-weight:650; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.hero {{ padding:24px 26px; border:1px solid #EBEBEB; border-radius:10px; background:linear-gradient(115deg,#FAFAFA 0%,#FFFFFF 65%,#F4F8FF 100%); margin-bottom:18px; position:relative; overflow:hidden; }}
.hero:after {{ content:""; position:absolute; right:-110px; top:-120px; width:330px; height:260px; background:radial-gradient(ellipse at center,rgba(0,112,243,.10),rgba(80,227,194,.09),rgba(255,0,128,.035),transparent 72%); pointer-events:none; }}
.hero-title {{ font-size:25px; font-weight:650; letter-spacing:-.9px; color:#171717; }}
.hero-sub {{ color:{MUTED}; font-size:13px; margin-top:5px; }}
.status-pill {{ display:inline-block; padding:5px 10px; border-radius:6px; background:#F0F7FF; color:#0761D1; border:1px solid #D3E5FF; font-size:10px; font-weight:650; letter-spacing:.6px; }}
.flow-wrap {{ display:flex; gap:8px; align-items:stretch; justify-content:space-between; flex-wrap:wrap; padding:10px 0 4px; }}
.flow-card {{ flex:1; min-width:132px; background:{PANEL}; border:1px solid #EAEAEA; border-radius:8px; padding:14px 15px; }}
.flow-card .label {{ font-size:10px; color:{MUTED}; letter-spacing:1px; text-transform:uppercase; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.flow-card .value {{ font-size:21px; font-weight:650; margin-top:7px; color:{TEXT}; letter-spacing:-.5px; }}
.flow-card .foot {{ font-size:11px; color:{MUTED}; margin-top:4px; }}
.flow-arrow {{ align-self:center; color:{CYAN}; font-size:18px; font-weight:700; padding:0 1px; }}
.section-head {{ display:flex; align-items:center; gap:9px; margin:5px 0 11px; }}
.section-dot {{ width:6px; height:6px; border-radius:50%; background:{CYAN}; }}
button, [data-testid="stDownloadButton"] button {{ border-radius:6px !important; }}
[data-testid="stDataFrame"] {{ border:1px solid #EBEBEB; border-radius:8px; }}
</style>""", unsafe_allow_html=True);


@st.cache_data(show_spinner="Generating the polar station simulation…")
def load_data():
    return generate_year()


@st.cache_data(show_spinner="Training load forecast…")
def get_forecast(_df):
    return forecast_load(_df)

@st.cache_data(show_spinner="Optimising energy dispatch…")
def get_dispatch(start, hours):
    d = load_data().iloc[start:start + hours]
    dispatch, status = optimize_dispatch(d.load_kw, d.solar_kw, d.wind_kw)
    dispatch.index = d.index
    return dispatch, status

def litres(kwh):
    return kwh / C.KWH_PER_LITRE_DIESEL

def co2(kwh):
    return litres(kwh) * C.CO2_KG_PER_LITRE_DIESEL

def chart_layout(fig, height=330, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT, size=11),
        margin=dict(l=18, r=18, t=36, b=24), hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)") if legend else dict(visible=False),
        xaxis=dict(gridcolor="#F0F0F0", zerolinecolor="#E5E5E5", showline=False),
        yaxis=dict(gridcolor="#F0F0F0", zerolinecolor="#E5E5E5", showline=False),
    )
    return fig

def section(label):
    st.markdown(f'<div class="section-head"><span class="section-dot"></span><span class="smallcaps">{label}</span></div>', unsafe_allow_html=True)


# ---------- Dataset and forecast ----------
df = load_data()

scenarios = {
    "SUMMER / MIDNIGHT SUN": pd.Timestamp(f"{C.SIM_YEAR}-01-10"),
    "POLAR NIGHT / JULY": pd.Timestamp(f"{C.SIM_YEAR}-07-10"),
    "SHOULDER SEASON / APRIL": pd.Timestamp(f"{C.SIM_YEAR}-04-10"),
}

# Sidebar controls are tied to the real simulation window.
with st.sidebar:
    st.markdown("<div class='smallcaps'>POLAR EMS · CONTROL</div>", unsafe_allow_html=True)
    st.markdown("## Station operations")
    st.caption("SIH26061 · Polar research microgrid")
    scenario = st.selectbox("Operating scenario", list(scenarios), index=0)
    days = st.slider("Dispatch window", min_value=2, max_value=14, value=7, step=1, format="%d days")
    st.markdown("---")
    st.markdown("**System configuration**")
    st.caption(f"PV · {C.PV_CAPACITY_KW:.0f} kW installed")
    st.caption(f"Wind · {C.WIND_CAPACITY_KW:.0f} kW installed")
    st.caption(f"Battery · {C.BATTERY_CAPACITY_KWH:.0f} kWh")
    st.caption(f"Diesel · {C.DIESEL_CAPACITY_KW:.0f} kW backup")
    st.markdown("---")
    st.caption("DEMO MODE · Deterministic simulation")
    st.caption("No live station telemetry")

start = int(df.index.get_loc(scenarios[scenario]))
disp, solver_status = get_dispatch(start, days * 24)
res, mae = get_forecast(df)
mape = float((np.abs(res.load_actual - res.load_pred) / res.load_actual.clip(lower=1)).mean() * 100)

# Dispatch metrics calculated from selected window.
load_energy = float(disp.load_kw.sum())
diesel_kwh = float(disp.diesel_kw.sum())
base_kwh = float(baseline_diesel(disp.load_kw, disp.solar_avail, disp.wind_avail).sum())
saved_kwh = max(0.0, base_kwh - diesel_kwh)
renewable_kwh = float((disp.solar_used + disp.wind_used).sum())
renew_share = renewable_kwh / load_energy * 100 if load_energy else 0.0
avg_load = float(disp.load_kw.mean())
peak_load = float(disp.load_kw.max())
soc_now = float(disp.soc_kwh.iloc[-1]) if len(disp) else 0
soc_pct = soc_now / C.BATTERY_CAPACITY_KWH * 100 if C.BATTERY_CAPACITY_KWH else 0

# ---------- Sidebar-independent header ----------
st.markdown(f"""<div class="hero"><div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
<div><div class="smallcaps">ENERGY OPERATIONS CENTRE · {C.SIM_YEAR}</div><div class="hero-title">Polar EMS · Energy Operations</div><div class="hero-sub">Research-station microgrid · forecast-led dispatch · renewable integration</div></div>
<div><span class="status-pill">● SIMULATION ONLINE</span></div></div>""", unsafe_allow_html=True)

# Top metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Station demand", f"{avg_load:.1f} kW", f"Peak {peak_load:.1f} kW")
m2.metric("Diesel consumed", f"{litres(diesel_kwh):,.0f} L", f"{diesel_kwh:,.0f} kWh in window")
m3.metric("Renewable contribution", f"{renew_share:.1f}%", f"{renewable_kwh:,.0f} kWh supplied")
m4.metric("Fuel avoided", f"{litres(saved_kwh):,.0f} L", f"{(saved_kwh/base_kwh*100 if base_kwh else 0):.1f}% vs baseline")
m5.metric("Battery reserve", f"{soc_pct:.0f}%", f"{soc_now:.0f} / {C.BATTERY_CAPACITY_KWH:.0f} kWh")

# Contextual operating condition, from actual selected scenario.
enewable_available_mean = float((disp.solar_avail + disp.wind_avail).mean())
diesel_hours = int((disp.diesel_kw > 0.1).sum())
low_soc = bool((disp.soc_kwh <= C.BATTERY_MIN_SOC_FRAC * C.BATTERY_CAPACITY_KWH + 1).any())
alerts = []
if renew_share < 30:
    alerts.append(("WATCH", "Low renewable contribution", f"Renewables supplied {renew_share:.1f}% of selected-window demand."))
if low_soc:
    alerts.append(("WATCH", "Battery reserve touched", "The dispatch reached the configured minimum state-of-charge threshold."))
if diesel_hours:
    alerts.append(("INFO", "Backup generation used", f"Diesel was active for {diesel_hours} of {len(disp)} simulated hours."))
if not alerts:
    alerts.append(("OK", "No operating alerts", "Selected dispatch window remained within configured battery reserve."))

# Main app navigation
nav = st.tabs(["OVERVIEW", "FORECAST", "DISPATCH", "ASSETS & ALERTS", "MODEL NOTES"])
with nav[0]:
    section("POWER FLOW · SELECTED SCENARIO")
    # Data-driven flow summary: actual energy in selected dispatch window.
    solar = float(disp.solar_used.sum())
    wind = float(disp.wind_used.sum())
    battery = float(disp.discharge_kw.sum())
    diesel = float(disp.diesel_kw.sum())
    flow_items = [
        ("SOLAR", f"{solar:,.0f} kWh", f"Peak available {disp.solar_avail.max():.1f} kW", "#F4C75A"),
        ("WIND", f"{wind:,.0f} kWh", f"Peak available {disp.wind_avail.max():.1f} kW", BLUE),
        ("BATTERY", f"{battery:,.0f} kWh", f"Reserve at end {soc_pct:.0f}%", GREEN),
        ("DIESEL", f"{diesel:,.0f} kWh", f"Generator active {diesel_hours} h", AMBER),
        ("STATION LOAD", f"{load_energy:,.0f} kWh", f"Average {avg_load:.1f} kW", CYAN),
    ]
    html = '<div class="flow-wrap">'
    for i, (name, value, foot, color) in enumerate(flow_items):
        html += f'<div class="flow-card"><div class="label">{name}</div><div class="value" style="color:{color}">{value}</div><div class="foot">{foot}</div></div>'
        if i < len(flow_items)-1:
            html += '<div class="flow-arrow">›</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    left, right = st.columns([1.65, 1])
    with left:
        section("ENERGY MIX · HOURLY")
        fig = go.Figure()
        mix_colors = [("Solar", "solar_used", "#F4C75A"), ("Wind", "wind_used", BLUE), ("Battery", "discharge_kw", GREEN), ("Diesel", "diesel_kw", AMBER)]
        for name, key, color in mix_colors:
            fig.add_trace(go.Scatter(x=disp.index, y=disp[key], name=name, stackgroup="supply", mode="lines", line=dict(width=0.8, color=color), hovertemplate=f"{name}: %{{y:.1f}} kW<extra></extra>"))
        fig.add_trace(go.Scatter(x=disp.index, y=disp.load_kw, name="Demand", mode="lines", line=dict(color="#171717", width=2.3), hovertemplate="Demand: %{y:.1f} kW<extra></extra>"))
        fig.update_layout(title="Power supply vs station demand", yaxis_title="Power (kW)")
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    with right:
        section("BATTERY RESERVE")
        fig_soc = go.Figure(go.Indicator(mode="gauge+number", value=soc_pct, number={"suffix":"%", "font":{"color":TEXT,"size":34}}, gauge={"axis":{"range":[0,100],"tickcolor":MUTED},"bar":{"color":GREEN},"bgcolor":"#FFFFFF","bordercolor":"#E5E5E5","steps":[{"range":[0,C.BATTERY_MIN_SOC_FRAC*100],"color":"#F7D4D6"},{"range":[C.BATTERY_MIN_SOC_FRAC*100,100],"color":"#F3F8FF"}],"threshold":{"line":{"color":AMBER,"width":3},"thickness":0.8,"value":C.BATTERY_MIN_SOC_FRAC*100}}))
        fig_soc.update_layout(height=260, paper_bgcolor=PANEL, font=dict(color=TEXT), margin=dict(t=20,b=15,l=10,r=10))
        st.plotly_chart(fig_soc, use_container_width=True)
        st.caption(f"Minimum reserve: {C.BATTERY_MIN_SOC_FRAC*100:.0f}% · Capacity: {C.BATTERY_CAPACITY_KWH:.0f} kWh")
        section("DISPATCH ENGINE")
        solver_color = GREEN if solver_status == "Optimal" else AMBER
        st.markdown(f"<div class='flow-card'><div class='label'>SOLVER STATUS</div><div class='value' style='color:{solver_color};font-size:18px'>{solver_status}</div><div class='foot'>Selected {days}-day window · {len(disp)} hourly steps</div></div>", unsafe_allow_html=True)

    section("WINDOW SUMMARY")
    daily = disp.resample("D").sum(numeric_only=True)
    fig_daily = go.Figure()
    for name, key, color in mix_colors:
        fig_daily.add_bar(x=daily.index, y=daily[key], name=name, marker_color=color)
    fig_daily.update_layout(barmode="stack", title="Daily energy contribution", yaxis_title="Energy (kWh)")
    st.plotly_chart(chart_layout(fig_daily, 290), use_container_width=True)

    st.download_button("Download dispatch data (CSV)", disp.to_csv().encode("utf-8"), file_name="polar_ems_dispatch.csv", mime="text/csv")

with nav[1]:
    section("LOAD FORECAST · HELD-OUT TEST PERIOD")
    st.markdown("Model trained on earlier simulated data and evaluated on the final 30-day window.")
    a,b,c = st.columns(3)
    a.metric("Mean absolute error", f"{mae:.2f} kW")
    b.metric("MAPE", f"{mape:.2f}%")
    c.metric("Test window", f"{len(res):,} hours")
    figf = go.Figure()
    figf.add_trace(go.Scatter(x=res.index, y=res.load_actual, name="Actual demand", line=dict(color=CYAN,width=1.6), fill="tozeroy", fillcolor="rgba(57,214,232,.08)"))
    figf.add_trace(go.Scatter(x=res.index, y=res.load_pred, name="Predicted demand", line=dict(color=AMBER,width=1.7,dash="dot")))
    figf.update_layout(title="Actual demand and model prediction", yaxis_title="Load (kW)")
    st.plotly_chart(chart_layout(figf, 410), use_container_width=True)
    errors = res.load_actual-res.load_pred
    n_bins = 30
    figerr = go.Figure(go.Histogram(x=errors, nbinsx=n_bins, marker_color=BLUE, name="Forecast error"))
    figerr.update_layout(title="Forecast error distribution", xaxis_title="Actual − predicted (kW)", yaxis_title="Frequency")
    st.plotly_chart(chart_layout(figerr, 260, legend=False), use_container_width=True)
    with st.expander("Forecast methodology"):
        st.markdown("Gradient-boosted regression uses cyclical calendar features, temperature, 24-hour and 168-hour load lags, and a rolling 24-hour mean. The final 30 days are held out from training.")

with nav[2]:
    section("OPTIMISED ENERGY DISPATCH")
    figd = go.Figure()
    figd.add_trace(go.Scatter(x=disp.index, y=disp.load_kw, name="Station demand", line=dict(color=TEXT,width=2.2)))
    figd.add_trace(go.Scatter(x=disp.index, y=disp.solar_avail, name="Solar available", line=dict(color="#F4C75A",width=1.4,dash="dot")))
    figd.add_trace(go.Scatter(x=disp.index, y=disp.wind_avail, name="Wind available", line=dict(color=BLUE,width=1.4,dash="dot")))
    figd.update_layout(title="Demand and renewable availability", yaxis_title="Power (kW)")
    st.plotly_chart(chart_layout(figd, 340), use_container_width=True)
    figbat = go.Figure()
    figbat.add_trace(go.Scatter(x=disp.index, y=disp.soc_kwh, name="Battery energy", line=dict(color=GREEN,width=2), fill="tozeroy", fillcolor="rgba(70,214,162,.12)"))
    figbat.add_hline(y=C.BATTERY_MIN_SOC_FRAC*C.BATTERY_CAPACITY_KWH, line_dash="dash", line_color=AMBER, annotation_text="Reserve floor")
    figbat.add_hline(y=C.BATTERY_CAPACITY_KWH, line_dash="dot", line_color=MUTED, annotation_text="Capacity")
    figbat.update_layout(title="Battery state-of-charge trajectory", yaxis_title="Stored energy (kWh)")
    st.plotly_chart(chart_layout(figbat, 320, legend=False), use_container_width=True)
    detail = disp.copy()
    detail.index.name = "timestamp"
    st.dataframe(detail.round(2), use_container_width=True, height=300)
    st.download_button("Export detailed dispatch", detail.to_csv().encode("utf-8"), "polar_dispatch_detail.csv", "text/csv")

with nav[3]:
    section("ASSET STATUS · CONFIGURED SYSTEM")
    assets = [
        ("Solar PV array", f"{C.PV_CAPACITY_KW:.0f} kW", f"Peak available {disp.solar_avail.max():.1f} kW", "GENERATION", "#F4C75A"),
        ("Wind turbine system", f"{C.WIND_CAPACITY_KW:.0f} kW", f"Peak available {disp.wind_avail.max():.1f} kW", "GENERATION", BLUE),
        ("Battery storage", f"{C.BATTERY_CAPACITY_KWH:.0f} kWh", f"Current simulation endpoint {soc_pct:.0f}%", "STORAGE", GREEN),
        ("Diesel genset", f"{C.DIESEL_CAPACITY_KW:.0f} kW", f"Active in {diesel_hours} simulated hours", "BACKUP", AMBER),
    ]
    cols = st.columns(2)
    for idx, (name, capacity, detail_text, category, color) in enumerate(assets):
        with cols[idx % 2]:
            st.markdown(f"<div class='flow-card' style='margin-bottom:12px'><div class='label'>{category}</div><div class='value' style='color:{color}'>{capacity}</div><div style='font-weight:650;margin-top:6px'>{name}</div><div class='foot'>{detail_text}</div></div>", unsafe_allow_html=True)
    section("OPERATIONS / ALERT LOG")
    for level, title, description in alerts:
        color = GREEN if level == "OK" else AMBER if level == "WATCH" else BLUE
        st.markdown(f"<div class='flow-card' style='margin-bottom:8px;border-left:3px solid {color}'><div class='label' style='color:{color}'>{level} · SIMULATION EVENT</div><div style='font-weight:650;margin:5px 0'>{title}</div><div class='foot'>{description}</div></div>", unsafe_allow_html=True)
    st.caption("Alerts are generated from the selected simulation window; they are not real station alarms.")

with nav[4]:
    section("MODEL / SIMULATION TRANSPARENCY")
    st.markdown("### System assumptions")
    st.markdown("- **Telemetry:** generated from a reproducible hourly synthetic year (seeded), not a live SCADA feed.\n- **Location model:** 70°S solar geometry; Antarctic polar night and midnight sun emerge from the solar model.\n- **Forecasting:** gradient-boosted regressor with calendar, temperature, and lag features; final 30 days held out.\n- **Dispatch:** PuLP/CBC linear optimization when available; deterministic renewable-first fallback when solver/API fails.\n- **Fuel baseline:** renewables used greedily, no battery; remaining load supplied by diesel. Fuel saving is measured against this baseline.")
    st.markdown("### Installed capacity")
    config_table = pd.DataFrame({"Asset / parameter":["PV", "Wind", "Diesel backup", "Battery", "Minimum battery SOC", "Initial battery SOC", "Latitude", "Diesel energy factor", "Diesel CO₂ factor"], "Configured value":[f"{C.PV_CAPACITY_KW:.1f} kW", f"{C.WIND_CAPACITY_KW:.1f} kW", f"{C.DIESEL_CAPACITY_KW:.1f} kW", f"{C.BATTERY_CAPACITY_KWH:.1f} kWh", f"{C.BATTERY_MIN_SOC_FRAC*100:.0f}%", f"{C.BATTERY_INIT_SOC_FRAC*100:.0f}%", f"{C.LATITUDE_DEG:.1f}°", f"{C.KWH_PER_LITRE_DIESEL:.2f} kWh/L", f"{C.CO2_KG_PER_LITRE_DIESEL:.2f} kg/L"]})
    st.dataframe(config_table, hide_index=True, use_container_width=True)
    with st.expander("Project pipeline"):
        st.code("config.py → data_sim.py → forecast.py → optimizer.py → app.py", language="text")
        st.caption("Synthetic data generation · ML forecast · fuel-minimising dispatch · operations dashboard")

st.markdown("<div style='text-align:center;color:#60788B;font-size:11px;padding-top:16px'>POLAR EMS · SIH 2026 · SIH26061 · SIMULATION ENVIRONMENT</div>", unsafe_allow_html=True)
