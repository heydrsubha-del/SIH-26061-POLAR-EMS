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
from ems_core import calculate_metrics, build_operational_alerts, build_run_manifest, export_report
from health import runtime_diagnostics, configuration_diagnostics


st.set_page_config(page_title="POLAR EMS | Energy Operations", page_icon="❄️", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme ----------
st.session_state.setdefault("polar_theme_dark", False)
head_l, head_r = st.columns([6, 1.3])
with head_r:
    dark_mode = st.toggle("Dark mode", key="polar_theme_dark")

if dark_mode:
    BG, PANEL, PANEL_2, TEXT, MUTED = "#0A0C10", "#11151B", "#161B22", "#F2F4F7", "#8B96A3"
    BLUE, GREEN, AMBER, RED, SOLAR = "#4DA3FF", "#3DDC97", "#FFBD4A", "#FF5B65", "#F4C75A"
    BORDER, GRID, CHIP_BG, CHIP_FG = "#262D37", "#1E252E", "#12263B", "#8CC4FF"
else:
    BG, PANEL, PANEL_2, TEXT, MUTED = "#FFFFFF", "#FFFFFF", "#FAFAFA", "#171717", "#6B7280"
    BLUE, GREEN, AMBER, RED, SOLAR = "#0070F3", "#00A884", "#D98200", "#E5484D", "#E5A800"
    BORDER, GRID, CHIP_BG, CHIP_FG = "#EAEAEA", "#F1F1F1", "#F0F7FF", "#0060D0"
CYAN = BLUE

st.markdown(f"""<style>
:root {{ color-scheme:{'dark' if dark_mode else 'light'}; }}
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:{BG} !important; color:{TEXT} !important; }}
[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility:hidden; }}
[data-stale="true"] {{ opacity:1 !important; transition:none !important; }}
.block-container {{ padding-top:1.4rem; padding-bottom:2.5rem; max-width:1500px; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ color:{TEXT}; }}
[data-testid="stWidgetLabel"] p, label {{ color:{MUTED} !important; font-size:.8rem; }}
h1,h2,h3 {{ letter-spacing:-.03em; color:{TEXT}; font-weight:600; }}
input, textarea, [data-baseweb="input"] > div, [data-baseweb="select"] > div {{ background:{PANEL} !important; color:{TEXT} !important; border-color:{BORDER} !important; box-shadow:none !important; }}
[data-baseweb="select"] span {{ color:{TEXT} !important; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"] {{ background:{PANEL} !important; }}
[data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"] {{ background:{PANEL} !important; color:{TEXT} !important; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [role="option"]:hover {{ background:{PANEL_2} !important; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{BORDER} !important; border-radius:10px; background:{PANEL}; }}
[data-testid="stToggle"] [role="switch"][aria-checked="true"] {{ background:{BLUE} !important; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ background:{PANEL_2}; border-right:1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color:{TEXT}; }}
[data-testid="stTabs"] button {{ color:{MUTED}; font-size:.8rem; letter-spacing:.04em; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color:{TEXT}; }}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:10px; overflow:hidden; background:{PANEL}; }}
button, [data-testid="stDownloadButton"] button {{ border-radius:8px !important; }}
hr {{ border-color:{BORDER}; }}
.smallcaps, .eyebrow {{ color:{MUTED}; font-size:10px; letter-spacing:1.5px; font-weight:650; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.hdr-t {{ font-size:26px; font-weight:650; letter-spacing:-.9px; color:{TEXT}; margin-top:4px; }}
.hdr-t span {{ color:{MUTED}; font-weight:500; }}
.hdr-s {{ color:{MUTED}; font-size:13px; margin-top:4px; }}
.chips {{ display:flex; gap:6px; flex-wrap:wrap; }}
.chip {{ display:inline-block; padding:4px 9px; border-radius:6px; background:{CHIP_BG}; color:{CHIP_FG}; border:1px solid {BORDER}; font-size:10px; font-weight:650; letter-spacing:.6px; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:14px 0 18px; }}
.kpi, .flow-card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:15px 17px; }}
.kpi-l, .flow-card .label {{ font-size:11px; color:{MUTED}; letter-spacing:.8px; text-transform:uppercase; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.kpi-v, .flow-card .value {{ font-size:26px; font-weight:650; margin-top:8px; color:{TEXT}; letter-spacing:-.6px; line-height:1.1; }}
.flow-card .value {{ font-size:21px; }}
.kpi-f, .flow-card .foot {{ font-size:12px; color:{MUTED}; margin-top:6px; }}
.flow-wrap {{ display:flex; gap:8px; align-items:stretch; flex-wrap:wrap; padding:6px 0 4px; }}
.flow-wrap .flow-card {{ flex:1; min-width:150px; }}
.flow-arrow {{ align-self:center; color:{MUTED}; font-size:18px; }}
.section-head {{ display:flex; align-items:center; gap:9px; margin:8px 0 10px; }}
.section-dot {{ width:6px; height:6px; border-radius:50%; background:{BLUE}; }}
</style>""", unsafe_allow_html=True)

head_l.markdown(f"<div class='eyebrow'>ENERGY OPERATIONS CENTRE · {C.SIM_YEAR}</div><div class='hdr-t'>Polar EMS <span>Energy Operations</span></div><div class='hdr-s'>Research-station microgrid · forecast-led dispatch · renewable integration</div><div class='chips' style='margin-top:10px'><span class='chip'>● SIMULATION ONLINE</span></div>", unsafe_allow_html=True)


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
        height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT, size=12),
        title_font=dict(size=13, color=TEXT), title_x=0.01,
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BORDER, font_color=TEXT),
        margin=dict(l=18, r=18, t=44, b=24), hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)") if legend else dict(visible=False),
        xaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, showline=False),
        yaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, showline=False),
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

# Primary controls live on the main canvas, not inside a collapsible sidebar.
# This keeps them usable even when the sidebar is minimized by the user.
with st.sidebar:
    st.markdown("<div class='smallcaps'>POLAR EMS · CONTROL</div>", unsafe_allow_html=True)
    st.markdown("## Station operations")
    st.caption("SIH26061 · Polar research microgrid")
    st.markdown("---")
    st.markdown("**System configuration**")
    st.caption(f"PV · {C.PV_CAPACITY_KW:.0f} kW installed")
    st.caption(f"Wind · {C.WIND_CAPACITY_KW:.0f} kW installed")
    st.caption(f"Battery · {C.BATTERY_CAPACITY_KWH:.0f} kWh")
    st.caption(f"Diesel · {C.DIESEL_CAPACITY_KW:.0f} kW backup")
    st.markdown("---")
    st.caption("DEMO MODE · Deterministic simulation")
    st.caption("No live station telemetry")

with st.container(border=True):
    control_scenario, control_days, control_note = st.columns([2.2, 1.4, 1.6])
    scenario = control_scenario.selectbox("Operating scenario", list(scenarios), key="main_scenario")
    days = control_days.slider("Dispatch window", 2, 14, 7, 1, format="%d days", key="main_days")
    control_note.markdown(f"<div class='chips' style='padding-top:28px'><span class='chip'>{days} DAYS</span><span class='chip'>SIMULATED DATA</span></div>", unsafe_allow_html=True)

start = int(df.index.get_loc(scenarios[scenario]))
disp, solver_status = get_dispatch(start, days * 24)
res, mae = get_forecast(df)
mape = float((np.abs(res.load_actual - res.load_pred) / res.load_actual.clip(lower=1)).mean() * 100)

# Centralized, validated domain metrics. All panels and exports consume this object.
ops = calculate_metrics(disp, solver_status)
run_manifest = build_run_manifest(disp, scenario, solver_status)
operational_alerts = build_operational_alerts(ops)
load_energy = ops.load_kwh
diesel_kwh = ops.diesel_kwh
base_kwh = ops.baseline_diesel_kwh
saved_kwh = max(0.0, base_kwh - diesel_kwh)
renewable_kwh = ops.renewable_kwh
renew_share = ops.renewable_share_pct
avg_load = ops.average_load_kw
peak_load = ops.peak_load_kw
soc_now = ops.ending_soc_kwh
soc_pct = ops.ending_soc_pct

def kpi(label, value, foot):
    return f"<div class='kpi'><div class='kpi-l'>{label}</div><div class='kpi-v'>{value}</div><div class='kpi-f'>{foot}</div></div>"

st.markdown("<div class='kpi-grid'>" + "".join([
    kpi("Station demand", f"{avg_load:.1f} kW", f"Peak {peak_load:.1f} kW"),
    kpi("Diesel consumed", f"{litres(diesel_kwh):,.0f} L", f"{diesel_kwh:,.0f} kWh in window"),
    kpi("Renewable contribution", f"{renew_share:.1f}%", f"{renewable_kwh:,.0f} kWh supplied"),
    kpi("Fuel avoided", f"{litres(saved_kwh):,.0f} L", f"{(saved_kwh/base_kwh*100 if base_kwh else 0):.1f}% vs baseline"),
    kpi("Battery reserve", f"{soc_pct:.0f}%", f"{soc_now:.0f} / {C.BATTERY_CAPACITY_KWH:.0f} kWh"),
]) + "</div>", unsafe_allow_html=True)

alerts = operational_alerts
diesel_hours = ops.diesel_active_hours

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
        fig.add_trace(go.Scatter(x=disp.index, y=disp.load_kw, name="Demand", mode="lines", line=dict(color=TEXT, width=2.3), hovertemplate="Demand: %{y:.1f} kW<extra></extra>"))
        fig.update_layout(title="Power supply vs station demand", yaxis_title="Power (kW)")
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    with right:
        section("BATTERY RESERVE")
        fig_soc = go.Figure(go.Indicator(mode="gauge+number", value=soc_pct, number={"suffix":"%", "font":{"color":TEXT,"size":34}}, gauge={"axis":{"range":[0,100],"tickcolor":MUTED},"bar":{"color":GREEN},"bgcolor":PANEL,"bordercolor":BORDER,"steps":[{"range":[0,C.BATTERY_MIN_SOC_FRAC*100],"color":"rgba(229,72,77,.22)"},{"range":[C.BATTERY_MIN_SOC_FRAC*100,100],"color":PANEL_2}],"threshold":{"line":{"color":AMBER,"width":3},"thickness":0.8,"value":C.BATTERY_MIN_SOC_FRAC*100}}))
        fig_soc.update_layout(height=260, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT), margin=dict(t=20,b=15,l=10,r=10))
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
    with st.expander("Run provenance"):
        st.json(run_manifest)

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
    for alert in alerts:
        # Defensive display contract: malformed alert data must never crash the UI.
        alert = alert if isinstance(alert, dict) else {}
        severity = alert.get("severity", "notice")
        severity = severity.lower() if isinstance(severity, str) else "notice"
        code = str(alert.get("code", "GENERAL"))
        title = str(alert.get("title", "Operational notice"))
        detail = str(alert.get("detail", "No additional details provided."))
        color = {"success": GREEN, "info": BLUE, "warning": AMBER, "notice": RED}.get(severity, MUTED)
        st.markdown(f"<div class='flow-card' style='margin-bottom:8px;border-left:3px solid {color}'><div class='label' style='color:{color}'>{severity.upper()} · {code}</div><div style='font-weight:650;margin:5px 0'>{title}</div><div class='foot'>{detail}</div></div>", unsafe_allow_html=True)
    st.caption("Alerts are generated from the selected simulation window; they are not real station alarms.")
    report_md = export_report(ops, alerts, run_manifest)
    st.download_button("Download operations report (Markdown)", report_md, file_name="polar_ems_operations_report.md", mime="text/markdown")
    st.download_button("Download run manifest (JSON)", __import__("json").dumps(run_manifest, indent=2), file_name="polar_ems_run_manifest.json", mime="application/json")

with nav[4]:
    section("MODEL / SIMULATION TRANSPARENCY")
    st.markdown("### Runtime diagnostics")
    diag = pd.DataFrame(runtime_diagnostics())
    st.dataframe(diag, hide_index=True, use_container_width=True)
    cfg_diag = pd.DataFrame(configuration_diagnostics())
    st.markdown("### Configuration validation")
    st.dataframe(cfg_diag, hide_index=True, use_container_width=True)
    st.caption("PuLP is optional at runtime: when unavailable, the optimizer reports fallback status and the report records that fact.")
    st.markdown("### System assumptions")
    st.markdown("- **Telemetry:** generated from a reproducible hourly synthetic year (seeded), not a live SCADA feed.\n- **Location model:** 70°S solar geometry; Antarctic polar night and midnight sun emerge from the solar model.\n- **Forecasting:** gradient-boosted regressor with calendar, temperature, and lag features; final 30 days held out.\n- **Dispatch:** PuLP/CBC linear optimization when available; deterministic renewable-first fallback when solver/API fails.\n- **Fuel baseline:** renewables used greedily, no battery; remaining load supplied by diesel. Fuel saving is measured against this baseline.")
    st.markdown("### Installed capacity")
    config_table = pd.DataFrame({"Asset / parameter":["PV", "Wind", "Diesel backup", "Battery", "Minimum battery SOC", "Initial battery SOC", "Latitude", "Diesel energy factor", "Diesel CO₂ factor"], "Configured value":[f"{C.PV_CAPACITY_KW:.1f} kW", f"{C.WIND_CAPACITY_KW:.1f} kW", f"{C.DIESEL_CAPACITY_KW:.1f} kW", f"{C.BATTERY_CAPACITY_KWH:.1f} kWh", f"{C.BATTERY_MIN_SOC_FRAC*100:.0f}%", f"{C.BATTERY_INIT_SOC_FRAC*100:.0f}%", f"{C.LATITUDE_DEG:.1f}°", f"{C.KWH_PER_LITRE_DIESEL:.2f} kWh/L", f"{C.CO2_KG_PER_LITRE_DIESEL:.2f} kg/L"]})
    st.dataframe(config_table, hide_index=True, use_container_width=True)
    with st.expander("Project pipeline"):
        st.code("config.py → data_sim.py → forecast.py → optimizer.py → app.py", language="text")
        st.caption("Synthetic data generation · ML forecast · fuel-minimising dispatch · operations dashboard")

st.markdown(f"<div style='text-align:center;color:{MUTED};font-size:11px;padding-top:16px'>POLAR EMS · SIH 2026 · SIH26061 · SIMULATION ENVIRONMENT</div>", unsafe_allow_html=True)
