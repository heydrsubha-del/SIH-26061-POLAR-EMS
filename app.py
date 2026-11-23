"""Polar EMS - energy operations console (SIH26061).

All values come from the seeded simulation plus the project's forecast,
dispatch and smart_ops modules. Simulated telemetry, not live SCADA.
"""
import html
import json
import urllib.parse as _up

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from data_sim import generate_year
from forecast import forecast_load
from optimizer import optimize_dispatch
from ems_core import calculate_metrics, build_operational_alerts, build_run_manifest, export_report
from health import runtime_diagnostics, configuration_diagnostics
from smart_ops import (battery_health, maintenance_signals, run_stress_scenario,
                       forecast_aware_dispatch, load_history, save_history_run, build_run_summary)

st.set_page_config(page_title="Polar EMS | Energy Operations", page_icon="❄️",
                   layout="wide", initial_sidebar_state="expanded")

# ====================== state + theme ======================
st.session_state.setdefault("polar_dark", True)
NAV = ["Overview", "Forecast", "Dispatch", "Scenarios", "Assets & Alerts", "Model Notes"]
st.session_state.setdefault("active_panel", NAV[0])

with st.container(key="theme_row"):
    _l, _r = st.columns([9, 1.6])
    with _r:
        DARK = st.toggle("Dark mode", key="polar_dark")

if DARK:
    TOK = dict(bg="#0a0d12", panel="#10151c", panel2="#141b24", panel3="#0c1117", line="#1c2531", line2="#2a3544",
               text="#e9eef5", muted="#8b97a8", on="#0a0d12", glassa="#131923", glassb="#0d1219",
               sidea="#0c1322", sideb="#070b12", dot="#3a4658", ok="#34d399", warn="#f5b14a", bad="#fb7185",
               shadow="0 14px 34px -18px rgba(0,0,0,.85)", shadows="0 8px 20px -14px rgba(0,0,0,.8)")
    TONES = {"Overview": "#2fb68e", "Forecast": "#a78bfa", "Dispatch": "#38bdf8",
             "Scenarios": "#f59e0b", "Assets & Alerts": "#fb7185", "Model Notes": "#94a3b8"}
    SOLAR, WINDC, BATT, DIESEL = "#f4c75a", "#4da3ff", "#3ddc97", "#ffb04a"
else:
    TOK = dict(bg="#f3f5f8", panel="#ffffff", panel2="#f8fafc", panel3="#f1f4f8", line="#e4e8ee", line2="#d3d9e2",
               text="#121722", muted="#5d6877", on="#ffffff", glassa="#ffffff", glassb="#f1f4f8",
               sidea="#ffffff", sideb="#f4f6f9", dot="#b9c2cf", ok="#0f9a73", warn="#b97800", bad="#d6334d",
               shadow="0 14px 30px -20px rgba(20,30,50,.28)", shadows="0 6px 16px -12px rgba(20,30,50,.25)")
    TONES = {"Overview": "#0f9a73", "Forecast": "#7c5ce6", "Dispatch": "#0b84d6",
             "Scenarios": "#c47f00", "Assets & Alerts": "#d6334d", "Model Notes": "#5d6877"}
    SOLAR, WINDC, BATT, DIESEL = "#d99a00", "#0070f3", "#0f9a73", "#d9741a"

PANEL = st.session_state["active_panel"]
TONE = TONES.get(PANEL, "#2fb68e")

_VARS = {
    "--bg": TOK["bg"], "--panel": TOK["panel"], "--panel-2": TOK["panel2"], "--panel-3": TOK["panel3"],
    "--line": TOK["line"], "--line-strong": TOK["line2"], "--text": TOK["text"], "--muted": TOK["muted"],
    "--on-accent": TOK["on"], "--glass-a": TOK["glassa"], "--glass-b": TOK["glassb"],
    "--side-a": TOK["sidea"], "--side-b": TOK["sideb"], "--dot": TOK["dot"], "--ok": TOK["ok"],
    "--warn": TOK["warn"], "--bad": TOK["bad"], "--shadow": TOK["shadow"], "--shadow-sm": TOK["shadows"],
    "--tone": TONE, "--tone-soft": f"color-mix(in srgb,{TONE} 14%,transparent)",
}
_root = "html body .stApp{" + ";".join(f"{k}:{v}" for k, v in _VARS.items()) + ";color-scheme:" + ("dark" if DARK else "light") + "}"
_root += "body [data-baseweb=\"popover\"]{" + ";".join(f"{k}:{v}" for k, v in _VARS.items()) + "}"

STATIC_CSS = r'''
html body .stApp{background:var(--bg) !important;color:var(--text) !important;font-family:Inter,"Segoe UI",system-ui,sans-serif}
html body .stApp [data-testid="stAppViewContainer"],html body .stApp [data-testid="stMain"]{background:transparent !important}
html body .stApp [data-testid="stHeader"]{background:transparent !important}
html,body{overflow-x:hidden !important;max-width:100vw !important}
#MainMenu,footer,[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stAppDeployButton"]{visibility:hidden !important}
[data-testid="stSidebarCollapsedControl"],[data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapseButton"]{visibility:visible !important;display:flex !important}
[data-stale="true"]{opacity:1 !important;transition:none !important}
.stApp .block-container{padding:2rem 1.35rem 3rem !important;max-width:1500px}
.stApp :is(p,li,h1,h2,h3,h4,h5,h6){color:var(--text)}
.stApp [data-testid="stWidgetLabel"] p,.stApp label p{color:var(--muted) !important;font-size:12px !important;font-weight:600 !important;letter-spacing:.02em}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] p{color:var(--muted) !important}
.stApp hr{border:0 !important;height:1px;background:linear-gradient(90deg,var(--tone),var(--line) 40%,transparent) !important;opacity:.7}
/* page-level flow: collapse style-only blocks */
.stApp .stElementContainer:has(> [data-testid="stMarkdown"] > [data-testid="stMarkdownContainer"] > style:only-child),
.stApp [data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] > style:only-child){display:none !important}
/* inputs */
.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="input"]>div,.stApp input,.stApp textarea{background:var(--panel-3) !important;color:var(--text) !important;border-color:var(--line-strong) !important;border-radius:10px !important;box-shadow:none !important}
.stApp [data-baseweb="select"] span,.stApp [data-baseweb="select"] svg{color:var(--text) !important;fill:var(--text) !important}
.stApp [data-baseweb="select"]:focus-within>div,.stApp input:focus{border-color:var(--tone) !important;box-shadow:0 0 0 3px var(--tone-soft) !important}
body [data-baseweb="popover"] ul,body [data-baseweb="popover"] [role="listbox"]{background:var(--panel) !important;border:1px solid var(--line-strong) !important;border-radius:12px !important}
body [data-baseweb="popover"] li,body [data-baseweb="popover"] [role="option"]{background:var(--panel) !important;color:var(--text) !important}
body [data-baseweb="popover"] li:hover,body [data-baseweb="popover"] [role="option"]:hover,body [data-baseweb="popover"] [aria-selected="true"]{background:var(--tone-soft) !important}
.stApp [data-testid="stSlider"] [data-testid="stThumbValue"],.stApp [data-testid="stSlider"] [data-testid="stTickBarMin"],.stApp [data-testid="stSlider"] [data-testid="stTickBarMax"]{color:var(--muted) !important}
.stApp [data-testid="stSlider"] [role="slider"]{background:var(--tone) !important;border-color:var(--tone) !important;box-shadow:0 0 0 4px var(--tone-soft) !important}
/* toggle */
.stApp [data-testid="stToggle"] label p{color:var(--text) !important;font-weight:600 !important}
.stApp [data-testid="stToggle"] [role="switch"]{background:var(--line-strong) !important;border:1px solid var(--line-strong) !important}
.stApp [data-testid="stToggle"] [role="switch"][aria-checked="true"]{background:var(--tone) !important;border-color:var(--tone) !important}
/* buttons */
.stApp :is(.stButton,.stDownloadButton) button{min-height:38px;border-radius:10px !important;font:650 13px/1 Inter,sans-serif !important;color:var(--text) !important;background:var(--panel-2) !important;border:1px solid var(--line-strong) !important;box-shadow:none !important;transition:border-color .18s,background .18s,transform .18s !important}
.stApp :is(.stButton,.stDownloadButton) button:hover{border-color:var(--tone) !important;background:var(--tone-soft) !important;transform:translateY(-1px)}
.stApp .stButton button[kind="primary"]{background:linear-gradient(180deg,var(--tone),color-mix(in srgb,var(--tone) 70%,#000)) !important;color:var(--on-accent) !important;border-color:transparent !important}
.stApp .stButton button[kind="primary"] p{color:var(--on-accent) !important}
/* expander / plotly / code */
.stApp [data-testid="stExpander"]{background:var(--panel) !important;border:1px solid var(--line-strong) !important;border-left:3px solid var(--tone) !important;border-radius:12px !important}
.stApp [data-testid="stExpander"] summary,.stApp [data-testid="stExpander"] summary p{color:var(--text) !important;font-weight:650}
.stApp [data-testid="stPlotlyChart"]{border:1px solid var(--line-strong);border-radius:14px;overflow:hidden;background:var(--panel);box-shadow:var(--shadow)}
.stApp .st-key-ctl_card{border:1px solid var(--line-strong) !important;border-radius:14px !important;background:linear-gradient(180deg,var(--panel-2),var(--panel)) !important;padding:14px 18px 10px !important;box-shadow:var(--shadow)}
/* theme toggle row */
.stApp .st-key-theme_row{display:flex;justify-content:flex-end}
.stApp .st-key-theme_row [data-testid="stToggle"]{padding:6px 12px;border:1px solid var(--line-strong);border-radius:999px;background:var(--panel)}
/* ---------- top banner ---------- */
.hdr{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:4px 20px;padding:2px 0 10px 14px;margin:0 0 14px;border-left:3px solid var(--tone);border-bottom:1px solid var(--line)}
.hdr h1{margin:0;padding:0;font-size:20px;font-weight:700;letter-spacing:-.01em;color:var(--text)}
.hdr-meta{font:500 12px/1.4 ui-monospace,monospace;color:var(--muted)}
.hdr-meta b{color:var(--ok);font-weight:600}
@media(max-width:640px){.hdr h1{font-size:16px}}
/* ---------- capsule nav ---------- */
.stApp .st-key-topnav{position:relative;margin:38px 0 14px !important;padding:0 !important;border-radius:32px;border:1px solid var(--line-strong);background:linear-gradient(180deg,var(--glass-a),var(--glass-b));box-shadow:var(--shadow)}
.stApp .st-key-topnav [data-testid="stWidgetLabel"]{display:none !important}
.stApp .st-key-topnav [role="radiogroup"]{display:flex !important;flex-wrap:nowrap !important;align-items:flex-end;gap:4px !important;margin-top:-28px !important;padding:34px 28px 8px !important;overflow-x:auto;overflow-y:hidden;scrollbar-width:none;background:none !important;border:0 !important}
.stApp .st-key-topnav [role="radiogroup"]::-webkit-scrollbar{display:none}
.stApp .st-key-topnav [role="radiogroup"] label{position:relative;flex:0 0 auto;display:flex !important;flex-direction:column;align-items:center;justify-content:flex-end;min-width:96px;padding:12px 14px 11px !important;border:0 !important;border-radius:18px;background:transparent !important;cursor:pointer}
.stApp .st-key-topnav [role="radiogroup"] label>*:not(:has(p)),.stApp .st-key-topnav [role="radiogroup"] label input{display:none !important}
.stApp .st-key-topnav [role="radiogroup"] label>div:first-child:not(:has(p)),.stApp .st-key-topnav [role="radiogroup"] label>div:empty,.stApp .st-key-topnav [role="radiogroup"] label [data-baseweb="radio"]>div:first-child:not(:has(p)){display:none !important;width:0 !important;height:0 !important;margin:0 !important;opacity:0 !important}
.stApp .st-key-topnav [role="radiogroup"] label *:not(p):not(p *):not(:has(p)){display:none !important;visibility:hidden !important;width:0 !important;height:0 !important;border:0 !important;background:none !important;box-shadow:none !important}
.stApp .st-key-topnav [role="radiogroup"] label::after{content:none !important}
.stApp .st-key-topnav [role="radiogroup"] label p{position:relative;z-index:2;display:flex;flex-direction:column;align-items:center;gap:7px;margin:0 !important;white-space:nowrap;font:600 11.5px/1.15 Inter,sans-serif !important;letter-spacing:.025em;color:var(--muted) !important;transition:color .2s}
.stApp .st-key-topnav [role="radiogroup"] label p::before{content:"";width:22px;height:22px;background-color:var(--c,#2fb68e);opacity:.85;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;-webkit-mask-size:contain;mask-size:contain;transition:transform .32s cubic-bezier(.34,1.56,.64,1),opacity .2s}
.stApp .st-key-topnav [role="radiogroup"] label:hover p{color:var(--text) !important}
.stApp .st-key-topnav [role="radiogroup"] label:hover p::before{opacity:1;transform:translateY(-3px) scale(1.12)}
.stApp .st-key-topnav [role="radiogroup"] label:has(input:checked)::before{content:"";position:absolute;z-index:1;left:50%;top:-20px;width:52px;height:52px;margin-left:-26px;border-radius:50%;background:linear-gradient(150deg,var(--c),color-mix(in srgb,var(--c) 62%,#000));border:6px solid var(--bg);box-shadow:0 0 0 1px color-mix(in srgb,var(--c) 60%,transparent),0 12px 22px -6px var(--c);animation:navLift .42s cubic-bezier(.34,1.56,.64,1)}
.stApp .st-key-topnav [role="radiogroup"] label:has(input:checked) p{color:var(--c) !important;font-weight:750 !important}
.stApp .st-key-topnav [role="radiogroup"] label:has(input:checked) p::before{background-color:var(--on-accent);opacity:1;transform:translateY(-17px) scale(1.04)}
@keyframes navLift{from{transform:translateY(14px) scale(.7);opacity:0}to{transform:none;opacity:1}}
/* ---------- sidebar ---------- */
.stApp [data-testid="stSidebar"]{background:linear-gradient(180deg,var(--side-a),var(--side-b)) !important;border-right:1px solid var(--line) !important}
.stApp [data-testid="stSidebar"]>div{background:transparent !important}
.stApp [data-testid="stSidebar"] [data-testid="stSidebarHeader"]{height:26px !important;min-height:0 !important}
.stApp [data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:0 14px !important}
.stApp [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:clamp(2px,.5vh,6px) !important}
.stApp [data-testid="stSidebar"] [data-testid="stMarkdown"]{margin:0 !important}
.sidebar-brand-v2{display:flex;align-items:center;gap:12px;margin:2px 0 8px;padding:2px 4px}
.brand-mark{width:40px;height:40px;flex:0 0 40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:19px;background:linear-gradient(145deg,color-mix(in srgb,#4c8dff 22%,var(--panel)),var(--panel-3));border:1px solid color-mix(in srgb,#4c8dff 50%,transparent);box-shadow:0 10px 20px -10px #4c8dff}
.brand-name{font:750 15px/1.1 Inter,sans-serif;letter-spacing:.06em;color:var(--text)}
.brand-role{font:500 11.5px/1.3 Inter,sans-serif;color:var(--muted);margin-top:3px}
.sidebar-group-label{display:flex;align-items:center;gap:10px;margin:0 4px 0 14px;padding:clamp(10px,2.2vh,20px) 0 clamp(4px,.8vh,8px);font:650 10px/1 Inter,sans-serif;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);--g:#4c8dff}
.sidebar-group-label::before{content:"";flex:none;width:6px;height:6px;border-radius:50%;background:var(--g);box-shadow:0 0 0 3px color-mix(in srgb,var(--g) 18%,transparent)}
.sidebar-group-label::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,color-mix(in srgb,var(--g) 45%,transparent),transparent)}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button{position:relative;min-height:clamp(32px,4.4vh,42px) !important;padding:2px 12px 2px 10px !important;border:1px solid transparent !important;border-radius:12px !important;background:transparent !important;transform:none !important;overflow:visible}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button p{display:flex !important;align-items:center;gap:12px;width:100%;margin:0 !important;font:560 13.5px/1.2 Inter,sans-serif !important;color:var(--muted) !important;transition:transform .3s,color .2s;position:relative;z-index:1}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button p::before{content:"";flex:0 0 26px;width:26px;height:26px;background-color:var(--c,#4c8dff);-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;-webkit-mask-size:15px 15px;mask-size:15px 15px}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button{display:flex !important;align-items:center !important;justify-content:flex-start !important;margin:0 !important;min-height:40px !important;height:auto !important}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button>div,.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button [data-testid="stMarkdownContainer"]{width:100% !important;margin:0 !important;text-align:left !important}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button div{width:100% !important;justify-content:flex-start !important;text-align:left !important}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button p{justify-content:flex-start !important;text-align:left !important}
.stApp [data-testid="stSidebar"] .st-key-nav_overview{margin-top:16px !important}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button:hover{background:linear-gradient(90deg,color-mix(in srgb,var(--c,#4c8dff) 15%,transparent),transparent) !important}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button:hover p{transform:translateX(3px);color:var(--text) !important}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"]{background:linear-gradient(90deg,color-mix(in srgb,var(--c,#4c8dff) 17%,var(--panel-2)),transparent) !important;border-color:color-mix(in srgb,var(--c,#4c8dff) 30%,transparent) !important}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"] p{color:var(--text) !important;font-weight:700 !important}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"] p::before{background-color:var(--c)}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"]::after{content:"";position:absolute;left:-11px;top:50%;width:4px;height:20px;border-radius:0 4px 4px 0;transform:translateY(-50%);background:var(--c);box-shadow:0 0 14px var(--c)}
.sb-status{display:flex;align-items:center;gap:14px;margin-top:22px;padding:10px 12px;border-radius:13px;border:1px solid var(--line-strong);background:var(--panel-2)}
.sb-status .dot{flex:none;width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 22%,transparent)}
.sb-stat{display:flex;flex-direction:column;gap:3px}.sb-stat b{font:750 15px/1 Inter,sans-serif;color:var(--text)}.sb-stat i{font:600 9.5px/1 Inter,sans-serif;font-style:normal;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.sb-prof{display:flex;align-items:center;gap:11px;margin-top:10px;padding:8px 11px;border-radius:13px;border:1px solid var(--line-strong);background:var(--panel-2)}
.sb-prof .av{width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:14px;background:linear-gradient(145deg,#4c8dff,#8b7cff)}
.sb-prof b{display:block;font:650 12.5px/1.2 Inter,sans-serif;color:var(--text)}.sb-prof i{display:block;font:500 11px/1.2 Inter,sans-serif;font-style:normal;color:var(--muted)}
/* ---------- banners / sections / kpis ---------- */
.part-banner{position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;padding:18px 24px 18px 30px;margin:8px 0 16px;border-radius:14px;border:1px solid color-mix(in srgb,var(--tone) 30%,var(--line-strong));
background:radial-gradient(110% 170% at 0% 0%,color-mix(in srgb,var(--tone) 20%,transparent),transparent 60%),linear-gradient(180deg,var(--panel-2),var(--panel));box-shadow:var(--shadow)}
.part-banner::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--tone)}
.pb-step{display:inline-flex;align-items:center;gap:10px;margin-bottom:9px;font:700 10.5px/1 ui-monospace,Consolas,monospace;letter-spacing:.18em;text-transform:uppercase;color:var(--tone)}
.pb-step::before{content:"";width:18px;height:1px;background:currentColor;opacity:.65}
.pb-title{font:750 22px/1.25 Inter,sans-serif;color:var(--text);letter-spacing:-.01em}
.pb-sub{margin-top:4px;max-width:78ch;font-size:13px;line-height:1.5;color:var(--muted)}
.pb-scope{display:inline-flex;align-items:center;gap:9px;font:600 10.5px/1 ui-monospace,Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.pb-scope::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--tone);box-shadow:0 0 0 3px var(--tone-soft)}
.sec-head{position:relative;margin:28px 0 12px;padding:2px 0 2px 16px}
.sec-head::before{content:"";position:absolute;left:0;top:3px;bottom:3px;width:3px;border-radius:3px;background:var(--tone)}
.sh-title{font:750 17px/1.3 Inter,sans-serif;color:var(--text)}.sh-sub{margin-top:3px;font-size:12.5px;color:var(--muted)}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:6px 0 16px}
.kpi{position:relative;overflow:hidden;padding:15px 16px 14px 20px;border-radius:12px;border:1px solid var(--line-strong);background:radial-gradient(130% 160% at 0% 0%,color-mix(in srgb,var(--k) 13%,transparent),transparent 60%),linear-gradient(180deg,var(--panel-2),var(--panel));box-shadow:var(--shadow-sm);transition:border-color .2s}
.kpi:hover{border-color:color-mix(in srgb,var(--k) 45%,var(--line-strong))}
.kpi::before{content:"";position:absolute;left:0;top:14px;bottom:14px;width:3px;border-radius:0 3px 3px 0;background:linear-gradient(180deg,var(--k),color-mix(in srgb,var(--k) 20%,transparent))}
.kpi-l{display:flex;align-items:center;gap:8px;font:600 10.5px/1.2 ui-monospace,Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.kpi-v{margin-top:9px;font:750 27px/1.1 Inter,sans-serif;letter-spacing:-.02em;color:var(--text);font-variant-numeric:tabular-nums}
.kpi-f{margin-top:7px;font-size:12px;color:var(--muted)}
.flow-wrap{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap}
.flow-card{flex:1;min-width:150px;position:relative;padding:15px 17px;border-radius:12px;border:1px solid var(--line-strong);background:linear-gradient(180deg,var(--panel-2),var(--panel));box-shadow:var(--shadow-sm)}
.flow-card .label{font:600 10.5px/1 ui-monospace,Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.flow-card .value{margin-top:9px;font:750 21px/1.1 Inter,sans-serif;letter-spacing:-.02em;color:var(--fc,var(--text))}
.flow-card .foot{margin-top:6px;font-size:11.5px;color:var(--muted)}
.flow-arrow{align-self:center;color:var(--muted);font-size:20px}
.stat-card{padding:15px 17px;border-radius:12px;border:1px solid var(--line-strong);background:linear-gradient(180deg,var(--panel-2),var(--panel));margin-bottom:12px;border-left:3px solid var(--sc,var(--tone))}
.stat-card .t{font:650 14px/1.3 Inter,sans-serif;color:var(--text);margin:5px 0 2px}.stat-card .d{font-size:12.5px;color:var(--muted);line-height:1.5}
.stat-card .e{font:700 10.5px/1 ui-monospace,Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--sc,var(--tone))}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{padding:5px 9px;border-radius:6px;font:600 10.5px/1 ui-monospace,Consolas,monospace;letter-spacing:.06em;color:var(--tone);border:1px solid color-mix(in srgb,var(--tone) 40%,transparent);background:var(--tone-soft)}
/* ---------- tables ---------- */
.pt-shell{margin:8px 0 14px}
.pt-wrap{position:relative;overflow:auto;border:1px solid var(--line-strong);border-bottom:0;border-radius:14px 14px 0 0;background:var(--panel);box-shadow:var(--shadow-sm)}
.pt-wrap::before{content:"";display:block;position:sticky;top:0;left:0;z-index:3;height:2px;margin-bottom:-2px;background:linear-gradient(90deg,var(--tone),transparent 85%)}
table.pt{width:100%;border-collapse:separate;border-spacing:0;font:13px/1.4 Inter,sans-serif;font-variant-numeric:tabular-nums}
table.pt thead th{position:sticky;top:0;z-index:1;text-align:left;padding:13px 18px 11px;background:var(--panel-2);color:var(--muted);font:700 10.5px/1 Inter,sans-serif;letter-spacing:.13em;text-transform:uppercase;border-bottom:1px solid var(--line-strong);white-space:nowrap}
table.pt thead th::before{content:"";display:inline-block;vertical-align:middle;margin:-2px 9px 0 0;width:6px;height:6px;border-radius:50%;background:var(--tc);box-shadow:0 0 0 3px color-mix(in srgb,var(--tc) 24%,transparent)}
table.pt thead th:nth-child(5n+1){--tc:#2fb68e}table.pt thead th:nth-child(5n+2){--tc:#d9a35f}table.pt thead th:nth-child(5n+3){--tc:#e0708c}table.pt thead th:nth-child(5n+4){--tc:#8b7cf6}table.pt thead th:nth-child(5n+5){--tc:#38b2c8}
table.pt tbody td{padding:11px 18px;height:48px;box-sizing:border-box;color:var(--text);border-bottom:1px solid var(--line);vertical-align:middle}
table.pt tbody tr:last-child td{border-bottom:0}
table.pt tbody tr{transition:background .15s}
table.pt tbody tr:hover{background:linear-gradient(90deg,var(--tone-soft),transparent 88%)}
table.pt tbody td:first-child{position:relative;font-weight:650}
table.pt tbody td:first-child::before{content:"";position:absolute;left:0;top:12px;bottom:12px;width:3px;border-radius:0 3px 3px 0;background:var(--tone);opacity:.4;transition:all .18s}
table.pt tbody tr:hover td:first-child::before{opacity:1;top:0;bottom:0}
table.pt td.num,table.pt th.num{text-align:right}
.pt-foot{display:flex;align-items:center;gap:9px;padding:8px 16px;border:1px solid var(--line-strong);border-radius:0 0 14px 14px;background:var(--panel-2);font:600 10.5px/1 ui-monospace,Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.pt-foot i{width:6px;height:6px;border-radius:50%;background:var(--tone)}
.pill{display:inline-flex;align-items:center;gap:7px;padding:3px 11px 3px 9px;border-radius:999px;font:700 10.5px/1.3 Inter,sans-serif;letter-spacing:.09em;text-transform:uppercase;white-space:nowrap;color:var(--p);background:color-mix(in srgb,var(--p) 12%,transparent);box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--p) 40%,transparent)}
.pill::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.pill.good{--p:var(--ok)}.pill.warn{--p:var(--warn)}.pill.bad{--p:var(--bad)}.pill.info{--p:#38b2c8}.pill.mute{--p:var(--muted)}
.bar{display:flex;align-items:center;gap:10px;min-width:120px}.bar-t{flex:1;height:6px;border-radius:6px;background:var(--line-strong);overflow:hidden}.bar-f{height:100%;border-radius:6px;background:var(--tone)}.bar-n{min-width:46px;text-align:right;font-weight:700;font-size:12.5px}
pre.codebox{margin:0;padding:16px 18px;border-radius:12px;border:1px solid var(--line-strong);border-left:3px solid var(--tone);background:var(--panel-3);color:var(--text);font:12.5px/1.6 'JetBrains Mono',Consolas,monospace;overflow:auto;max-height:420px}
.foot{text-align:center;color:var(--muted);font-size:11px;padding-top:22px;letter-spacing:.08em}
@media(prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
'''

# per-item colours + icons for the capsule nav and the sidebar
_ICONS = {
    "grid": "<rect x='3' y='3' width='7' height='9' rx='1.5'/><rect x='14' y='3' width='7' height='5' rx='1.5'/><rect x='14' y='12' width='7' height='9' rx='1.5'/><rect x='3' y='16' width='7' height='5' rx='1.5'/>",
    "chip": "<rect x='6' y='6' width='12' height='12' rx='2'/><path d='M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4'/>",
    "bolt": "<path d='M13 2L4 14h7l-1 8 9-12h-7z'/>",
    "flask": "<path d='M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3'/>",
    "shield": "<path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z'/><path d='M9 12l2 2 4-4'/>",
    "info": "<circle cx='12' cy='12' r='9'/><path d='M12 11v5M12 8h.01'/>",
}
_NAV_ICON = {"Overview": "grid", "Forecast": "chip", "Dispatch": "bolt", "Scenarios": "flask",
             "Assets & Alerts": "shield", "Model Notes": "info"}


def _mask(icon):
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' "
           "stroke-linecap='round' stroke-linejoin='round'>" + _ICONS[icon] + "</svg>")
    return 'url("data:image/svg+xml,' + _up.quote(svg, safe="/:=' ()") + '")'


_rules = []
for _i, _n in enumerate(NAV, start=1):
    _sel = '.stApp .st-key-topnav [role="radiogroup"]>:nth-child(' + str(_i) + ")"
    _m = _mask(_NAV_ICON[_n])
    _rules.append(f"{_sel}{{--c:{TONES[_n]}}}")
    _rules.append(f"{_sel} p::before{{-webkit-mask-image:{_m};mask-image:{_m}}}")
_SB = {  # key -> (panel, group colour)
    "nav_overview": "Overview", "nav_forecast": "Forecast", "nav_dispatch": "Dispatch",
    "nav_scenarios": "Scenarios", "nav_assets": "Assets & Alerts", "nav_notes": "Model Notes",
}
for _k, _p in _SB.items():
    _m = _mask(_NAV_ICON[_p])
    _s = f'.stApp [data-testid="stSidebar"] .st-key-{_k}'
    _rules.append(f"{_s}{{--c:{TONES[_p]}}}")
    _rules.append(f"{_s} button p::before{{-webkit-mask-image:{_m};mask-image:{_m}}}")

st.markdown("<style>" + _root + STATIC_CSS + "".join(_rules) + "</style>", unsafe_allow_html=True)

# ====================== helpers ======================
class Raw(str):
    """Pre-built, trusted HTML cell."""


def esc(x):
    return html.escape(str(x))


def pill(text, kind="mute"):
    return Raw(f"<span class='pill {kind}'>{esc(text)}</span>")


def bar(pct, label=None):
    pct = max(0.0, min(100.0, float(pct)))
    return Raw(f"<div class='bar'><div class='bar-t'><div class='bar-f' style='width:{pct:.0f}%'></div></div>"
               f"<span class='bar-n'>{esc(label if label is not None else f'{pct:.0f}%')}</span></div>")


def table(headers, rows, num_cols=(), max_h=None, foot=None):
    th = "".join(f"<th class='{'num' if i in num_cols else ''}'>{esc(h)}</th>" for i, h in enumerate(headers))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td class='{'num' if i in num_cols else ''}'>{c if isinstance(c, Raw) else esc(c)}</td>" for i, c in enumerate(r)
        ) + "</tr>"
    style = f" style='max-height:{int(max_h)}px'" if max_h else ""
    f = foot if foot is not None else f"{len(rows)} rows"
    st.markdown(f"<div class='pt-shell'><div class='pt-wrap'{style}><table class='pt'><thead><tr>{th}</tr></thead>"
                f"<tbody>{body}</tbody></table></div><div class='pt-foot'><i></i>{esc(f)}</div></div>",
                unsafe_allow_html=True)


def banner(step, title, sub, scope):
    st.markdown(f"<div class='part-banner'><div><div class='pb-step'>{esc(step)}</div><div class='pb-title'>{esc(title)}</div>"
                f"<div class='pb-sub'>{esc(sub)}</div></div><div class='pb-scope'>{esc(scope)}</div></div>", unsafe_allow_html=True)


def section(title, sub=""):
    s = f"<div class='sh-sub'>{esc(sub)}</div>" if sub else ""
    st.markdown(f"<div class='sec-head'><div class='sh-title'>{esc(title)}</div>{s}</div>", unsafe_allow_html=True)


def kpis(items):
    cols = [TONE, "#4da3ff", "#a78bfa", "#3ddc97", "#f5b14a"] if DARK else [TONE, "#0070f3", "#7c5ce6", "#0f9a73", "#c47f00"]
    h = "<div class='kpi-grid'>"
    for i, (lab, val, foot) in enumerate(items):
        h += (f"<div class='kpi' style='--k:{cols[i % 5]}'><div class='kpi-l'>{esc(lab)}</div>"
              f"<div class='kpi-v'>{esc(val)}</div><div class='kpi-f'>{esc(foot)}</div></div>")
    st.markdown(h + "</div>", unsafe_allow_html=True)


def stat_card(eyebrow, title, detail, color):
    st.markdown(f"<div class='stat-card' style='--sc:{color}'><div class='e'>{esc(eyebrow)}</div>"
                f"<div class='t'>{esc(title)}</div><div class='d'>{esc(detail)}</div></div>", unsafe_allow_html=True)


def chart(fig, height=330, legend=True, title=None):
    fig.update_layout(
        height=height, paper_bgcolor=TOK["panel"], plot_bgcolor=TOK["panel"], font=dict(color=TOK["text"], size=12),
        title=dict(text=title, font=dict(size=13, color=TOK["text"]), x=0.01, xanchor="left") if title else None,
        margin=dict(l=18, r=18, t=46 if title else 22, b=24), hovermode="x unified",
        hoverlabel=dict(bgcolor=TOK["panel2"], bordercolor=TOK["line2"], font_color=TOK["text"]),
        legend=dict(orientation="h", y=-0.2, x=0, bgcolor="rgba(0,0,0,0)") if legend else dict(visible=False),
        xaxis=dict(gridcolor=TOK["line"], zerolinecolor=TOK["line2"], showline=False),
        yaxis=dict(gridcolor=TOK["line"], zerolinecolor=TOK["line2"], showline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def litres(kwh):
    return kwh / C.KWH_PER_LITRE_DIESEL


def go_to(panel):
    st.session_state["active_panel"] = panel
    st.rerun()


# ====================== data ======================
@st.cache_data(show_spinner="Generating the polar station simulation…")
def load_data():
    return generate_year()


@st.cache_data(show_spinner="Training load forecast…")
def get_forecast():
    return forecast_load(load_data())


@st.cache_data(show_spinner="Optimising energy dispatch…")
def get_dispatch(start, hours):
    d = load_data().iloc[start:start + hours]
    out, status = optimize_dispatch(d.load_kw, d.solar_kw, d.wind_kw)
    out.index = d.index
    return out, status


df = load_data()
SCENARIOS = {
    "Summer / midnight sun": pd.Timestamp(f"{C.SIM_YEAR}-01-10"),
    "Polar night / July": pd.Timestamp(f"{C.SIM_YEAR}-07-10"),
    "Shoulder season / April": pd.Timestamp(f"{C.SIM_YEAR}-04-10"),
}

# ====================== sidebar ======================
def nav_btn(label, key, panel):
    if st.button(label, key=key, use_container_width=True, type="primary" if PANEL == panel else "secondary"):
        go_to(panel)


def group(label, color):
    st.markdown(f"<div class='sidebar-group-label' style='--g:{color}'>{esc(label)}</div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("<div class='sidebar-brand-v2'><div class='brand-mark'>❄️</div><div><div class='brand-name'>POLAR EMS</div>"
                "<div class='brand-role'>Station microgrid console</div></div></div>", unsafe_allow_html=True)
    nav_btn("Overview", "nav_overview", "Overview")
    group("Operations", TONES["Forecast"])
    nav_btn("Forecast", "nav_forecast", "Forecast")
    nav_btn("Dispatch", "nav_dispatch", "Dispatch")
    group("Analysis", TONES["Scenarios"])
    nav_btn("Scenarios", "nav_scenarios", "Scenarios")
    nav_btn("Assets & Alerts", "nav_assets", "Assets & Alerts")
    group("System", TONES["Model Notes"])
    nav_btn("Model Notes", "nav_notes", "Model Notes")
    st.markdown(
        f"<div class='sb-status'><span class='dot'></span><div class='sb-stat'><b>{C.PV_CAPACITY_KW + C.WIND_CAPACITY_KW:.0f} kW</b><i>Renewables</i></div>"
        f"<div class='sb-stat'><b>{C.BATTERY_CAPACITY_KWH:.0f} kWh</b><i>Battery</i></div></div>"
        f"<div class='sb-prof'><div class='av'>🧊</div><div><b>Simulation mode</b><i>Seed {C.RANDOM_SEED} · no live SCADA</i></div></div>",
        unsafe_allow_html=True)

# ====================== top banner + nav ======================
st.markdown(
    f"""<div class="hdr"><h1>❄️ Polar EMS · Research-Station Energy</h1><div class="hdr-meta">{C.LATITUDE_DEG:.0f}° lat · {C.SIM_YEAR} · <b>sim online</b></div></div>""",
    unsafe_allow_html=True)

with st.container(key="topnav"):
    st.radio("Workflow", NAV, key="active_panel", horizontal=True, label_visibility="collapsed")
PANEL = st.session_state["active_panel"]
TONE = TONES.get(PANEL, TONE)
st.markdown(f"<style>html body .stApp{{--tone:{TONE};--tone-soft:color-mix(in srgb,{TONE} 14%,transparent)}}</style>", unsafe_allow_html=True)

# ====================== controls + shared run ======================
with st.container(key="ctl_card"):
    c1, c2, c3 = st.columns([2.2, 1.4, 1.6])
    scenario = c1.selectbox("Operating scenario", list(SCENARIOS), key="main_scenario")
    days = c2.slider("Dispatch window", 2, 14, 7, 1, format="%d days", key="main_days")
    c3.markdown(f"<div class='chips' style='padding-top:28px'><span class='chip'>{days} DAYS</span><span class='chip'>SIMULATED</span></div>", unsafe_allow_html=True)

start = int(df.index.get_loc(SCENARIOS[scenario]))
disp, solver_status = get_dispatch(start, days * 24)
ops = calculate_metrics(disp, solver_status)
manifest = build_run_manifest(disp, scenario, solver_status)
alerts = build_operational_alerts(ops)
saved_kwh = max(0.0, ops.baseline_diesel_kwh - ops.diesel_kwh)
MIX = [("Solar", "solar_used", SOLAR), ("Wind", "wind_used", WINDC), ("Battery", "discharge_kw", BATT), ("Diesel", "diesel_kw", DIESEL)]

# ====================== pages ======================
if PANEL == "Overview":
    banner("01 · Overview", "Station energy at a glance", "Supply mix, fuel and battery reserve for the selected window.", f"{scenario} · {days} days")
    kpis([
        ("Station demand", f"{ops.average_load_kw:.1f} kW", f"Peak {ops.peak_load_kw:.1f} kW"),
        ("Diesel consumed", f"{ops.diesel_litres:,.0f} L", f"{ops.diesel_kwh:,.0f} kWh in window"),
        ("Renewable share", f"{ops.renewable_share_pct:.1f}%", f"{ops.renewable_kwh:,.0f} kWh supplied"),
        ("Fuel avoided", f"{litres(saved_kwh):,.0f} L", f"{(saved_kwh / ops.baseline_diesel_kwh * 100 if ops.baseline_diesel_kwh else 0):.1f}% vs no-battery baseline"),
        ("Battery reserve", f"{ops.ending_soc_pct:.0f}%", f"{ops.ending_soc_kwh:.0f} / {C.BATTERY_CAPACITY_KWH:.0f} kWh"),
    ])
    section("Power flow", "Energy delivered by each source during the window")
    flow = [("SOLAR", f"{disp.solar_used.sum():,.0f} kWh", f"Peak available {disp.solar_avail.max():.1f} kW", SOLAR),
            ("WIND", f"{disp.wind_used.sum():,.0f} kWh", f"Peak available {disp.wind_avail.max():.1f} kW", WINDC),
            ("BATTERY", f"{ops.battery_discharge_kwh:,.0f} kWh", f"Ends at {ops.ending_soc_pct:.0f}%", BATT),
            ("DIESEL", f"{ops.diesel_kwh:,.0f} kWh", f"Active {ops.diesel_active_hours} h", DIESEL),
            ("STATION LOAD", f"{ops.load_kwh:,.0f} kWh", f"Average {ops.average_load_kw:.1f} kW", TOK["text"])]
    h = "<div class='flow-wrap'>"
    for i, (n, v, f, col) in enumerate(flow):
        h += f"<div class='flow-card' style='--fc:{col}'><div class='label'>{n}</div><div class='value'>{v}</div><div class='foot'>{f}</div></div>"
        if i < len(flow) - 1:
            h += "<div class='flow-arrow'>›</div>"
    st.markdown(h + "</div>", unsafe_allow_html=True)

    section("Supply vs demand", "Hourly dispatch stack against station load")
    left, right = st.columns([1.7, 1])
    with left:
        fig = go.Figure()
        for name, key, col in MIX:
            fig.add_trace(go.Scatter(x=disp.index, y=disp[key], name=name, stackgroup="s", mode="lines", line=dict(width=.8, color=col),
                                     hovertemplate=f"{name}: %{{y:.1f}} kW<extra></extra>"))
        fig.add_trace(go.Scatter(x=disp.index, y=disp.load_kw, name="Demand", line=dict(color=TOK["text"], width=2.2),
                                 hovertemplate="Demand: %{y:.1f} kW<extra></extra>"))
        fig.update_yaxes(title="Power (kW)")
        chart(fig, 360, title="Power supply vs station demand")
    with right:
        g = go.Figure(go.Indicator(mode="gauge+number", value=ops.ending_soc_pct, number=dict(suffix="%", font=dict(color=TOK["text"], size=34)),
                                   gauge=dict(axis=dict(range=[0, 100], tickcolor=TOK["muted"]), bar=dict(color=BATT), bgcolor=TOK["panel2"], bordercolor=TOK["line2"],
                                              steps=[dict(range=[0, C.BATTERY_MIN_SOC_FRAC * 100], color="rgba(214,51,77,.28)")],
                                              threshold=dict(line=dict(color=DIESEL, width=3), thickness=.8, value=C.BATTERY_MIN_SOC_FRAC * 100))))
        g.update_layout(height=250, paper_bgcolor=TOK["panel"], font=dict(color=TOK["text"]), margin=dict(t=30, b=10, l=16, r=16),
                        title=dict(text="Battery reserve", font=dict(size=13, color=TOK["text"]), x=.02))
        st.plotly_chart(g, use_container_width=True, config={"displayModeBar": False})
        stat_card("Dispatch engine", solver_status, f"{len(disp)} hourly steps · LP with deterministic fallback", TOK["ok"] if solver_status == "Optimal" else TOK["warn"])

    section("Daily contribution")
    daily = disp.resample("D").sum(numeric_only=True)
    fd = go.Figure()
    for name, key, col in MIX:
        fd.add_bar(x=daily.index, y=daily[key], name=name, marker_color=col)
    fd.update_layout(barmode="stack")
    fd.update_yaxes(title="Energy (kWh)")
    chart(fd, 300, title="Daily energy contribution")
    d1, d2, _ = st.columns([1.2, 1.2, 3])
    d1.download_button("Dispatch CSV", disp.to_csv().encode(), "polar_ems_dispatch.csv", "text/csv", use_container_width=True)
    d2.download_button("Report (Markdown)", export_report(ops, alerts, manifest), "polar_ems_report.md", "text/markdown", use_container_width=True)
    with st.expander("Run provenance"):
        st.markdown(f"<pre class='codebox'>{esc(json.dumps(manifest, indent=2))}</pre>", unsafe_allow_html=True)

elif PANEL == "Forecast":
    res, mae = get_forecast()
    mape = float((np.abs(res.load_actual - res.load_pred) / res.load_actual.clip(lower=1)).mean() * 100)
    banner("02 · Forecast", "AI load forecasting", "Gradient-boosted model scored on a held-out 30-day window.", "Held-out test period")
    kpis([("Mean absolute error", f"{mae:.2f} kW", "Lower is better"), ("MAPE", f"{mape:.2f}%", "Mean absolute % error"),
          ("Test window", f"{len(res):,} h", "Final 30 days, never trained on"),
          ("Peak actual", f"{res.load_actual.max():.1f} kW", "In test window"), ("Peak predicted", f"{res.load_pred.max():.1f} kW", "In test window")])
    ff = go.Figure()
    ff.add_trace(go.Scatter(x=res.index, y=res.load_actual, name="Actual", line=dict(color=TONE, width=1.6), fill="tozeroy", fillcolor="rgba(120,140,255,.08)"))
    ff.add_trace(go.Scatter(x=res.index, y=res.load_pred, name="Predicted", line=dict(color=DIESEL, width=1.7, dash="dot")))
    ff.update_yaxes(title="Load (kW)")
    chart(ff, 410, title="Actual demand vs model prediction")
    err = res.load_actual - res.load_pred
    fe = go.Figure(go.Histogram(x=err, nbinsx=30, marker_color=WINDC))
    fe.update_xaxes(title="Actual − predicted (kW)"); fe.update_yaxes(title="Frequency")
    chart(fe, 260, legend=False, title="Forecast error distribution")
    with st.expander("Methodology"):
        st.markdown("Gradient-boosted regression on cyclical calendar features, temperature, 24 h and 168 h load lags and a rolling 24 h mean. The final 30 days are held out from training.")

elif PANEL == "Dispatch":
    banner("03 · Dispatch", "Optimised energy dispatch", "Renewables first, battery second, diesel last.", f"{scenario} · {days} days")
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=disp.index, y=disp.load_kw, name="Demand", line=dict(color=TOK["text"], width=2.2)))
    f1.add_trace(go.Scatter(x=disp.index, y=disp.solar_avail, name="Solar available", line=dict(color=SOLAR, width=1.4, dash="dot")))
    f1.add_trace(go.Scatter(x=disp.index, y=disp.wind_avail, name="Wind available", line=dict(color=WINDC, width=1.4, dash="dot")))
    f1.update_yaxes(title="Power (kW)")
    chart(f1, 340, title="Demand and renewable availability")
    f2 = go.Figure(go.Scatter(x=disp.index, y=disp.soc_kwh, name="Battery", line=dict(color=BATT, width=2), fill="tozeroy", fillcolor="rgba(61,220,151,.12)"))
    f2.add_hline(y=C.BATTERY_MIN_SOC_FRAC * C.BATTERY_CAPACITY_KWH, line_dash="dash", line_color=DIESEL, annotation_text="Reserve floor")
    f2.add_hline(y=C.BATTERY_CAPACITY_KWH, line_dash="dot", line_color=TOK["muted"], annotation_text="Capacity")
    f2.update_yaxes(title="Stored energy (kWh)")
    chart(f2, 320, legend=False, title="Battery state-of-charge")
    section("Hourly dispatch log", "First 168 hours of the window")
    rows = []
    for ts, r in disp.head(168).iterrows():
        rows.append([ts.strftime("%d %b %H:%M"), f"{r.load_kw:.1f}", f"{r.solar_used:.1f}", f"{r.wind_used:.1f}", f"{r.discharge_kw:.1f}",
                     f"{r.charge_kw:.1f}", pill(f"{r.diesel_kw:.1f} kW", "warn") if r.diesel_kw > .1 else pill("Off", "good"),
                     bar(r.soc_kwh / C.BATTERY_CAPACITY_KWH * 100)])
    table(["Time", "Load kW", "Solar kW", "Wind kW", "Discharge kW", "Charge kW", "Diesel", "Battery"], rows, num_cols=(1, 2, 3, 4, 5), max_h=420,
          foot=f"{min(168, len(disp))} of {len(disp)} hours")
    st.download_button("Export full dispatch (CSV)", disp.to_csv().encode(), "polar_dispatch_detail.csv", "text/csv")

elif PANEL == "Scenarios":
    banner("04 · Scenarios", "Stress tests and forecast margin", "What-if load and renewable multipliers, dispatched by the same optimiser.", "Bounded what-if analysis")
    section("Stress scenario", "Multipliers apply to the selected window")
    s1, s2, s3, s4 = st.columns([1, 1, 1, 1.2])
    lf = s1.slider("Load ×", 0.5, 1.8, 1.0, 0.05, key="sc_load")
    sf = s2.slider("Solar ×", 0.0, 1.5, 1.0, 0.05, key="sc_solar")
    wf = s3.slider("Wind ×", 0.0, 1.5, 1.0, 0.05, key="sc_wind")
    label = s4.text_input("Label", value="Custom", key="sc_label")
    win = df.iloc[start:start + days * 24]
    b1, b2, _ = st.columns([1.2, 1.2, 3])
    if b1.button("Run scenario", type="primary", use_container_width=True):
        try:
            st.session_state.setdefault("stress_runs", []).append(run_stress_scenario(win, lf, sf, wf, label or "Scenario"))
        except Exception as exc:  # bounded inputs, but never crash the console
            st.error(f"Scenario failed: {exc}")
    if b2.button("Clear results", use_container_width=True):
        st.session_state["stress_runs"] = []
    runs = [{"scenario": "Baseline (current window)", "solver_status": solver_status, "load_factor": 1.0, "solar_factor": 1.0, "wind_factor": 1.0,
             "diesel_litres": ops.diesel_litres, "renewable_share_pct": ops.renewable_share_pct, "ending_soc_pct": ops.ending_soc_pct}] + st.session_state.get("stress_runs", [])
    section("Results", "Baseline plus every scenario you have run")
    table(["Scenario", "Load ×", "Solar ×", "Wind ×", "Diesel", "Renewable", "Battery end", "Solver"],
          [[r["scenario"], f"{r['load_factor']:.2f}", f"{r['solar_factor']:.2f}", f"{r['wind_factor']:.2f}", f"{r['diesel_litres']:,.0f} L",
            bar(r["renewable_share_pct"], f"{r['renewable_share_pct']:.1f}%"), f"{r['ending_soc_pct']:.0f}%",
            pill(r["solver_status"], "good" if r["solver_status"] == "Optimal" else "warn")] for r in runs], num_cols=(1, 2, 3, 4))
    if len(runs) > 1:
        fr = go.Figure(go.Bar(x=[r["scenario"] for r in runs], y=[r["diesel_litres"] for r in runs], marker_color=[TONE] + [DIESEL] * (len(runs) - 1)))
        fr.update_yaxes(title="Diesel (L)")
        chart(fr, 300, legend=False, title="Diesel consumption by scenario")
    section("Forecast-aware margin", "Dispatch against a conservative load envelope")
    m1, m2 = st.columns([1.4, 1])
    margin = m1.slider("Load margin (%)", 0, 25, 5, 1, key="sc_margin")
    if m2.button("Apply margin", use_container_width=True):
        mdisp, mstatus = forecast_aware_dispatch(win, float(margin))
        mm = calculate_metrics(mdisp, mstatus)
        stat_card("Result", f"{mm.diesel_litres:,.0f} L diesel at +{margin}% load",
                  f"{mm.diesel_litres - ops.diesel_litres:+,.0f} L vs baseline · renewable {mm.renewable_share_pct:.1f}% · battery end {mm.ending_soc_pct:.0f}% · {mstatus}", TONE)

elif PANEL == "Assets & Alerts":
    banner("05 · Assets & Alerts", "Asset status, health and alerts", "Configured capacity, simulated wear estimates and operating alerts.", f"{scenario} · {days} days")
    section("Installed assets")
    ac = st.columns(4)
    for col, (name, cap, det, cat, colr) in zip(ac, [
        ("Solar PV array", f"{C.PV_CAPACITY_KW:.0f} kW", f"Peak available {disp.solar_avail.max():.1f} kW", "GENERATION", SOLAR),
        ("Wind turbines", f"{C.WIND_CAPACITY_KW:.0f} kW", f"Peak available {disp.wind_avail.max():.1f} kW", "GENERATION", WINDC),
        ("Battery storage", f"{C.BATTERY_CAPACITY_KWH:.0f} kWh", f"Window ends at {ops.ending_soc_pct:.0f}%", "STORAGE", BATT),
        ("Diesel genset", f"{C.DIESEL_CAPACITY_KW:.0f} kW", f"Active {ops.diesel_active_hours} simulated hours", "BACKUP", DIESEL)]):
        with col:
            st.markdown(f"<div class='flow-card' style='--fc:{colr}'><div class='label'>{cat}</div><div class='value'>{cap}</div>"
                        f"<div style='font-weight:650;margin-top:7px'>{name}</div><div class='foot'>{det}</div></div>", unsafe_allow_html=True)
    bh = battery_health(disp)
    section("Battery health", bh["assumption"])
    kpis([("Estimated SOH", f"{bh['estimated_soh_pct']:.1f}%", "Linear cycle estimate"), ("Capacity now", f"{bh['estimated_capacity_kwh']:.0f} kWh", "Of nominal"),
          ("Cycles this run", f"{bh['equivalent_full_cycles_this_run']:.2f}", "Equivalent full cycles"), ("Discharged", f"{bh['discharged_kwh']:,.0f} kWh", "In window"),
          ("Charged", f"{bh['charged_kwh']:,.0f} kWh", "In window")])
    section("Maintenance signals", "Threshold-based, explainable, simulated")
    kind = {"HIGH": "bad", "WATCH": "warn", "OK": "good"}
    table(["Priority", "Asset", "Signal", "Basis"], [[pill(s["priority"], kind.get(s["priority"], "mute")), s["asset"], s["signal"], s["basis"]] for s in maintenance_signals(disp, bh)])
    section("Operations alert log", "Generated from the selected window, not real station alarms")
    sev = {"success": "good", "info": "info", "warning": "warn", "notice": "bad"}
    table(["Severity", "Code", "Title", "Detail"], [[pill(a["severity"], sev.get(a["severity"], "mute")), a["code"], a["title"], a["detail"]] for a in alerts])
    section("Run history", "Newest 50 runs saved locally")
    h1, h2, _ = st.columns([1.2, 1.2, 3])
    if h1.button("Save this run", type="primary", use_container_width=True):
        save_history_run(build_run_summary(disp, scenario, solver_status))
    hist = load_history()
    if hist:
        table(["Saved (UTC)", "Scenario", "Hours", "Diesel", "Renewable", "Battery end", "SOH"],
              [[str(r.get("saved_at_utc", ""))[:16].replace("T", " "), r.get("scenario", ""), r.get("hours", ""), f"{r.get('diesel_litres', 0):,.0f} L",
                f"{r.get('renewable_share_pct', 0):.1f}%", f"{r.get('ending_soc_pct', 0):.0f}%", f"{r.get('battery_soh_pct', 0):.1f}%"] for r in reversed(hist)],
              num_cols=(2, 3), max_h=320)
    else:
        st.markdown("<div class='stat-card'><div class='d'>No runs saved yet.</div></div>", unsafe_allow_html=True)
    d1, d2, _ = st.columns([1.2, 1.2, 3])
    d1.download_button("Report (Markdown)", export_report(ops, alerts, manifest), "polar_ems_report.md", "text/markdown", use_container_width=True)
    d2.download_button("Manifest (JSON)", json.dumps(manifest, indent=2), "polar_ems_manifest.json", "application/json", use_container_width=True)

else:  # Model Notes
    banner("06 · Model Notes", "Transparency and diagnostics", "Runtime checks, configuration validation and modelling assumptions.", "System")
    section("Runtime diagnostics")
    table(["Component", "Status", "Detail"], [[d["component"], pill("Available" if d["available"] else "Missing", "good" if d["available"] else "bad"), d["detail"]] for d in runtime_diagnostics()])
    section("Configuration validation")
    table(["Check", "Result", "Rule"], [[d["check"], pill("Pass" if d["passed"] else "Fail", "good" if d["passed"] else "bad"), d["detail"]] for d in configuration_diagnostics()])
    section("Installed capacity and parameters")
    table(["Parameter", "Value"], [[a, b] for a, b in [
        ("PV", f"{C.PV_CAPACITY_KW:.1f} kW"), ("Wind", f"{C.WIND_CAPACITY_KW:.1f} kW"), ("Diesel backup", f"{C.DIESEL_CAPACITY_KW:.1f} kW"),
        ("Battery", f"{C.BATTERY_CAPACITY_KWH:.1f} kWh"), ("Minimum SOC", f"{C.BATTERY_MIN_SOC_FRAC * 100:.0f}%"), ("Initial SOC", f"{C.BATTERY_INIT_SOC_FRAC * 100:.0f}%"),
        ("Latitude", f"{C.LATITUDE_DEG:.1f}°"), ("Diesel energy factor", f"{C.KWH_PER_LITRE_DIESEL:.2f} kWh/L"), ("Diesel CO₂ factor", f"{C.CO2_KG_PER_LITRE_DIESEL:.2f} kg/L")]])
    section("Assumptions")
    for e, t, d in [("Telemetry", "Seeded synthetic year", "Reproducible hourly data, not a live SCADA feed."),
                    ("Solar geometry", "70°S", "Polar night and midnight sun emerge from the solar model."),
                    ("Forecast", "Gradient-boosted regressor", "Calendar, temperature and lag features; final 30 days held out."),
                    ("Dispatch", "PuLP/CBC LP", "Deterministic renewable-first fallback when the solver is unavailable."),
                    ("Fuel baseline", "Renewables, no battery", "Remaining load by diesel; savings are measured against this.")]:
        stat_card(e, t, d, TONE)
    with st.expander("Project pipeline"):
        st.markdown("<pre class='codebox'>config.py → data_sim.py → forecast.py → optimizer.py → ems_core.py / smart_ops.py → app.py</pre>", unsafe_allow_html=True)

st.markdown("<div class='foot'>POLAR EMS · SIH 2026 · SIH26061 · SIMULATION ENVIRONMENT</div>", unsafe_allow_html=True)
