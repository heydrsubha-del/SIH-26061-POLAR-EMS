# Polar EMS — Advanced Energy Operations System

**SIH 2026 · SIH26061**

Polar EMS forecasts research-station electricity demand and dispatches solar, wind, battery storage, and diesel backup to reduce diesel use. The project uses a deterministic Antarctic-inspired simulation and must not be represented as connected to real station SCADA unless a real telemetry adapter is separately configured.

## Architecture

```text
config.py          Physical assumptions and simulation configuration
       │
data_sim.py        Seeded hourly weather, renewable generation and demand
       ├───────────────┐
forecast.py        optimizer.py
ML load forecast   LP dispatch / deterministic safe fallback
       └───────┬───────┘
            ems_core.py     Validated metrics, alerts, run manifest, report
            health.py       Runtime and configuration diagnostics
                │
              app.py         Streamlit operations dashboard
```

The dashboard is the presentation layer. Metrics, validation, alert generation, run provenance, and report creation are implemented as framework-independent services so they can be tested without launching Streamlit.

## Advanced engineering features

- **Validated dispatch contract:** required columns, finite values, non-negative physical quantities, and battery SOC bounds are checked before KPIs and exports are generated.
- **Central metric service:** energy, renewable contribution, diesel, baseline fuel, avoided fuel, CO₂, demand, and battery reserve are computed in one place.
- **Deterministic operations alerts:** alert codes and severity levels are generated from the selected dispatch window. Fallback solver use is explicitly surfaced.
- **Reproducible run manifest:** scenario, simulated period, seed, solver status, data mode, generation time, and SHA-256 digest of dispatch output are captured.
- **Downloadable report:** Markdown operations summary and JSON run manifest accompany the raw dispatch CSV.
- **Runtime and configuration diagnostics:** dependency availability and physical parameter relationships are shown in the app.
- **Pure-core tests:** calculation, validation, alert, reporting, and provenance tests run independently of Streamlit.
- **Cached simulation, forecast, and dispatch:** avoids recomputing the seeded year and model work on every Streamlit rerun.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Run checks

```bash
python -m pytest
python data_sim.py
python forecast.py
python optimizer.py
```

The optimizer script may report `Fallback (PuLP unavailable)` if PuLP/CBC is not installed in the active environment. This is expected in minimal test environments; install `pulp` and a working CBC solver to verify the LP path.

### Launch

```bash
streamlit run app.py
```

## Solver behavior

PuLP/CBC is used when available. The optimizer module provides a deterministic fallback when PuLP or its solver is unavailable. The dashboard does not conceal this: the solver status is shown, a fallback alert is created, and the run manifest/report record the status. For authoritative LP optimization, install a working PuLP/CBC environment and confirm the solver reports `Optimal`.

## Data and operational honesty

- The included input series is **seeded synthetic data**, not live station telemetry.
- The station location and installed capacities are configurable assumptions in `config.py`.
- Fuel and CO₂ numbers are derived from the conversion factors in `config.py`; they are estimates, not measured emissions.
- Alerts refer only to the selected simulated window. They are not real operational alarms.
- Forecast accuracy is measured on the final held-out 30-day window from the seeded dataset.

## Extending toward a production EMS

A production deployment should add a separately tested telemetry adapter, authenticated operator roles, durable time-series storage, event/audit retention, model drift monitoring, signed configuration releases, and a safety review of control actions. The current project is decision-support simulation: it does not send commands to physical assets.
