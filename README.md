# Polar Smart Energy Management — Operations Dashboard

**Smart India Hackathon 2026 · SIH26061**

A simulated energy operations dashboard for a polar research station. The system generates reproducible hourly weather/load data, forecasts demand with a trained gradient-boosted model, and optimizes renewable/battery/diesel dispatch.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd:       .venv\Scripts\activate.bat
# macOS/Linux:       source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```


## Dashboard

- **Overview:** selected-window energy supply mix, station demand, battery reserve gauge, dispatch status, daily energy and CSV export.
- **Forecast:** held-out test-period actual vs prediction, MAE/MAPE, error distribution.
- **Dispatch:** renewable availability, load, battery trajectory, dispatch detail and CSV.
- **Assets & Alerts:** configured asset capacities and scenario-derived operational alerts.
- **Model Notes:** transparent assumptions and simulation parameters.


## Data transparency

This project is a **simulation environment**, not a live SCADA connection. Data is generated with a fixed seed for reproducibility. The forecast model is evaluated on the final 30 days held out from training. Dispatch uses PuLP/CBC when available and the optimizer's deterministic fallback when unavailable.
