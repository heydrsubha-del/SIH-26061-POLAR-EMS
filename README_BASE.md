# Polar EMS — Vercel-inspired operations dashboard

Polar EMS is a simulation-driven energy management dashboard for SIH26061. This version keeps the existing simulator, load forecast, and dispatch optimizer while improving usability:

- Main-canvas **Dark mode** toggle (theme remains accessible when the sidebar is collapsed)
- Scenario and dispatch-window controls moved to the main canvas
- Sidebar now contains supplementary system information only
- Navigation remains in the main canvas tabs, so collapsing the sidebar does not hide primary navigation or controls
- Theme-aware dashboard surfaces, metrics, Plotly chart backgrounds, borders, and typography


## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```


## Data note

The app uses reproducible simulated polar-station data. It is not live SCADA telemetry.
