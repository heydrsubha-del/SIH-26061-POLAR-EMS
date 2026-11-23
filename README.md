# AI-Driven Smart Energy Management for Polar Research Stations
**Smart India Hackathon 2026 · Problem Statement SIH26061 (MoES / NCPOR)**

A software system that (1) **forecasts** a polar station's electricity demand with
a trained ML model and (2) **optimises** how solar, wind, a battery, and a diesel
genset are used so that **diesel fuel is minimised** — shown on a live dashboard.

---

## How it fits together (the pipeline)

```
config.py     shared numbers (capacities, battery, latitude)
   │
data_sim.py   Step 1  →  one simulated year of hourly weather + load
   │
forecast.py   Step 2  →  ML model predicts load (trained, held-out tested)
   │
optimizer.py  Step 3  →  linear program: cheapest fuel dispatch, hour by hour
   │
app.py        Step 4  →  Streamlit dashboard = your live demo
```

Each of steps 1–3 runs on its own (`python data_sim.py`, etc.) and prints a
sanity check, so you can build and verify one piece at a time.

---

## Setup (Windows)

1. **Install Python 3.11+** — from https://www.python.org/downloads/ and tick
   *"Add python.exe to PATH"*, or in PowerShell:
   ```
   winget install Python.Python.3.12
   ```
   Close and reopen the terminal, then check: `python --version`

2. **Create & activate a virtual environment** (in this folder):
   ```
   python -m venv .venv
   ```
   - PowerShell:  `.venv\Scripts\Activate.ps1`
   - cmd.exe:     `.venv\Scripts\activate.bat`
   - Git Bash:    `source .venv/Scripts/activate`

3. **Install the libraries:**
   ```
   pip install -r requirements.txt
   ```

## Run

Verify each piece (optional but recommended):
```
python data_sim.py     # prints yearly stats + polar-night / midnight-sun check
python forecast.py     # prints forecast error (MAE / MAPE)
python optimizer.py    # prints solver status + fuel saving %
```

Launch the demo:
```
streamlit run app.py
```
A browser tab opens at http://localhost:8501.

---

## 90-second demo script (what to say to judges)

1. **"Polar stations burn expensive, hard-to-resupply diesel. We cut it two
   ways: predict demand, then optimise supply."**
2. Point at **Section 1**: *"A trained model forecasts load on 30 days it never
   saw — error is only a few percent."*
3. Switch scenario to **Summer**: *"Midnight sun → lots of solar → almost no
   diesel; the battery soaks up the surplus."*
4. Switch to **Polar night**: *"Sun never rises, so we lean on wind + battery,
   and diesel only fills the gaps — here's the fuel and CO₂ we still save."*
5. Point at the KPI tiles: **litres of fuel saved, renewable share, CO₂ avoided.**

---

## Honesty notes (say these — judges reward them)

- Inputs are **physically simulated**, because no polar-station energy data is
  public. The solar model uses real solar geometry for 70°S, so polar night and
  the midnight sun are genuine, not faked.
- The "AI" is a **real trained model** (gradient boosting) with honest held-out
  error — not a rule dressed up as AI.
- Feeding **real** station data later needs only swapping `data_sim.py` for a CSV
  loader with the same columns.

## Easy ways to go further (if you have time)

- Swap in a real dataset (UCI ElectricityLoadDiagrams) via a CSV loader.
- Add a diesel on/off **binary** variable to `optimizer.py` (makes it a true
  MILP with minimum-run constraints).
- Add a weather/uncertainty slider to show the plan adapting.
- Export a PDF "energy report" from the dashboard.

## Troubleshooting

- `python` not found → reopen the terminal after install; Windows may need the
  Microsoft Store alias disabled (Settings → Apps → App execution aliases).
- Solver prints noise → it's silenced (`msg=0`); status should read `Optimal`.
- Charts empty → make sure you ran `streamlit run app.py`, not `python app.py`.
