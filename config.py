"""Central configuration for the Polar Energy Management System.

Every physical assumption lives here so all modules agree on the same numbers.
Tweak these to tell a different story (bigger battery, more wind, etc.).
"""

# --- Location: a stand-in Antarctic coastal research station ---
LATITUDE_DEG = -70.0          # high southern latitude -> polar night / midnight sun

# --- Installed capacity (kW) ---
PV_CAPACITY_KW     = 50.0     # solar panels
WIND_CAPACITY_KW   = 120.0    # wind turbine(s): the primary renewable at the poles
DIESEL_CAPACITY_KW = 200.0    # backup genset. MUST exceed peak load so a feasible
                              # solution ALWAYS exists (the demo can never crash).

# --- Battery pack ---
BATTERY_CAPACITY_KWH     = 400.0
BATTERY_MAX_CHARGE_KW    = 100.0
BATTERY_MAX_DISCHARGE_KW = 100.0
BATTERY_MIN_SOC_FRAC     = 0.10   # never drain below 10 %
BATTERY_MAX_SOC_FRAC     = 1.00
BATTERY_INIT_SOC_FRAC    = 0.50
CHARGE_EFFICIENCY        = 0.95
DISCHARGE_EFFICIENCY     = 0.95

# --- Diesel conversion factors (for the KPI tiles) ---
KWH_PER_LITRE_DIESEL    = 3.5    # electrical kWh a small genset makes per litre
CO2_KG_PER_LITRE_DIESEL = 2.68   # kg CO2 per litre of diesel burned

# --- Simulation ---
SIM_YEAR    = 2025
RANDOM_SEED = 42
