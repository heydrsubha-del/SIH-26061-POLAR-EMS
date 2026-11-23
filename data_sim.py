"""Step 1 - Synthetic data for a polar research station.

Real station SCADA data is not public, so we generate ONE physically-plausible
year of HOURLY data: ambient temperature, wind speed, available solar & wind
power, and electrical load. Everything is seeded, so the demo is 100%
reproducible (same numbers every run).

Sanity-check standalone:   python data_sim.py
"""
import numpy as np
import pandas as pd

import config as C


def _solar_elevation_sin(day_of_year, hour, latitude_deg):
    """sin(solar elevation angle). A value <= 0 means the sun is below the
    horizon (so no solar). This one formula gives us polar night in winter and
    the midnight sun in summer automatically."""
    phi = np.radians(latitude_deg)
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)
    hour_angle = np.radians(15.0 * (hour - 12.0))
    return (np.sin(phi) * np.sin(decl)
            + np.cos(phi) * np.cos(decl) * np.cos(hour_angle))


def _wind_power_kw(wind_ms, capacity_kw):
    """Turbine power curve: cut-in 3 m/s, rated at 12 m/s, cut-out 25 m/s."""
    v = np.asarray(wind_ms, dtype=float)
    p = np.zeros_like(v)
    ramp = (v >= 3) & (v < 12)
    p[ramp] = ((v[ramp] - 3) / (12 - 3)) ** 3      # smooth cubic ramp-up
    rated = (v >= 12) & (v <= 25)
    p[rated] = 1.0
    return p * capacity_kw


def generate_year(seed=C.RANDOM_SEED):
    """Return a DataFrame indexed by hour for one full year."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{C.SIM_YEAR}-01-01", periods=8760, freq="h")
    doy = idx.dayofyear.to_numpy()
    hod = idx.hour.to_numpy()

    # --- Ambient temperature (southern hemisphere: coldest around July) ---
    seasonal = -25 + 15 * np.cos(2 * np.pi * (doy - 200) / 365.0)
    diurnal = 3 * np.sin(2 * np.pi * (hod - 3) / 24.0)
    temp_c = seasonal + diurnal + rng.normal(0, 2, size=8760)

    # --- Available solar power ---
    sin_elev = _solar_elevation_sin(doy, hod, C.LATITUDE_DEG)
    clearness = np.clip(sin_elev, 0, None)
    cloud = rng.uniform(0.35, 1.0, size=8760)
    solar_kw = clearness * cloud * C.PV_CAPACITY_KW

    # --- Wind speed: AR(1) persistence + windier winters, then power curve ---
    mean_wind = 7 + 2 * np.cos(2 * np.pi * (doy - 200) / 365.0)
    wind = np.empty(8760)
    wind[0] = mean_wind[0]
    for t in range(1, 8760):
        wind[t] = mean_wind[t] + 0.85 * (wind[t - 1] - mean_wind[t - 1]) + rng.normal(0, 1.2)
    wind_ms = np.clip(wind, 0, 30)
    wind_kw = _wind_power_kw(wind_ms, C.WIND_CAPACITY_KW)

    # --- Electrical load (heating dominated: colder -> more load) ---
    base = 40.0
    heating = 1.4 * np.clip(18 - temp_c, 0, None)
    activity = 12 * np.clip(np.sin(2 * np.pi * (hod - 8) / 24.0), 0, None)
    load_kw = base + heating + activity + rng.normal(0, 3, size=8760)
    load_kw = np.clip(load_kw, 25, C.DIESEL_CAPACITY_KW - 5)   # stay below backstop

    return pd.DataFrame(
        {
            "temp_c": temp_c,
            "wind_ms": wind_ms,
            "solar_kw": solar_kw,
            "wind_kw": wind_kw,
            "load_kw": load_kw,
        },
        index=idx,
    )


if __name__ == "__main__":
    df = generate_year()
    print("One year of hourly data:", df.shape)
    print(df.describe().round(1))
    jul_noon = df.loc[f"{C.SIM_YEAR}-07-01 12:00"]
    jan_noon = df.loc[f"{C.SIM_YEAR}-01-01 12:00"]
    print(f"\nPolar-night check  -> Jul 1 noon solar: {jul_noon.solar_kw:.1f} kW (should be ~0)")
    print(f"Midnight-sun check -> Jan 1 noon solar: {jan_noon.solar_kw:.1f} kW (should be > 0)")
