"""Step 2 - AI load forecasting.

Train a gradient-boosted model to predict electrical load from calendar +
weather + recent-history features. We hold out the LAST 30 days as an unseen
test window so we can honestly report accuracy. This is the "AI" the problem
statement asks for - a real trained model, not a hand-tuned rule.

Sanity-check standalone:   python forecast.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

TEST_HOURS = 24 * 30   # forecast (and score) the final 30 days


def build_features(df):
    """Calendar (cyclical), weather, and lagged-load features.
    All are values that would be known at prediction time."""
    X = pd.DataFrame(index=df.index)
    hod = df.index.hour
    doy = df.index.dayofyear
    X["hour_sin"] = np.sin(2 * np.pi * hod / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hod / 24)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    X["temp_c"] = df["temp_c"].to_numpy()
    X["lag_24"] = df["load_kw"].shift(24)      # load 1 day ago
    X["lag_168"] = df["load_kw"].shift(168)    # load 1 week ago
    X["roll_24"] = df["load_kw"].shift(1).rolling(24).mean()
    return X


def forecast_load(df):
    """Fit on the training period, predict the held-out test window.
    Returns (result_df, mae) where result_df has load_actual & load_pred."""
    X = build_features(df)
    y = df["load_kw"]

    valid = X.dropna().index          # drop early rows that lack lag history
    X, y = X.loc[valid], y.loc[valid]

    split = len(X) - TEST_HOURS
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    result = pd.DataFrame(
        {"load_actual": y_test.to_numpy(), "load_pred": pred},
        index=X_test.index,
    )
    return result, mean_absolute_error(y_test, pred)


if __name__ == "__main__":
    from data_sim import generate_year

    res, mae = forecast_load(generate_year())
    mape = (np.abs(res.load_actual - res.load_pred) / res.load_actual).mean() * 100
    print(f"Test window: {res.index[0]}  ->  {res.index[-1]}  ({len(res)} hours)")
    print(f"MAE  = {mae:.2f} kW")
    print(f"MAPE = {mape:.1f} %   (lower is better; expect a few %)")
