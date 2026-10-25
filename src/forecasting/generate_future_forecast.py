
from pathlib import Path

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "weekly_revenue.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "future_forecast.csv"

FORECAST_HORIZON = 12

print("Loading weekly revenue data...")
df = pd.read_csv(DATA_PATH, parse_dates=["week_ending"])
df = df.sort_values("week_ending").reset_index(drop=True)

y = df["revenue"].astype(float)

if len(y) < 20:
    raise ValueError("Not enough weekly observations to fit the forecast model.")

# Fit the model using all available historical observations.
model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal=None,
    initialization_method="estimated",
)

fitted_model = model.fit(optimized=True)
predictions = fitted_model.forecast(FORECAST_HORIZON)

last_week = df["week_ending"].max()
future_dates = pd.date_range(
    start=last_week + pd.Timedelta(days=7),
    periods=FORECAST_HORIZON,
    freq="7D",
)

forecast = pd.DataFrame({
    "week_ending": future_dates,
    "forecast_revenue": predictions.to_numpy(),
})

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
forecast.to_csv(OUTPUT_PATH, index=False)

print("\nFUTURE REVENUE FORECAST")
print("=" * 38)
print(f"Historical weeks used: {len(y)}")
print(f"Forecast horizon: {FORECAST_HORIZON} weeks")
print(f"Last historical week: {last_week.date()}")

print("\nForecast values:")
print(
    forecast.assign(
        forecast_revenue=forecast["forecast_revenue"].round(2)
    ).to_string(index=False)
)

print(f"\nSaved to: {OUTPUT_PATH}")
print(
    "\nThese are model estimates, not guaranteed results. "
    "No prediction intervals are included."
)
