
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "weekly_revenue.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "future_forecast.csv"

FORECAST_HORIZON = 12
SEASON_LENGTH = 52

print("Loading weekly revenue data...")
df = pd.read_csv(DATA_PATH, parse_dates=["week_ending"])
df = df.sort_values("week_ending").reset_index(drop=True)

if df["week_ending"].isna().any():
    raise ValueError("Historical data contains invalid week-ending dates.")

if df["revenue"].isna().any():
    raise ValueError("Historical revenue contains missing values.")

y = df["revenue"].astype(float)

if len(y) < SEASON_LENGTH:
    raise ValueError(
        f"Seasonal naive forecasting requires at least "
        f"{SEASON_LENGTH} weeks of historical data."
    )

if not y.index.is_unique:
    raise ValueError("Historical observations must have unique row indices.")

# Seasonal naive: repeat the revenue pattern from the last 52 weeks.
# For a 12-week forecast, use the final 12 observations from that season.
predictions = y.iloc[-SEASON_LENGTH:-SEASON_LENGTH + FORECAST_HORIZON].to_numpy()

last_week = df["week_ending"].max()
future_dates = pd.date_range(
    start=last_week + pd.Timedelta(days=7),
    periods=FORECAST_HORIZON,
    freq="7D",
)

forecast = pd.DataFrame({
    "week_ending": future_dates,
    "forecast_revenue": predictions,
})

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
forecast.to_csv(OUTPUT_PATH, index=False)

print("\nFUTURE REVENUE FORECAST")
print("=" * 38)
print("Selected model: Seasonal naive (52-week season)")
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
    "\nThese estimates repeat the corresponding weeks from the previous "
    "year. They assume the annual pattern persists and are not guaranteed "
    "results. No prediction intervals are included."
)