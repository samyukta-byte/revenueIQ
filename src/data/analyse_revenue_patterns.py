
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/weekly_revenue.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["week_ending"])
df = df.sort_values("week_ending").reset_index(drop=True)

# Calendar features
df["year"] = df["week_ending"].dt.year
df["month"] = df["week_ending"].dt.month

print("\nREVENUE PATTERN ANALYSIS")
print("=" * 45)

print("\nAverage weekly value by month:")
monthly = (
    df.groupby("month")["revenue"]
    .agg(["mean", "median", "count"])
    .round(2)
)
print(monthly.to_string())

print("\nAverage weekly value by year:")
yearly = (
    df.groupby("year")["revenue"]
    .agg(["mean", "median", "count"])
    .round(2)
)
print(yearly.to_string())

# Identify unusually high/low weeks using a simple baseline.
mean = df["revenue"].mean()
std = df["revenue"].std()

df["z_score"] = (df["revenue"] - mean) / std

print("\nTop 10 unusually high weeks:")
print(
    df.nlargest(10, "z_score")[
        ["week_ending", "revenue", "z_score"]
    ].round(2).to_string(index=False)
)

print("\nTop 10 unusually low weeks:")
print(
    df.nsmallest(10, "z_score")[
        ["week_ending", "revenue", "z_score"]
    ].round(2).to_string(index=False)
)

print("\nNote: These are statistical flags for investigation,")
print("not proof of fraud or business problems.")
