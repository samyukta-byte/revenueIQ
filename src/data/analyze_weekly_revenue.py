
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/weekly_revenue.csv")
OUTPUT_DIR = Path("data/processed")

weekly = pd.read_csv(DATA_PATH, parse_dates=["week_ending"])
weekly = weekly.sort_values("week_ending").reset_index(drop=True)

print("\nWEEKLY REVENUE QUALITY CHECK")
print("=" * 40)
print(f"Number of weeks: {len(weekly)}")
print(f"Missing revenue values: {weekly['revenue'].isna().sum()}")
print(f"Zero-revenue weeks: {(weekly['revenue'] == 0).sum()}")
print(f"Negative-revenue weeks: {(weekly['revenue'] < 0).sum()}")
print(f"Duplicate week dates: {weekly['week_ending'].duplicated().sum()}")

print("\nLowest 10 weeks:")
print(weekly.nsmallest(10, "revenue").to_string(index=False))

print("\nHighest 10 weeks:")
print(weekly.nlargest(10, "revenue").to_string(index=False))

plt.figure(figsize=(12, 5))
plt.plot(weekly["week_ending"], weekly["revenue"], linewidth=1.8)
plt.title("Weekly Positive-Sale Candidate Value")
plt.xlabel("Week ending")
plt.ylabel("Value (dataset currency units)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = OUTPUT_DIR / "weekly_revenue_trend.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print(f"\nChart saved to: {chart_path}")
