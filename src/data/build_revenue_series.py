
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/sales_candidates.csv")
OUTPUT_PATH = Path("data/processed/weekly_revenue.csv")

print("Loading sales candidates...")
df = pd.read_csv(INPUT_PATH, parse_dates=["invoice_date"])

# Recalculate line value explicitly.
df["line_value"] = df["quantity"] * df["price"]

# Aggregate sales into weekly totals, with weeks ending Sunday.
weekly = (
    df.set_index("invoice_date")["line_value"]
    .resample("W-SUN")
    .sum()
    .rename("revenue")
    .reset_index()
)

weekly = weekly.sort_values("invoice_date")

# Include weeks without recorded sales as zero.
full_weeks = pd.date_range(
    start=weekly["invoice_date"].min(),
    end=weekly["invoice_date"].max(),
    freq="W-SUN",
)

weekly = (
    weekly.set_index("invoice_date")
    .reindex(full_weeks, fill_value=0)
    .rename_axis("week_ending")
    .reset_index()
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
weekly.to_csv(OUTPUT_PATH, index=False)

print("\nWeekly revenue summary")
print("-" * 35)
print(f"Weeks: {len(weekly):,}")
print(f"First week ending: {weekly['week_ending'].min()}")
print(f"Last week ending: {weekly['week_ending'].max()}")
print(f"Mean weekly revenue: {weekly['revenue'].mean():,.2f}")
print(f"Median weekly revenue: {weekly['revenue'].median():,.2f}")
print(f"Minimum weekly revenue: {weekly['revenue'].min():,.2f}")
print(f"Maximum weekly revenue: {weekly['revenue'].max():,.2f}")
print(f"\nSaved to: {OUTPUT_PATH}")
