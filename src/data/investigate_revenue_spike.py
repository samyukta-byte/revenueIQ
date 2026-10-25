
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/spike_investigation.csv")

df = pd.read_csv(
    DATA_PATH,
    dtype={"invoice": "string", "stockcode": "string"},
    low_memory=False,
    parse_dates=["invoice_date"],
)

df["line_value"] = df["quantity"] * df["price"]

week_total = df["line_value"].sum()

suspect_mask = (
    df["invoice"].astype("string").str.strip().eq("581483")
    & df["stockcode"].astype("string").str.strip().eq("23843")
)

suspect_value = df.loc[suspect_mask, "line_value"].sum()
adjusted_total = week_total - suspect_value

print("\nSPIKE IMPACT ANALYSIS")
print("=" * 40)
print(f"Original weekly candidate value: {week_total:,.2f}")
print(f"Suspect line value: {suspect_value:,.2f}")
print(f"Share of weekly value: {suspect_value / week_total:.1%}")
print(f"Value excluding suspect line: {adjusted_total:,.2f}")

# Compare with the overall weekly baseline.
weekly = pd.read_csv(
    "data/processed/weekly_revenue.csv",
    parse_dates=["week_ending"],
)

mean = weekly["revenue"].mean()
std = weekly["revenue"].std()

print(f"\nOverall weekly mean: {mean:,.2f}")
print(f"Overall weekly standard deviation: {std:,.2f}")
print(f"Original week's z-score: {(week_total - mean) / std:.2f}")
print(f"Adjusted diagnostic z-score: {(adjusted_total - mean) / std:.2f}")

print("\nThis is a diagnostic comparison, not a corrected revenue figure.")
print("Validate the record before deciding whether to exclude it.")
