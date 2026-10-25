
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/weekly_revenue.csv")
OUTPUT_PATH = Path("data/processed/anomaly_method_comparison.csv")

df = pd.read_csv(INPUT_PATH, parse_dates=["week_ending"])
df = df.sort_values("week_ending").reset_index(drop=True)

values = df["revenue"]

# Method 1: Mean and standard deviation
mean = values.mean()
std = values.std()

df["z_score"] = (values - mean) / std
df["z_score_flag"] = df["z_score"].abs() >= 2

# Method 2: Robust modified z-score using median and MAD
median = values.median()
mad = (values - median).abs().median()

if mad > 0:
    df["modified_z_score"] = (
        0.6745 * (values - median) / mad
    )
    df["robust_flag"] = df["modified_z_score"].abs() > 3.5
else:
    df["modified_z_score"] = float("nan")
    df["robust_flag"] = False

df["agreement"] = df["z_score_flag"] & df["robust_flag"]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nANOMALY METHOD COMPARISON")
print("=" * 42)
print(f"Weeks analyzed: {len(df)}")
print(f"Mean baseline: {mean:,.2f}")
print(f"Median baseline: {median:,.2f}")
print(f"Median absolute deviation: {mad:,.2f}")

print(f"\nStandard-deviation method flags: {df['z_score_flag'].sum()}")
print(f"Robust method flags: {df['robust_flag'].sum()}")
print(f"Flagged by both methods: {df['agreement'].sum()}")

print("\nRobust-method flagged weeks:")
print(
    df.loc[
        df["robust_flag"],
        ["week_ending", "revenue", "z_score", "modified_z_score"]
    ].round({"revenue": 2, "z_score": 2, "modified_z_score": 2})
    .to_string(index=False)
)

print(f"\nComparison saved to: {OUTPUT_PATH}")
print("\nThese are exploratory flags, not confirmed errors or fraud.")
