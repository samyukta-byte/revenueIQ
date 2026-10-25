
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/weekly_revenue.csv")
OUTPUT_PATH = Path("data/processed/revenue_anomalies.csv")

df = pd.read_csv(INPUT_PATH, parse_dates=["week_ending"])
df = df.sort_values("week_ending").reset_index(drop=True)

# Initial baseline: all weeks, used for exploratory anomaly flags.
mean = df["revenue"].mean()
std = df["revenue"].std()

df["z_score"] = (df["revenue"] - mean) / std

# Flag values more than 2 standard deviations from the mean.
df["anomaly_flag"] = df["z_score"].abs() >= 2

df["anomaly_type"] = "normal"
df.loc[df["z_score"] >= 2, "anomaly_type"] = "unusually_high"
df.loc[df["z_score"] <= -2, "anomaly_type"] = "unusually_low"

df["investigation_status"] = "not_reviewed"
df.loc[df["anomaly_flag"], "investigation_status"] = "needs_investigation"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

anomalies = df[df["anomaly_flag"]]

print("\nREVENUE ANOMALY DETECTION")
print("=" * 40)
print(f"Weeks analyzed: {len(df)}")
print(f"Baseline mean: {mean:,.2f}")
print(f"Baseline standard deviation: {std:,.2f}")
print(f"Anomalies flagged: {len(anomalies)}")

print("\nFlagged weeks:")
print(
    anomalies[
        ["week_ending", "revenue", "z_score", "anomaly_type"]
    ].round({"revenue": 2, "z_score": 2}).to_string(index=False)
)

print(f"\nSaved results to: {OUTPUT_PATH}")
print("Flags indicate unusual values, not confirmed errors or fraud.")
