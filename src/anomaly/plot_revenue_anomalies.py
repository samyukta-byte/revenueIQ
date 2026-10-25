
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = Path("data/processed/anomaly_method_comparison.csv")
OUTPUT_PATH = Path("data/processed/revenue_anomalies_plot.png")

df = pd.read_csv(INPUT_PATH, parse_dates=["week_ending"])
df = df.sort_values("week_ending")

normal = df[~df["robust_flag"].astype(bool)]
anomalies = df[df["robust_flag"].astype(bool)]

plt.figure(figsize=(13, 6))

plt.plot(
    df["week_ending"],
    df["revenue"],
    label="Weekly candidate value",
    linewidth=1.8,
)

plt.scatter(
    anomalies["week_ending"],
    anomalies["revenue"],
    label="Robust-method flags",
    marker="o",
    s=65,
)

plt.title("RevenueIQ: Weekly Revenue Anomaly Detection")
plt.xlabel("Week ending")
plt.ylabel("Candidate value (dataset currency units)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=160)
plt.close()

print("Anomaly chart created successfully.")
print(f"Robust-method flags plotted: {len(anomalies)}")
print(f"Chart saved to: {OUTPUT_PATH}")
