
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/online_retail_II.xlsx")

print("Loading both sheets...")
sheets = pd.read_excel(DATA_PATH, sheet_name=None)

frames = []

for sheet_name, sheet_df in sheets.items():
    print(f"{sheet_name}: {len(sheet_df):,} rows")
    frames.append(sheet_df)

df = pd.concat(frames, ignore_index=True)

print("\n" + "=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

# Missing values
print("\nMissing values by column:")
print(df.isna().sum().sort_values(ascending=False))

# Duplicate records
print(f"\nExact duplicate rows: {df.duplicated().sum():,}")

# Check transaction characteristics
print(f"Rows with quantity <= 0: {(df['Quantity'] <= 0).sum():,}")
print(f"Rows with price <= 0: {(df['Price'] <= 0).sum():,}")

# Cancellation invoices often have an invoice number starting with C.
invoice = df["Invoice"].astype("string").str.strip()
cancelled = invoice.str.startswith("C", na=False)

print(f"Cancellation invoice rows: {cancelled.sum():,}")

# Revenue is calculated from quantity multiplied by unit price.
df["Revenue"] = df["Quantity"] * df["Price"]

print(f"\nGross transaction value: {df['Revenue'].sum():,.2f}")

print("\nTop 10 countries by row count:")
print(df["Country"].value_counts().head(10))

print("\nData quality inspection complete.")
