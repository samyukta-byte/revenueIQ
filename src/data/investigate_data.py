
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/online_retail_II.xlsx")

print("Loading workbook...")
sheets = pd.read_excel(DATA_PATH, sheet_name=None)

df = pd.concat(sheets.values(), ignore_index=True)

# Create a temporary invoice text column for inspection.
invoice_text = df["Invoice"].astype("string").str.strip()
is_cancellation = invoice_text.str.startswith("C", na=False)

print("\n" + "=" * 55)
print("1. INVOICE TYPES")
print("=" * 55)

print("Cancellation invoice rows:", int(is_cancellation.sum()))
print("Other invoice rows:", int((~is_cancellation).sum()))

print("\nSample cancellation records:")
print(
    df.loc[
        is_cancellation,
        ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 55)
print("2. NON-POSITIVE QUANTITIES")
print("=" * 55)

bad_quantity = df["Quantity"] <= 0

print("Rows with quantity <= 0:", int(bad_quantity.sum()))
print(
    df.loc[
        bad_quantity,
        ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]
    ].head(15).to_string(index=False)
)

print("\n" + "=" * 55)
print("3. NON-POSITIVE PRICES")
print("=" * 55)

bad_price = df["Price"] <= 0

print("Rows with price <= 0:", int(bad_price.sum()))
print(
    df.loc[
        bad_price,
        ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]
    ].head(15).to_string(index=False)
)

print("\n" + "=" * 55)
print("4. EXACT DUPLICATES")
print("=" * 55)

duplicate_mask = df.duplicated(keep=False)

print("Rows belonging to duplicate groups:", int(duplicate_mask.sum()))
print(
    df.loc[duplicate_mask]
    .sort_values(["Invoice", "StockCode", "InvoiceDate"])
    .head(20)
    .to_string(index=False)
)

print("\nInvestigation complete. No data was modified or deleted.")

print("\n" + "=" * 55)
print("5. DUPLICATE IMPACT ANALYSIS")
print("=" * 55)

original_rows = len(df)
original_revenue = (df["Quantity"] * df["Price"]).sum()

# Diagnostic copy only: original df remains unchanged.
deduplicated_df = df.drop_duplicates()

deduplicated_rows = len(deduplicated_df)
deduplicated_revenue = (
    deduplicated_df["Quantity"] * deduplicated_df["Price"]
).sum()

print(f"Original rows: {original_rows:,}")
print(f"Rows after exact deduplication: {deduplicated_rows:,}")
print(f"Rows removed in diagnostic copy: {original_rows - deduplicated_rows:,}")

print(f"\nOriginal gross transaction value: {original_revenue:,.2f}")
print(f"Value after exact deduplication: {deduplicated_revenue:,.2f}")
print(f"Difference: {original_revenue - deduplicated_revenue:,.2f}")

print("\nThis is a diagnostic comparison, not a final cleaning decision.")
