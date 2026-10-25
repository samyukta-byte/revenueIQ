
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/online_retail_II.xlsx")

print("Loading workbook...")
sheets = pd.read_excel(DATA_PATH, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

columns = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country"
]

# Mark only extra occurrences; the first copy is retained.
extra_copy_mask = df.duplicated(subset=columns, keep="first")
extra_copies = df.loc[extra_copy_mask].copy()

# Rows that belong to groups containing at least one duplicate.
duplicate_group_mask = df.duplicated(subset=columns, keep=False)

invoice_text = df["Invoice"].astype("string").str.strip()
cancelled = invoice_text.str.startswith("C", na=False)
missing_customer = df["Customer ID"].isna()

df["RowValue"] = df["Quantity"] * df["Price"]

print("\n" + "=" * 55)
print("DUPLICATE COUNTS")
print("=" * 55)
print(f"Total rows: {len(df):,}")
print(f"Rows in duplicate groups: {duplicate_group_mask.sum():,}")
print(f"Extra copies after keeping first: {extra_copy_mask.sum():,}")

print("\n" + "=" * 55)
print("EXTRA COPIES BY CATEGORY")
print("=" * 55)

extra_indices = df.index[extra_copy_mask]

print(f"Extra copies with cancellation invoice: {cancelled.loc[extra_indices].sum():,}")
print(f"Extra copies with missing customer ID: {missing_customer.loc[extra_indices].sum():,}")

print("\n" + "=" * 55)
print("VALUE IMPACT")
print("=" * 55)

original_value = df["RowValue"].sum()
extra_copy_value = extra_copies["Quantity"].mul(extra_copies["Price"]).sum()

print(f"Original gross transaction value: {original_value:,.2f}")
print(f"Value associated with extra copies: {extra_copy_value:,.2f}")
print(f"Value after diagnostic deduplication: {original_value - extra_copy_value:,.2f}")

print("\n" + "=" * 55)
print("EXTRA COPIES BY INVOICE TYPE")
print("=" * 55)

extra_invoice_text = extra_copies["Invoice"].astype("string").str.strip()
extra_cancelled = extra_invoice_text.str.startswith("C", na=False)

print("Cancellation invoice extra copies:", int(extra_cancelled.sum()))
print("Other invoice extra copies:", int((~extra_cancelled).sum()))

print("\nDiagnostic only: no source data was modified.")
