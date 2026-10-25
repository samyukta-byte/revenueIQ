
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/online_retail_II.xlsx")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading workbook...")
sheets = pd.read_excel(RAW_PATH, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

# Standardise column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[\s_]+", "_", regex=True)
)

# The original header is InvoiceDate, without a space
df = df.rename(columns={"invoicedate": "invoice_date"})

# Standardise text columns
for col in ["invoice", "stockcode", "description", "country"]:
    df[col] = df[col].astype("string").str.strip()

# Convert data types
df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

initial_rows = len(df)

# Flag records before deduplication
df["is_cancellation_invoice"] = df["invoice"].str.startswith(
    "C", na=False
)
df["customer_id_missing"] = df["customer_id"].isna()
df["line_value"] = df["quantity"] * df["price"]

# Provisional rule: remove exact duplicate rows, keep first occurrence
df = df.drop_duplicates().copy()

# Flag quality issues; don't silently discard records
df["invalid_date"] = df["invoice_date"].isna()
df["invalid_quantity"] = df["quantity"].isna() | (df["quantity"] == 0)
df["nonpositive_price"] = df["price"].isna() | (df["price"] <= 0)

# Create a candidate dataset for sales analysis
df["is_positive_sale_candidate"] = (
    ~df["is_cancellation_invoice"]
    & df["quantity"].gt(0)
    & df["price"].gt(0)
    & df["invoice_date"].notna()
)

# Save the complete deduplicated data
cleaned_path = OUTPUT_DIR / "transactions_deduplicated.csv"
df.to_csv(cleaned_path, index=False)

# Save positive-sale candidates separately
sales = df.loc[df["is_positive_sale_candidate"]].copy()
sales_path = OUTPUT_DIR / "sales_candidates.csv"
sales.to_csv(sales_path, index=False)

print("\n" + "=" * 45)
print("CLEANING SUMMARY")
print("=" * 45)
print(f"Initial rows: {initial_rows:,}")
print(f"Rows after deduplication: {len(df):,}")
print(f"Rows removed: {initial_rows - len(df):,}")
print(f"Cancellation invoice rows retained: {df['is_cancellation_invoice'].sum():,}")
print(f"Rows with missing customer IDs retained: {df['customer_id_missing'].sum():,}")
print(f"Positive-sale candidates: {len(sales):,}")
print(f"\nSaved: {cleaned_path}")
print(f"Saved: {sales_path}")
print("\nOriginal Excel file was not modified.")
