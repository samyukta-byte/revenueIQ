
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "sales_candidates.csv"
PRODUCT_OUTPUT = DATA_DIR / "weekly_product_contributions.csv"
INVOICE_OUTPUT = DATA_DIR / "weekly_invoice_contributions.csv"
QUALITY_OUTPUT = DATA_DIR / "weekly_transaction_quality.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and common naming variations."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    aliases = {
        "invoicedate": "invoice_date",
        "invoice_date": "invoice_date",
        "customerid": "customer_id",
        "customer_id": "customer_id",
        "stock_code": "stockcode",
        "stockcode": "stockcode",
        "description": "description",
        "quantity": "quantity",
        "price": "price",
        "invoice": "invoice",
        "country": "country",
    }

    rename_map = {
        column: aliases[column]
        for column in df.columns
        if column in aliases and column != aliases[column]
    }

    return df.rename(columns=rename_map)


def load_sales(path: Path = INPUT_FILE) -> pd.DataFrame:
    """Load valid sale candidates and prepare investigation fields."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run src/data/clean_data.py first."
        )

    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)

    required = {
        "invoice",
        "stockcode",
        "description",
        "quantity",
        "invoice_date",
        "price",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found: {df.columns.tolist()}"
        )

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"], errors="coerce"
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(
        subset=[
            "invoice",
            "stockcode",
            "invoice_date",
            "quantity",
            "price",
        ]
    ).copy()

    # Only investigate positive-quantity, positive-price sales.
    df = df[(df["quantity"] > 0) & (df["price"] > 0)].copy()

    df["line_value"] = df["quantity"] * df["price"]
    df["week_ending"] = (
        df["invoice_date"]
        .dt.to_period("W-SUN")
        .dt.end_time
        .dt.normalize()
    )

    if "customer_id" not in df.columns:
        df["customer_id"] = pd.NA

    if "country" not in df.columns:
        df["country"] = pd.NA

    df["description"] = df["description"].fillna("Unknown description")
    df["stockcode"] = df["stockcode"].astype(str)
    df["invoice"] = df["invoice"].astype(str)

    # Percentile flags are investigation signals, not proof of errors.
    quantity_cutoff = df["quantity"].quantile(0.99)
    value_cutoff = df["line_value"].quantile(0.99)

    df["high_quantity_signal"] = df["quantity"] >= quantity_cutoff
    df["high_value_signal"] = df["line_value"] >= value_cutoff

    print(f"Loaded positive sale lines: {len(df):,}")
    print(f"99th percentile quantity: {quantity_cutoff:,.2f}")
    print(f"99th percentile line value: {value_cutoff:,.2f}")

    return df


def build_product_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each product's contribution to each week's revenue."""
    result = (
        df.groupby(
            ["week_ending", "stockcode", "description"],
            dropna=False,
            as_index=False,
        )
        .agg(
            revenue=("line_value", "sum"),
            total_quantity=("quantity", "sum"),
            line_count=("invoice", "size"),
            invoice_count=("invoice", "nunique"),
            customer_count=("customer_id", "nunique"),
            high_quantity_lines=("high_quantity_signal", "sum"),
            high_value_lines=("high_value_signal", "sum"),
        )
    )

    result["week_ending"] = pd.to_datetime(
        result["week_ending"]
    ).dt.strftime("%Y-%m-%d")

    return result.sort_values(
        ["week_ending", "revenue"],
        ascending=[True, False],
    )


def build_invoice_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each invoice's contribution to each week's revenue."""
    result = (
        df.groupby(
            ["week_ending", "invoice"],
            dropna=False,
            as_index=False,
        )
        .agg(
            revenue=("line_value", "sum"),
            total_quantity=("quantity", "sum"),
            line_count=("stockcode", "size"),
            product_count=("stockcode", "nunique"),
            customer_count=("customer_id", "nunique"),
            high_quantity_lines=("high_quantity_signal", "sum"),
            high_value_lines=("high_value_signal", "sum"),
        )
    )

    result["week_ending"] = pd.to_datetime(
        result["week_ending"]
    ).dt.strftime("%Y-%m-%d")

    return result.sort_values(
        ["week_ending", "revenue"],
        ascending=[True, False],
    )


def build_weekly_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize weekly transaction signals without declaring errors."""
    result = (
        df.groupby("week_ending", as_index=False)
        .agg(
            candidate_revenue=("line_value", "sum"),
            transaction_lines=("invoice", "size"),
            invoice_count=("invoice", "nunique"),
            product_count=("stockcode", "nunique"),
            high_quantity_lines=("high_quantity_signal", "sum"),
            high_value_lines=("high_value_signal", "sum"),
            largest_quantity=("quantity", "max"),
            largest_line_value=("line_value", "max"),
        )
    )

    result["week_ending"] = pd.to_datetime(
        result["week_ending"]
    ).dt.strftime("%Y-%m-%d")

    return result.sort_values("week_ending")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sales = load_sales()

    products = build_product_contributions(sales)
    invoices = build_invoice_contributions(sales)
    quality = build_weekly_quality(sales)

    products.to_csv(PRODUCT_OUTPUT, index=False)
    invoices.to_csv(INVOICE_OUTPUT, index=False)
    quality.to_csv(QUALITY_OUTPUT, index=False)

    print("\nInvestigation files created successfully.")
    print(f"Weekly product rows: {len(products):,}")
    print(f"Weekly invoice rows: {len(invoices):,}")
    print(f"Weekly quality summaries: {len(quality):,}")
    print(f"\nProducts: {PRODUCT_OUTPUT}")
    print(f"Invoices: {INVOICE_OUTPUT}")
    print(f"Quality:  {QUALITY_OUTPUT}")


if __name__ == "__main__":
    main()
