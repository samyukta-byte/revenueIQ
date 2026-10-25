
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

SALES_FILE = DATA / "sales_candidates.csv"
WEEKLY_FILE = DATA / "weekly_revenue.csv"


def main():
    print("Loading transaction data...")

    sales = pd.read_csv(SALES_FILE, low_memory=False)
    weekly_saved = pd.read_csv(WEEKLY_FILE)

    sales["invoice_date"] = pd.to_datetime(
        sales["invoice_date"], errors="coerce"
    )
    sales["quantity"] = pd.to_numeric(
        sales["quantity"], errors="coerce"
    )
    sales["price"] = pd.to_numeric(
        sales["price"], errors="coerce"
    )

    # Check the assumptions used by the revenue pipeline.
    invalid_dates = int(sales["invoice_date"].isna().sum())
    invalid_quantity = int(sales["quantity"].isna().sum())
    invalid_price = int(sales["price"].isna().sum())

    nonpositive_quantity = int((sales["quantity"] <= 0).sum())
    nonpositive_price = int((sales["price"] <= 0).sum())

    sales["line_value"] = sales["quantity"] * sales["price"]

    print("\n1. TRANSACTION DATA CHECKS")
    print("-" * 40)
    print(f"Transaction rows: {len(sales):,}")
    print(f"Invalid dates: {invalid_dates:,}")
    print(f"Invalid quantities: {invalid_quantity:,}")
    print(f"Invalid prices: {invalid_price:,}")
    print(f"Nonpositive quantities: {nonpositive_quantity:,}")
    print(f"Nonpositive prices: {nonpositive_price:,}")

    # Rebuild the weekly series independently.
    valid = sales.dropna(
        subset=["invoice_date", "quantity", "price"]
    ).copy()

    valid = valid[
        (valid["quantity"] > 0) & (valid["price"] > 0)
    ]

    expected = (
        valid.set_index("invoice_date")["line_value"]
        .resample("W-SUN")
        .sum()
        .rename("expected_revenue")
    )

    full_weeks = pd.date_range(
        start=expected.index.min(),
        end=expected.index.max(),
        freq="W-SUN",
    )

    expected = expected.reindex(full_weeks, fill_value=0)
    expected.index.name = "week_ending"

    saved = weekly_saved.copy()
    saved["week_ending"] = pd.to_datetime(
        saved["week_ending"], errors="coerce"
    ).dt.normalize()

    saved["revenue"] = pd.to_numeric(
        saved["revenue"], errors="coerce"
    )

    duplicate_dates = int(saved["week_ending"].duplicated().sum())

    comparison = pd.DataFrame({
        "expected_revenue": expected,
    }).rename_axis("week_ending").reset_index()

    comparison = comparison.merge(
        saved[["week_ending", "revenue"]],
        on="week_ending",
        how="outer",
        indicator=True,
    )

    missing_weeks = int(
        (comparison["_merge"] != "both").sum()
    )

    comparison["difference"] = (
        comparison["revenue"] - comparison["expected_revenue"]
    )

    matched = comparison["_merge"].eq("both")
    differences = comparison.loc[matched, "difference"].abs()

    mismatched_weeks = int(
        (~np.isclose(
            comparison.loc[matched, "revenue"],
            comparison.loc[matched, "expected_revenue"],
            rtol=1e-9,
            atol=0.01,
        )).sum()
    )

    print("\n2. WEEKLY REVENUE RECONCILIATION")
    print("-" * 40)
    print(f"Expected weekly observations: {len(expected):,}")
    print(f"Saved weekly observations: {len(saved):,}")
    print(f"Duplicate saved week dates: {duplicate_dates:,}")
    print(f"Missing or extra week dates: {missing_weeks:,}")
    print(f"Weeks with revenue mismatches: {mismatched_weeks:,}")

    print(
        f"Expected total candidate value: "
        f"{expected.sum():,.2f}"
    )
    print(
        f"Saved total candidate value: "
        f"{saved['revenue'].sum():,.2f}"
    )
    print(
        f"Absolute total difference: "
        f"{abs(expected.sum() - saved['revenue'].sum()):,.6f}"
    )

    if not differences.empty:
        print(
            f"Largest weekly absolute difference: "
            f"{differences.max():,.6f}"
        )

    if missing_weeks or duplicate_dates or mismatched_weeks:
        print("\nRESULT: CHECK FAILED — investigate the differences.")
        raise SystemExit(1)

    print("\nRESULT: PASS — weekly revenue reconciles with transactions.")


if __name__ == "__main__":
    main()
