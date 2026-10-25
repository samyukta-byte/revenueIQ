
import pandas as pd

from src.data.build_transaction_investigation import (
    build_invoice_contributions,
    build_product_contributions,
    build_weekly_quality,
    load_sales,
)


def sample_sales():
    return pd.DataFrame(
        {
            "week_ending": pd.to_datetime(
                ["2011-12-11", "2011-12-11", "2011-12-11"]
            ),
            "invoice": ["1001", "1001", "1002"],
            "stockcode": ["A", "B", "A"],
            "description": ["Product A", "Product B", "Product A"],
            "quantity": [2, 3, 5],
            "price": [10.0, 20.0, 10.0],
            "line_value": [20.0, 60.0, 50.0],
            "customer_id": [101, 101, 102],
            "country": ["UK", "UK", "France"],
            "high_quantity_signal": [False, False, False],
            "high_value_signal": [False, False, False],
        }
    )


def test_product_contributions_sum_revenue():
    result = build_product_contributions(sample_sales())

    assert len(result) == 2

    product_a = result[result["stockcode"] == "A"].iloc[0]
    assert product_a["revenue"] == 70.0
    assert product_a["total_quantity"] == 7
    assert product_a["invoice_count"] == 2


def test_invoice_contributions_sum_revenue():
    result = build_invoice_contributions(sample_sales())

    assert len(result) == 2

    invoice_1001 = result[result["invoice"] == "1001"].iloc[0]
    assert invoice_1001["revenue"] == 80.0
    assert invoice_1001["line_count"] == 2
    assert invoice_1001["product_count"] == 2


def test_weekly_quality_totals():
    result = build_weekly_quality(sample_sales())

    assert len(result) == 1
    row = result.iloc[0]

    assert row["candidate_revenue"] == 130.0
    assert row["transaction_lines"] == 3
    assert row["invoice_count"] == 2
    assert row["product_count"] == 2


def test_load_sales_filters_invalid_sales(tmp_path):
    source = tmp_path / "sales.csv"

    pd.DataFrame(
        {
            "Invoice": ["1001", "1002", "1003"],
            "StockCode": ["A", "B", "C"],
            "Description": ["Valid", "Zero quantity", "Negative price"],
            "Quantity": [2, 0, 3],
            "InvoiceDate": [
                "2011-12-09",
                "2011-12-09",
                "2011-12-09",
            ],
            "Price": [10.0, 5.0, -2.0],
            "Customer ID": [101, 102, 103],
            "Country": ["UK", "UK", "UK"],
        }
    ).to_csv(source, index=False)

    result = load_sales(source)

    assert len(result) == 1
    assert result.iloc[0]["invoice"] == "1001"
    assert result.iloc[0]["line_value"] == 20.0
    assert result.iloc[0]["week_ending"] == pd.Timestamp("2011-12-11")
