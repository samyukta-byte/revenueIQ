# RevenueIQ Intelligence — Initial Data Profile

## Dataset
- Source: UCI Online Retail II
- File: `data/raw/online_retail_II.xlsx`
- Total rows: 1,067,371
- Columns: 8
- Date range: 1 December 2009 – 9 December 2011

## Initial Data Quality Findings
- Missing Customer IDs: 243,007
- Missing descriptions: 4,382
- Exact duplicate rows: 34,335
- Rows with quantity <= 0: 22,950
- Rows with price <= 0: 6,207
- Cancellation invoice rows reported by initial inspection: 19,494

## Preliminary Revenue Calculation
The initial gross transaction value was 19,287,250.57, calculated by multiplying quantity by unit price across all original rows.

This preliminary figure uses the uncleaned dataset and is not directly comparable to the current weekly candidate sales series. The current pipeline removes exact duplicate rows and includes only rows with positive quantities and prices.

## Validated Pipeline Checks
The current reproducible pipeline processes 1,007,913 positive-sale rows into 106 weekly observations. The weekly candidate sales values sum to 20,476,260.45. Automated validation confirms that the saved weekly series matches the transaction-level aggregation.

These figures represent candidate sales value, not reconciled net revenue. Returns, cancellations and other accounting adjustments still require investigation before interpreting the values as verified net revenue.

## Next Steps
1. Investigate returns and cancellation records to establish net revenue.
2. Review unusually large transactions and invoice contributions.
3. Continue comparing forecasting methods using rolling validation.
4. Monitor data quality and anomaly signals as investigation priorities.

## Important Limitation
Missing customer IDs mean customer-level analysis will not cover every transaction. The dataset represents an online retailer, and detected anomalies are not automatically evidence of fraud.
