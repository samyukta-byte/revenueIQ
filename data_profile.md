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
Quantity multiplied by unit price across all rows produced a gross transaction value of 19,287,250.57.

This is a preliminary figure, not validated net revenue. Returns, cancellations, invalid prices, duplicate records and other adjustments still need investigation.

## Next Steps
1. Investigate invoice types and cancellation records.
2. Determine defensible duplicate-handling rules.
3. Define revenue calculation and cleaning policies.
4. Build a reproducible cleaning pipeline.
5. Validate the cleaned data before forecasting or anomaly detection.

## Important Limitation
Missing customer IDs mean customer-level analysis will not cover every transaction. The dataset represents an online retailer, and detected anomalies are not automatically evidence of fraud.
