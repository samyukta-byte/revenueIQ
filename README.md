
# RevenueIQ Intelligence 📈

**An explainable revenue analytics platform for forecasting, anomaly detection, and transaction-level investigation.**

RevenueIQ analyzes historical retail transactions, identifies unusual weekly revenue patterns, compares forecasting models, and presents the results through an interactive Streamlit dashboard.

## Features

- **Data cleaning:** Duplicate analysis, transaction filtering, and weekly revenue aggregation.
- **Revenue dashboard:** Interactive weekly trends and key metrics.
- **Anomaly detection:** Standard z-score and robust modified z-score comparison.
- **Transaction investigation:** Investigate unusual weekly revenue using product, invoice, and transaction-quality summaries.
- **Revenue forecasting:** Compare seasonal-naive, naive, training-mean, and exponential-smoothing models.
- **Rolling-origin validation:** Compare forecasting methods across four chronological test windows, with 48 total predictions per model.
- **Data validation:** Check transaction cleaning, weekly aggregation, and forecast reconciliation.
- **Interactive visualization:** Explore results through Streamlit and Plotly.

## Tech Stack

Python | Pandas | NumPy | Scikit-learn | Statsmodels | Plotly | Streamlit

## Model Evaluation

Rolling-origin validation across four chronological test windows produced the following mean fold errors:

| Model | Mean fold MAE | Mean fold RMSE |
|---|---:|---:|
| Seasonal naive | 34,883.89 | 45,898.23 |
| Naive | 50,976.29 | 63,137.07 |
| Exponential smoothing | 55,948.22 | 67,980.63 |
| Training-mean baseline | 60,598.50 | 69,223.08 |

Seasonal naive performed best among these four methods on this evaluation setup. It forecasts each week using the value from 52 weeks earlier. These metrics describe this historical dataset and are not a guarantee of future performance.

## Dataset

This project uses the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii), containing historical retail transactions.

The raw workbook and generated datasets are excluded from Git. Run the cleaning and processing scripts locally after obtaining the source dataset.

## Run Locally

Install the required packages:

```bash
pip install pandas numpy scikit-learn statsmodels streamlit plotly openpyxl matplotlib
```

Place `online_retail_II.xlsx` in `data/raw/`, then run:

```bash
python src/data/clean_data.py
python src/data/build_revenue_series.py
python src/anomaly/compare_anomaly_methods.py
python src/forecasting/forecast_revenue.py
python src/forecasting/compare_forecast_models.py
python src/forecasting/generate_future_forecast.py
python src/data/build_transaction_investigation.py
python src/data/validate_revenue_pipeline.py
python src/forecasting/validate_forecasting_results.py
python -m pytest tests/ -v
streamlit run app.py
```

Open the local URL shown in your terminal.

## Limitations

- Positive-sale candidate value is not verified net revenue.
- The current dataset contains 106 weekly observations.
- The rolling evaluation uses four non-overlapping test windows.
- The seasonal-naive forecast repeats values from 52 weeks earlier and does not provide prediction intervals.
- Anomaly flags and unusually large transactions are investigation signals, not proof of fraud or errors.

## Project Status

An exploratory data science portfolio project. Further work includes reconciling returns and cancellations into net revenue and improving forecast uncertainty estimates.
