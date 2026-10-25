
# RevenueIQ Intelligence 📈

**An explainable revenue analytics platform for forecasting, anomaly detection, and transaction-level investigation.**

RevenueIQ analyzes historical retail transactions, identifies unusual weekly revenue patterns, compares forecasting models, and presents the results through an interactive Streamlit dashboard.

## Features

- **Data cleaning:** Duplicate analysis, transaction filtering, and weekly revenue aggregation.
- **Revenue dashboard:** Interactive weekly trends and key metrics.
- **Anomaly detection:** Standard z-score and robust modified z-score comparison.
- **Transaction investigation:** Drill down into suspicious revenue spikes using product and invoice details.
- **Revenue forecasting:** Compare a training-mean baseline with exponential smoothing.
- **Rolling-origin validation:** Evaluate models across four chronological test windows.
- **Interactive visualization:** Explore results through Streamlit and Plotly.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Statsmodels · Plotly · Streamlit

## Model Evaluation

On the current dataset, rolling-origin validation produced these results:

| Model | MAE | RMSE |
|---|---:|---:|
| Training-mean baseline | 60,598.50 | 87,462.27 |
| Exponential smoothing | 55,948.22 | 74,320.60 |

Exponential smoothing reduced MAE by **7.67%** across 48 predictions from four chronological test windows.

These metrics describe this evaluation dataset and are not a guarantee of future performance.

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
python src/forecasting/generate_future_forecast.py
streamlit run app.py
```

Open the local URL shown in your terminal.

## Limitations

- Positive-sale candidate value is not verified net revenue.
- The current dataset contains 106 weekly observations.
- The rolling evaluation uses four non-overlapping test windows.
- The current forecasting model does not include seasonality or prediction intervals.
- Anomaly flags indicate unusual observations; they do not prove fraud or data errors.
- Transaction-level investigation currently uses a limited saved subset.

## Project Status

An exploratory data science portfolio project. Further work includes expanding transaction investigation, adding automated tests, and improving forecast uncertainty estimates.
