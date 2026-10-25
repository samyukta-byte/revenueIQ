from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

WEEKLY_FILE = DATA / "weekly_revenue.csv"
COMPARISON_FILE = DATA / "forecast_model_comparison.csv"
FUTURE_FILE = DATA / "future_forecast.csv"

EXPECTED_FORECAST_PERIODS = 12
EXPECTED_SEASONAL_MODEL = "Seasonal naive"


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    return (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(errors ** 2))),
    )


def main():
    print("FORECAST VALIDATION")
    print("=" * 48)

    # ---------------- Historical weekly data ----------------
    weekly = pd.read_csv(WEEKLY_FILE)
    required_weekly = {"week_ending", "revenue"}
    missing_weekly = required_weekly - set(weekly.columns)
    if missing_weekly:
        raise ValueError(f"Missing weekly-data columns: {sorted(missing_weekly)}")

    weekly["week_ending"] = pd.to_datetime(
        weekly["week_ending"], errors="coerce"
    )
    weekly["revenue"] = pd.to_numeric(weekly["revenue"], errors="coerce")

    if weekly.empty:
        raise ValueError("Weekly revenue data is empty.")
    if weekly["week_ending"].isna().any():
        raise ValueError("Invalid dates found in weekly revenue data.")
    if weekly["revenue"].isna().any():
        raise ValueError("Missing or invalid revenue values in weekly data.")
    if not np.isfinite(weekly["revenue"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite revenue values found in weekly data.")
    if weekly["week_ending"].duplicated().any():
        raise ValueError("Duplicate dates found in weekly revenue data.")

    weekly = weekly.sort_values("week_ending").reset_index(drop=True)

    # ---------------- Four-model rolling comparison ----------------
    comparison = pd.read_csv(COMPARISON_FILE)
    required_comparison = {
        "fold", "train_weeks", "test_weeks", "model", "mae", "rmse"
    }
    missing_comparison = required_comparison - set(comparison.columns)
    if missing_comparison:
        raise ValueError(
            f"Missing model-comparison columns: {sorted(missing_comparison)}"
        )
    if comparison.empty:
        raise ValueError("Model-comparison results are empty.")

    comparison["fold"] = pd.to_numeric(comparison["fold"], errors="coerce")
    comparison["train_weeks"] = pd.to_numeric(
        comparison["train_weeks"], errors="coerce"
    )
    comparison["test_weeks"] = pd.to_numeric(
        comparison["test_weeks"], errors="coerce"
    )
    comparison["mae"] = pd.to_numeric(comparison["mae"], errors="coerce")
    comparison["rmse"] = pd.to_numeric(comparison["rmse"], errors="coerce")

    numeric_columns = ["fold", "train_weeks", "test_weeks", "mae", "rmse"]
    if comparison[numeric_columns].isna().any().any():
        raise ValueError("Missing or invalid values in model-comparison results.")
    if not np.isfinite(
        comparison[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite values in model-comparison results.")
    if (comparison[["mae", "rmse", "test_weeks"]] < 0).any().any():
        raise ValueError("Negative error metrics or test-week counts detected.")

    comparison["fold"] = comparison["fold"].astype(int)
    if comparison.duplicated(["fold", "model"]).any():
        raise ValueError("Duplicate model result found within a validation fold.")

    fold_ids = sorted(comparison["fold"].unique())
    model_names = sorted(comparison["model"].astype(str).unique())
    if len(fold_ids) < 2:
        raise ValueError("At least two validation folds are required.")

    fold_sets = comparison.groupby("model")["fold"].apply(set)
    shared_folds = set(fold_ids)
    comparable_models = [
        model for model in model_names
        if fold_sets.get(model, set()) == shared_folds
    ]
    if len(comparable_models) < 2:
        raise ValueError(
            "Fewer than two models have results for every shared validation fold."
        )

    comparison = comparison[
        comparison["model"].isin(comparable_models)
    ].copy()

    summary = (
        comparison.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            total_test_windows=("test_weeks", "sum"),
            mean_fold_mae=("mae", "mean"),
            mean_fold_rmse=("rmse", "mean"),
        )
        .sort_values("mean_fold_mae")
        .reset_index(drop=True)
    )

    best_model = str(summary.iloc[0]["model"])
    seasonal_rows = summary[
        summary["model"].str.casefold() == EXPECTED_SEASONAL_MODEL.casefold()
    ]
    if seasonal_rows.empty:
        raise ValueError(
            f"Expected '{EXPECTED_SEASONAL_MODEL}' model not found in comparison."
        )

    seasonal = seasonal_rows.iloc[0]

    print(f"Historical weeks: {len(weekly):,}")
    print(f"Validation folds: {len(fold_ids)}")
    print(f"Models compared on all folds: {len(comparable_models)}")
    print("\nMEAN FOLD MODEL COMPARISON")
    print("-" * 72)
    print(
        summary.rename(columns={
            "model": "Model",
            "folds": "Folds",
            "total_test_windows": "Test weeks summed across folds",
            "mean_fold_mae": "Mean fold MAE",
            "mean_fold_rmse": "Mean fold RMSE",
        }).to_string(index=False, formatters={
            "Mean fold MAE": "{:,.2f}".format,
            "Mean fold RMSE": "{:,.2f}".format,
        })
    )

    print("\nPER-FOLD MAE")
    print("-" * 48)
    per_fold = comparison.pivot(index="fold", columns="model", values="mae")
    print(per_fold.to_string(float_format=lambda value: f"{value:,.2f}"))

    print("\nSELECTED MODEL CHECK")
    print("-" * 48)
    print(f"Best model by mean fold MAE: {best_model}")
    print(f"Seasonal-naive mean fold MAE:  {seasonal['mean_fold_mae']:,.2f}")
    print(f"Seasonal-naive mean fold RMSE: {seasonal['mean_fold_rmse']:,.2f}")
    if best_model != EXPECTED_SEASONAL_MODEL:
        raise ValueError(
            f"Expected seasonal naive to rank first, but '{best_model}' ranked first."
        )
    print("PASS: Seasonal naive ranks first by mean fold MAE.")

    # ---------------- Future forecast ----------------
    future = pd.read_csv(FUTURE_FILE)
    required_future = {"week_ending", "forecast_revenue"}
    missing_future = required_future - set(future.columns)
    if missing_future:
        raise ValueError(f"Missing future-forecast columns: {sorted(missing_future)}")
    if future.empty:
        raise ValueError("Future forecast is empty.")

    future["week_ending"] = pd.to_datetime(
        future["week_ending"], errors="coerce"
    )
    future["forecast_revenue"] = pd.to_numeric(
        future["forecast_revenue"], errors="coerce"
    )

    if future["week_ending"].isna().any():
        raise ValueError("Future forecast has missing or invalid dates.")
    if future["forecast_revenue"].isna().any():
        raise ValueError("Future forecast has missing or invalid values.")
    if not np.isfinite(
        future["forecast_revenue"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Future forecast contains non-finite values.")
    if future["week_ending"].duplicated().any():
        raise ValueError("Duplicate dates in future forecast.")
    if (future["forecast_revenue"] < 0).any():
        raise ValueError("Negative revenue forecasts detected.")

    future = future.sort_values("week_ending").reset_index(drop=True)
    last_history = weekly["week_ending"].max()
    expected_dates = pd.date_range(
        start=last_history + pd.Timedelta(days=7),
        periods=EXPECTED_FORECAST_PERIODS,
        freq="7D",
    )

    if len(future) != EXPECTED_FORECAST_PERIODS:
        raise ValueError(
            f"Expected {EXPECTED_FORECAST_PERIODS} forecast periods, "
            f"found {len(future)}."
        )
    if not future["week_ending"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="week_ending")
    ):
        raise ValueError(
            "Future forecast dates must be the 12 consecutive weeks "
            "immediately after the historical series."
        )

    # Seasonal-naive output should match the corresponding 52-week-lag values.
    if len(weekly) < 52 + EXPECTED_FORECAST_PERIODS:
        raise ValueError(
            "Not enough history to verify a 12-week, 52-week seasonal-naive forecast."
        )

    expected_values = weekly["revenue"].iloc[
        -52:-52 + EXPECTED_FORECAST_PERIODS
    ].to_numpy(dtype=float)
    actual_values = future["forecast_revenue"].to_numpy(dtype=float)

    if not np.allclose(actual_values, expected_values, rtol=1e-9, atol=1e-6):
        raise ValueError(
            "Future forecast values do not match the expected 52-week seasonal-naive values."
        )

    print("\nFUTURE FORECAST CHECK")
    print("-" * 48)
    print(f"Historical data ends: {last_history.date()}")
    print(f"First forecast date:  {future['week_ending'].iloc[0].date()}")
    print(f"Last forecast date:   {future['week_ending'].iloc[-1].date()}")
    print(f"Forecast periods:     {len(future):,}")
    print(f"Negative predictions: {(future['forecast_revenue'] < 0).sum():,}")
    print("Seasonal-naive value reconciliation: PASS")

    print("\nPASS: Model comparison and future forecast checks passed.")
    print(
        "Note: Structural checks and historical validation do not guarantee "
        "that the model will generalize to future retail data."
    )


if __name__ == "__main__":
    main()
