
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
INPUT_FILE = DATA / "weekly_revenue.csv"
OUTPUT_FILE = DATA / "forecast_model_comparison.csv"

INITIAL_TRAIN_SIZE = 58
HORIZON = 12
STEP = 12
SEASONAL_PERIOD = 52


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(errors**2)))


def make_predictions(train, horizon):
    """Fit candidate models using training data only."""
    predictions = {}

    # Existing benchmark: mean of the training window.
    predictions["Training mean"] = np.repeat(train.mean(), horizon)

    # Last observed value carried forward.
    predictions["Naive"] = np.repeat(train.iloc[-1], horizon)

    # Weekly seasonal-naive baseline, if enough history exists.
    if len(train) >= SEASONAL_PERIOD:
        seasonal_values = train.iloc[-SEASONAL_PERIOD:].to_numpy()
        predictions["Seasonal naive"] = np.resize(
            seasonal_values, horizon
        )

    # Additive-trend exponential smoothing without seasonality.
    try:
        fitted = ExponentialSmoothing(
            train,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)

        predictions["Exponential smoothing"] = np.asarray(
            fitted.forecast(horizon)
        )
    except Exception as exc:
        print(f"Exponential smoothing skipped: {exc}")

    # Seasonal model only when sufficient training history is available.
    if len(train) >= 2 * SEASONAL_PERIOD:
        try:
            fitted_seasonal = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=SEASONAL_PERIOD,
                initialization_method="estimated",
            ).fit(optimized=True)

            predictions["Seasonal exponential smoothing"] = np.asarray(
                fitted_seasonal.forecast(horizon)
            )
        except Exception as exc:
            print(f"Seasonal exponential smoothing skipped: {exc}")

    return predictions


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["week_ending"])
    df = df.sort_values("week_ending").reset_index(drop=True)

    if df["revenue"].isna().any():
        raise ValueError("Weekly revenue contains missing values.")

    values = df["revenue"].astype(float)

    fold_results = []
    train_size = INITIAL_TRAIN_SIZE
    fold_number = 1

    while train_size + HORIZON <= len(values):
        train = values.iloc[:train_size]
        test = values.iloc[train_size:train_size + HORIZON]

        predictions = make_predictions(train, HORIZON)

        for model_name, forecast in predictions.items():
            if len(forecast) != len(test):
                raise ValueError(
                    f"{model_name} returned an incorrect forecast length."
                )

            if not np.isfinite(forecast).all():
                raise ValueError(
                    f"{model_name} returned non-finite predictions."
                )

            fold_results.append({
                "fold": fold_number,
                "train_weeks": len(train),
                "test_weeks": len(test),
                "model": model_name,
                "mae": mae(test, forecast),
                "rmse": rmse(test, forecast),
            })

        train_size += STEP
        fold_number += 1

    results = pd.DataFrame(fold_results)

    if results.empty:
        raise ValueError("No validation folds were generated.")

    summary = (
        results.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            total_test_windows=("test_weeks", "sum"),
            mean_fold_mae=("mae", "mean"),
            mean_fold_rmse=("rmse", "mean"),
        )
        .sort_values("mean_fold_mae")
    )

    DATA.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nFORECAST MODEL COMPARISON")
    print("=" * 75)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\nPER-FOLD MAE")
    print("-" * 75)
    print(
        results.pivot(index="fold", columns="model", values="mae")
        .round(2)
        .to_string()
    )

    print(f"\nSaved detailed results to: {OUTPUT_FILE}")
    print(
        "\nNote: Mean fold metrics weight each fold equally. "
        "Use the same validation windows when comparing models."
    )


if __name__ == "__main__":
    main()