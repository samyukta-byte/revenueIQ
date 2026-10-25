
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "weekly_revenue.csv"
OUTPUT = ROOT / "data" / "processed" / "rolling_validation_results.csv"

INITIAL_TRAIN_WEEKS = 58
TEST_HORIZON = 12
STEP_SIZE = 12


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }


def main():
    df = pd.read_csv(INPUT, parse_dates=["week_ending"])
    df = df.sort_values("week_ending").reset_index(drop=True)
    values = df["revenue"].astype(float).to_numpy()
    dates = df["week_ending"]

    if len(values) < INITIAL_TRAIN_WEEKS + TEST_HORIZON:
        raise ValueError("Not enough weekly data for rolling validation.")

    results = []
    fold = 1
    train_end = INITIAL_TRAIN_WEEKS

    while train_end + TEST_HORIZON <= len(values):
        train = values[:train_end]
        actual = values[train_end:train_end + TEST_HORIZON]
        test_dates = dates.iloc[train_end:train_end + TEST_HORIZON]

        baseline = np.full(TEST_HORIZON, train.mean())

        try:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            smoothing = np.asarray(model.forecast(TEST_HORIZON))
        except Exception as exc:
            raise RuntimeError(
                f"Exponential smoothing failed on fold {fold}: {exc}"
            ) from exc

        for i, date in enumerate(test_dates):
            results.append({
                "fold": fold,
                "week_ending": date,
                "actual": actual[i],
                "baseline_prediction": baseline[i],
                "smoothing_prediction": smoothing[i],
                "baseline_absolute_error": abs(actual[i] - baseline[i]),
                "smoothing_absolute_error": abs(actual[i] - smoothing[i]),
            })

        print(
            f"Fold {fold}: train={len(train)} weeks, "
            f"test={len(actual)} weeks"
        )

        fold += 1
        train_end += STEP_SIZE

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT, index=False)

    print("\nROLLING-ORIGIN VALIDATION RESULTS")
    print("=" * 45)

    for name, column in [
        ("Training-mean baseline", "baseline_prediction"),
        ("Exponential smoothing", "smoothing_prediction"),
    ]:
        actual = result_df["actual"]
        predicted = result_df[column]
        metrics = calculate_metrics(actual, predicted)

        print(f"\n{name}:")
        print(f"  MAE:  {metrics['MAE']:,.2f}")
        print(f"  RMSE: {metrics['RMSE']:,.2f}")

    baseline_mae = result_df["baseline_absolute_error"].mean()
    smoothing_mae = result_df["smoothing_absolute_error"].mean()

    if baseline_mae > 0:
        improvement = (baseline_mae - smoothing_mae) / baseline_mae * 100
        print(f"\nMAE improvement across all test weeks: {improvement:.2f}%")

    print(f"\nValidation folds: {result_df['fold'].nunique()}")
    print(f"Total predictions evaluated: {len(result_df)}")
    print(f"Saved results to: {OUTPUT}")
    print(
        "\nNote: test windows are chronological and non-overlapping. "
        "These results evaluate this dataset, not guaranteed future performance."
    )


if __name__ == "__main__":
    main()
