from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():

    # ---------------------------------------
    # Paths
    # ---------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "outputs" / "forecast_predictions.csv"

    metrics_file = project_root / "outputs" / "forecast_metrics.csv"

    monitoring_file = project_root / "outputs" / "forecast_monitoring_dataset.csv"

    # ---------------------------------------
    # Load predictions
    # ---------------------------------------

    print("Loading prediction results...")

    df = pd.read_csv(input_file)

    # ---------------------------------------
    # Calculate errors
    # ---------------------------------------

    df["error"] = df["actual_admissions"] - df["predicted_admissions"]

    df["absolute_error"] = np.abs(df["error"])

    df["percentage_error"] = df["absolute_error"] / df["actual_admissions"]

    # ---------------------------------------
    # Model metrics
    # ---------------------------------------

    mae = mean_absolute_error(
        df["actual_admissions"], df["predicted_admissions"]
    )

    rmse = np.sqrt(
        mean_squared_error(df["actual_admissions"], df["predicted_admissions"])
    )

    mape = df["percentage_error"].mean()

    # ---------------------------------------
    # Baseline metrics
    # ---------------------------------------

    baseline_mae = mean_absolute_error(
        df["actual_admissions"], df["baseline_prediction"]
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(df["actual_admissions"], df["baseline_prediction"])
    )

    baseline_mape = (
        np.abs(df["actual_admissions"] - df["baseline_prediction"])
        / df["actual_admissions"]
    ).mean()

    # ---------------------------------------
    # Save metrics
    # ---------------------------------------

    metrics = pd.DataFrame({
        "model": ["Linear Regression", "Baseline"],
        "MAE": [mae, baseline_mae],
        "RMSE": [rmse, baseline_rmse],
        "MAPE": [mape, baseline_mape],
    })

    metrics.to_csv(metrics_file, index=False)

    print("\nForecast Metrics:")
    print(metrics)

    # ---------------------------------------
    # Save monitoring dataset
    # ---------------------------------------

    monitoring_cols = [
        "date",
        "actual_admissions",
        "predicted_admissions",
        "baseline_prediction",
        "error",
        "absolute_error",
        "percentage_error"
    ]

    df[monitoring_cols].to_csv(monitoring_file, index=False)

    print("\nMonitoring dataset saved to:")
    print(monitoring_file)


if __name__ == "__main__":
    main()