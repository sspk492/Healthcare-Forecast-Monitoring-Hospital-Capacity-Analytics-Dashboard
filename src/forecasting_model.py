from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression


def main():

    # ----------------------------------------
    # Paths
    # ----------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "data" / "processed" / "national_forecasting_dataset.csv"

    output_file = project_root / "outputs" / "forecast_predictions.csv"

    # ----------------------------------------
    # Load dataset
    # ----------------------------------------

    print("Loading forecasting dataset...")

    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])

    # ----------------------------------------
    # Train / Test Split
    # ----------------------------------------

    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    print("Training rows:", len(train))
    print("Testing rows:", len(test))

    # ----------------------------------------
    # Features and Target
    # ----------------------------------------

    feature_cols = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "avg_icu_utilization",
        "avg_bed_occupancy",
        "day_of_week",
        "month",
        "is_weekend"
    ]

    target = "total_admissions"

    X_train = train[feature_cols]
    y_train = train[target]

    X_test = test[feature_cols]
    y_test = test[target]

    # ----------------------------------------
    # Train Model
    # ----------------------------------------

    print("Training Linear Regression model...")

    model = LinearRegression()

    model.fit(X_train, y_train)

    # ----------------------------------------
    # Predictions
    # ----------------------------------------

    test["predicted_admissions"] = model.predict(X_test)

    # ----------------------------------------
    # Baseline Prediction
    # ----------------------------------------

    test["baseline_prediction"] = test["lag_1"]

    # ----------------------------------------
    # Combine results
    # ----------------------------------------

    results = test[[
        "date",
        "total_admissions",
        "predicted_admissions",
        "baseline_prediction"
    ]].copy()

    results.rename(columns={
        "total_admissions": "actual_admissions"
    }, inplace=True)

    results.to_csv(output_file, index=False)

    print("\nForecast predictions saved to:")
    print(output_file)


if __name__ == "__main__":
    main()