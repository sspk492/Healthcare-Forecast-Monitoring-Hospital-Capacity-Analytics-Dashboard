from pathlib import Path
import pandas as pd
import numpy as np


def main():

    # ----------------------------------------
    # Paths
    # ----------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "data" / "processed" / "hospital_capacity_clean.csv"

    output_file = project_root / "data" / "processed" / "national_forecasting_dataset.csv"

    # ----------------------------------------
    # Load data
    # ----------------------------------------

    print("Loading cleaned hospital dataset...")

    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])

    # ----------------------------------------
    # Aggregate to NATIONAL level
    # ----------------------------------------

    national = (
        df.groupby("date")
        .agg(
            total_admissions=("total_covid_admissions", "sum"),
            total_deaths=("deaths_covid", "sum"),
            avg_icu_utilization=("icu_utilization_rate_calc", "mean"),
            avg_bed_occupancy=("bed_occupancy_rate_calc", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )

    print("National dataset shape:", national.shape)

    # ----------------------------------------
    # Create Time Features
    # ----------------------------------------

    national["year"] = national["date"].dt.year
    national["month"] = national["date"].dt.month
    national["day_of_week"] = national["date"].dt.dayofweek
    national["is_weekend"] = national["day_of_week"].isin([5, 6]).astype(int)

    # ----------------------------------------
    # Create Lag Features
    # ----------------------------------------

    national["lag_1"] = national["total_admissions"].shift(1)
    national["lag_7"] = national["total_admissions"].shift(7)
    national["lag_14"] = national["total_admissions"].shift(14)

    # ----------------------------------------
    # Rolling Statistics
    # ----------------------------------------

    national["rolling_mean_7"] = national["total_admissions"].rolling(7).mean()
    national["rolling_mean_14"] = national["total_admissions"].rolling(14).mean()

    national["rolling_std_7"] = national["total_admissions"].rolling(7).std()

    # ----------------------------------------
    # Drop rows with missing lag values
    # ----------------------------------------

    national = national.dropna().reset_index(drop=True)

    print("Forecast dataset shape after lag features:", national.shape)

    # ----------------------------------------
    # Save forecasting dataset
    # ----------------------------------------

    national.to_csv(output_file, index=False)

    print("\nForecasting dataset saved to:")
    print(output_file)


if __name__ == "__main__":
    main()