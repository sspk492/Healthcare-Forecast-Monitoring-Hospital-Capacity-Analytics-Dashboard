from pathlib import Path
import pandas as pd


def main():

    # -----------------------------------
    # Paths
    # -----------------------------------

    project_root = Path(__file__).resolve().parents[1]

    hospital_file = project_root / "data" / "processed" / "hospital_capacity_clean.csv"

    forecast_file = project_root / "outputs" / "forecast_monitoring_dataset.csv"

    tableau_dir = project_root / "data" / "tableau"

    tableau_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # Load datasets
    # -----------------------------------

    hospital_df = pd.read_csv(hospital_file)

    forecast_df = pd.read_csv(forecast_file)

    hospital_df["date"] = pd.to_datetime(hospital_df["date"])

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    # -----------------------------------
    # Dataset 1 — Executive Overview
    # -----------------------------------

    executive = (
        hospital_df.groupby(["date", "state"])
        .agg(
            total_admissions=("total_covid_admissions", "sum"),
            icu_utilization=("icu_utilization_rate_calc", "mean"),
            bed_occupancy=("bed_occupancy_rate_calc", "mean"),
            covid_deaths=("deaths_covid", "sum"),
        )
        .reset_index()
    )

    executive.to_csv(
        tableau_dir / "executive_overview.csv", index=False
    )

    # -----------------------------------
    # Dataset 2 — Forecast Monitoring
    # -----------------------------------

    forecast_df.to_csv(
        tableau_dir / "forecast_monitoring.csv", index=False
    )

    # -----------------------------------
    # Dataset 3 — Demographic Dataset
    # -----------------------------------

    age_columns = [
        col for col in hospital_df.columns
        if "previous_day_admission_adult_covid_confirmed_" in col
    ]

    demographic = hospital_df[
        ["date", "state"] + age_columns
    ].copy()

    demographic = demographic.melt(
        id_vars=["date", "state"],
        value_vars=age_columns,
        var_name="age_group",
        value_name="admissions"
    )

    demographic["age_group"] = demographic["age_group"].str.replace(
        "previous_day_admission_adult_covid_confirmed_", "",
        regex=False
    )

    demographic.to_csv(
        tableau_dir / "demographic_analysis.csv", index=False
    )

    print("\nTableau datasets created in:")
    print(tableau_dir)


if __name__ == "__main__":
    main()