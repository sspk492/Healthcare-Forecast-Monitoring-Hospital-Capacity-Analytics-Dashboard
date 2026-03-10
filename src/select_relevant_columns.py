from pathlib import Path
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_file = project_root / "data" / "raw" / "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv"
    processed_dir = project_root / "data" / "processed"
    outputs_dir = project_root / "outputs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_file)

    candidate_columns = [
        "state",
        "date",
        "inpatient_beds",
        "inpatient_beds_used",
        "inpatient_beds_used_covid",
        "previous_day_admission_adult_covid_confirmed",
        "previous_day_admission_pediatric_covid_confirmed",
        "staffed_adult_icu_bed_occupancy",
        "total_staffed_adult_icu_beds",
        "inpatient_beds_utilization",
        "adult_icu_bed_utilization",
        "total_adult_patients_hospitalized_confirmed_covid",
        "total_pediatric_patients_hospitalized_confirmed_covid",
        "deaths_covid",
        "previous_day_admission_adult_covid_confirmed_18-19",
        "previous_day_admission_adult_covid_confirmed_20-29",
        "previous_day_admission_adult_covid_confirmed_30-39",
        "previous_day_admission_adult_covid_confirmed_40-49",
        "previous_day_admission_adult_covid_confirmed_50-59",
        "previous_day_admission_adult_covid_confirmed_60-69",
        "previous_day_admission_adult_covid_confirmed_70-79",
        "previous_day_admission_adult_covid_confirmed_80+",
    ]

    available_columns = [col for col in candidate_columns if col in df.columns]
    missing_columns = [col for col in candidate_columns if col not in df.columns]

    print("\n=== AVAILABLE COLUMNS ===")
    for col in available_columns:
        print(col)

    print("\n=== MISSING COLUMNS ===")
    for col in missing_columns:
        print(col)

    selected_df = df[available_columns].copy()

    selected_df.to_csv(processed_dir / "hospital_capacity_selected_columns.csv", index=False)
    pd.Series(available_columns, name="available_columns").to_csv(
        outputs_dir / "selected_columns_available.csv", index=False
    )
    pd.Series(missing_columns, name="missing_columns").to_csv(
        outputs_dir / "selected_columns_missing.csv", index=False
    )

    print("\nSaved selected dataset:")
    print(processed_dir / "hospital_capacity_selected_columns.csv")


if __name__ == "__main__":
    main()