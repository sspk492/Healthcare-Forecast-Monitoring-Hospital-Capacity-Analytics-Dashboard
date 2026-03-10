from pathlib import Path
import numpy as np
import pandas as pd


def main():
    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "processed" / "hospital_capacity_selected_columns.csv"
    processed_dir = project_root / "data" / "processed"
    outputs_dir = project_root / "outputs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading selected dataset...")
    df = pd.read_csv(input_file)

    print("\n=== INITIAL SHAPE ===")
    print(df.shape)

    # --------------------------------------------------
    # Parse date
    # --------------------------------------------------
    print("\nParsing date column...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with invalid date or missing state
    before_drop = len(df)
    df = df.dropna(subset=["state", "date"]).copy()
    after_drop = len(df)

    print(f"Dropped rows with missing state/date: {before_drop - after_drop:,}")

    # --------------------------------------------------
    # Standardize state
    # --------------------------------------------------
    df["state"] = df["state"].astype(str).str.strip().str.upper()

    # --------------------------------------------------
    # Remove duplicates by state + date
    # --------------------------------------------------
    before_dupes = len(df)
    df = df.sort_values(["state", "date"]).drop_duplicates(subset=["state", "date"], keep="first")
    after_dupes = len(df)

    print(f"Removed duplicate state-date rows: {before_dupes - after_dupes:,}")

    # --------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------
    exclude_cols = ["state", "date"]
    numeric_cols = [col for col in df.columns if col not in exclude_cols]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # Replace negative values with NaN for selected metrics
    # --------------------------------------------------
    cols_no_negative = [
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

    cols_no_negative = [col for col in cols_no_negative if col in df.columns]

    negative_summary = []

    for col in cols_no_negative:
        neg_count = (df[col] < 0).sum()
        negative_summary.append({"column_name": col, "negative_count": int(neg_count)})
        df.loc[df[col] < 0, col] = np.nan

    negative_summary_df = pd.DataFrame(negative_summary).sort_values(
        by="negative_count", ascending=False
    )
    negative_summary_df.to_csv(outputs_dir / "negative_values_summary.csv", index=False)

    # --------------------------------------------------
    # Create core engineered metrics
    # --------------------------------------------------
    print("\nCreating engineered features...")

    # Total COVID admissions
    adult_adm = df.get("previous_day_admission_adult_covid_confirmed", pd.Series(np.nan, index=df.index))
    ped_adm = df.get("previous_day_admission_pediatric_covid_confirmed", pd.Series(np.nan, index=df.index))
    df["total_covid_admissions"] = adult_adm.fillna(0) + ped_adm.fillna(0)

    # Bed occupancy rate
    if "inpatient_beds_used" in df.columns and "inpatient_beds" in df.columns:
        df["bed_occupancy_rate_calc"] = np.where(
            df["inpatient_beds"] > 0,
            df["inpatient_beds_used"] / df["inpatient_beds"],
            np.nan
        )
    else:
        df["bed_occupancy_rate_calc"] = np.nan

    # ICU utilization rate
    if "staffed_adult_icu_bed_occupancy" in df.columns and "total_staffed_adult_icu_beds" in df.columns:
        df["icu_utilization_rate_calc"] = np.where(
            df["total_staffed_adult_icu_beds"] > 0,
            df["staffed_adult_icu_bed_occupancy"] / df["total_staffed_adult_icu_beds"],
            np.nan
        )
    else:
        df["icu_utilization_rate_calc"] = np.nan

    # COVID bed share
    if "inpatient_beds_used_covid" in df.columns and "inpatient_beds" in df.columns:
        df["covid_bed_share"] = np.where(
            df["inpatient_beds"] > 0,
            df["inpatient_beds_used_covid"] / df["inpatient_beds"],
            np.nan
        )
    else:
        df["covid_bed_share"] = np.nan

    # --------------------------------------------------
    # Date-based features
    # --------------------------------------------------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    # --------------------------------------------------
    # Sort for grouped calculations
    # --------------------------------------------------
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    # --------------------------------------------------
    # Rolling and lag features by state
    # --------------------------------------------------
    df["admissions_lag_1"] = df.groupby("state")["total_covid_admissions"].shift(1)
    df["admissions_lag_7"] = df.groupby("state")["total_covid_admissions"].shift(7)

    df["admissions_7d_avg"] = (
        df.groupby("state")["total_covid_admissions"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )

    df["deaths_7d_avg"] = (
        df.groupby("state")["deaths_covid"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )

    # Admission growth rate
    prev_adm = df.groupby("state")["total_covid_admissions"].shift(1)
    df["admission_growth_rate"] = np.where(
        prev_adm > 0,
        (df["total_covid_admissions"] - prev_adm) / prev_adm,
        np.nan
    )

    # --------------------------------------------------
    # Capacity risk category
    # --------------------------------------------------
    def classify_risk(x):
        if pd.isna(x):
            return "Unknown"
        elif x < 0.70:
            return "Safe"
        elif x <= 0.85:
            return "Warning"
        return "Critical"

    df["capacity_risk_level"] = df["icu_utilization_rate_calc"].apply(classify_risk)

    # --------------------------------------------------
    # Basic quality flags
    # --------------------------------------------------
    if {"inpatient_beds_used", "inpatient_beds"}.issubset(df.columns):
        df["flag_beds_used_gt_capacity"] = (
            df["inpatient_beds_used"] > df["inpatient_beds"]
        ).astype(int)
    else:
        df["flag_beds_used_gt_capacity"] = 0

    if {"staffed_adult_icu_bed_occupancy", "total_staffed_adult_icu_beds"}.issubset(df.columns):
        df["flag_icu_occ_gt_capacity"] = (
            df["staffed_adult_icu_bed_occupancy"] > df["total_staffed_adult_icu_beds"]
        ).astype(int)
    else:
        df["flag_icu_occ_gt_capacity"] = 0

    # --------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------
    clean_file = processed_dir / "hospital_capacity_clean.csv"
    df.to_csv(clean_file, index=False)

    # --------------------------------------------------
    # Save data quality summary
    # --------------------------------------------------
    quality_summary = {
        "final_row_count": [len(df)],
        "final_column_count": [df.shape[1]],
        "min_date": [df["date"].min()],
        "max_date": [df["date"].max()],
        "unique_states": [df["state"].nunique()],
        "beds_used_gt_capacity_count": [int(df["flag_beds_used_gt_capacity"].sum())],
        "icu_occ_gt_capacity_count": [int(df["flag_icu_occ_gt_capacity"].sum())],
        "missing_total_covid_admissions": [int(df["total_covid_admissions"].isna().sum())],
        "missing_bed_occupancy_rate_calc": [int(df["bed_occupancy_rate_calc"].isna().sum())],
        "missing_icu_utilization_rate_calc": [int(df["icu_utilization_rate_calc"].isna().sum())],
    }

    pd.DataFrame(quality_summary).to_csv(
        outputs_dir / "clean_data_quality_summary.csv", index=False
    )

    print("\n=== FINAL SHAPE ===")
    print(df.shape)

    print("\nSaved cleaned dataset to:")
    print(clean_file)

    print("\nSaved quality summary to:")
    print(outputs_dir / "clean_data_quality_summary.csv")

    print("\nCleaning complete.")


if __name__ == "__main__":
    main()