from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():

    # -----------------------------------------
    # Paths
    # -----------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "data" / "processed" / "hospital_capacity_clean.csv"

    outputs_dir = project_root / "outputs"
    charts_dir = outputs_dir / "charts"

    charts_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------
    print("Loading cleaned dataset...")

    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])

    print("\nDataset shape:", df.shape)

    # -----------------------------------------
    # NATIONAL DAILY DATA
    # -----------------------------------------

    national = (
        df.groupby("date")
        .agg(
            total_admissions=("total_covid_admissions", "sum"),
            deaths=("deaths_covid", "sum"),
            icu_utilization=("icu_utilization_rate_calc", "mean"),
            bed_occupancy=("bed_occupancy_rate_calc", "mean"),
        )
        .reset_index()
    )

    # -----------------------------------------
    # Chart 1 — National Admissions Trend
    # -----------------------------------------

    plt.figure()

    plt.plot(national["date"], national["total_admissions"])

    plt.title("National COVID Hospital Admissions Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Admissions")

    plt.tight_layout()

    plt.savefig(charts_dir / "national_admissions_trend.png")
    plt.close()

    # -----------------------------------------
    # Chart 2 — ICU Utilization Trend
    # -----------------------------------------

    plt.figure()

    plt.plot(national["date"], national["icu_utilization"])

    plt.title("Average ICU Utilization Trend")
    plt.xlabel("Date")
    plt.ylabel("ICU Utilization Rate")

    plt.tight_layout()

    plt.savefig(charts_dir / "icu_utilization_trend.png")
    plt.close()

    # -----------------------------------------
    # Chart 3 — Bed Occupancy Trend
    # -----------------------------------------

    plt.figure()

    plt.plot(national["date"], national["bed_occupancy"])

    plt.title("Hospital Bed Occupancy Trend")
    plt.xlabel("Date")
    plt.ylabel("Bed Occupancy Rate")

    plt.tight_layout()

    plt.savefig(charts_dir / "bed_occupancy_trend.png")
    plt.close()

    # -----------------------------------------
    # Top States by Admissions
    # -----------------------------------------

    state_admissions = (
        df.groupby("state")["total_covid_admissions"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure()

    state_admissions.plot(kind="bar")

    plt.title("Top 10 States by Total COVID Admissions")
    plt.xlabel("State")
    plt.ylabel("Total Admissions")

    plt.tight_layout()

    plt.savefig(charts_dir / "top_states_admissions.png")
    plt.close()

    # -----------------------------------------
    # Top States by ICU Utilization
    # -----------------------------------------

    state_icu = (
        df.groupby("state")["icu_utilization_rate_calc"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure()

    state_icu.plot(kind="bar")

    plt.title("Top 10 States by Average ICU Utilization")
    plt.xlabel("State")
    plt.ylabel("ICU Utilization Rate")

    plt.tight_layout()

    plt.savefig(charts_dir / "top_states_icu_utilization.png")
    plt.close()

    # -----------------------------------------
    # Save summary tables
    # -----------------------------------------

    national.to_csv(outputs_dir / "national_time_series.csv", index=False)

    state_summary = (
        df.groupby("state")
        .agg(
            total_admissions=("total_covid_admissions", "sum"),
            avg_icu_utilization=("icu_utilization_rate_calc", "mean"),
            avg_bed_occupancy=("bed_occupancy_rate_calc", "mean"),
            total_deaths=("deaths_covid", "sum"),
        )
        .reset_index()
    )

    state_summary.to_csv(outputs_dir / "state_summary.csv", index=False)

    print("\nEDA Completed.")
    print("Charts saved in:", charts_dir)
    print("Summary tables saved in outputs folder.")


if __name__ == "__main__":
    main()