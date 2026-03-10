from pathlib import Path
import pandas as pd

def main():
    # -----------------------------
    # File paths
    # -----------------------------
    project_root = Path(__file__).resolve().parents[1]
    raw_file = project_root / "data" / "raw" / "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    print("\nLoading dataset...")
    df = pd.read_csv(raw_file)

    # -----------------------------
    # Basic shape
    # -----------------------------
    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    # -----------------------------
    # First few rows
    # -----------------------------
    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    # -----------------------------
    # Column names
    # -----------------------------
    print("\n=== COLUMN NAMES ===")
    for col in df.columns:
        print(col)

    # Save column names
    pd.Series(df.columns, name="column_name").to_csv(
        output_dir / "column_names.csv", index=False
    )

    # -----------------------------
    # Data types
    # -----------------------------
    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    dtypes_df = pd.DataFrame({
        "column_name": df.columns,
        "dtype": df.dtypes.astype(str).values
    })
    dtypes_df.to_csv(output_dir / "data_types.csv", index=False)

    # -----------------------------
    # Missing values summary
    # -----------------------------
    print("\n=== MISSING VALUES SUMMARY ===")
    missing_df = pd.DataFrame({
        "column_name": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values
    }).sort_values(by="missing_pct", ascending=False)

    print(missing_df.head(30))
    missing_df.to_csv(output_dir / "missing_values_summary.csv", index=False)

    # -----------------------------
    # Duplicate rows
    # -----------------------------
    print("\n=== DUPLICATE ROWS ===")
    duplicate_count = df.duplicated().sum()
    print(f"Full duplicate rows: {duplicate_count:,}")

    # -----------------------------
    # Date checks
    # -----------------------------
    print("\n=== DATE CHECKS ===")
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"Null parsed dates: {df['date_parsed'].isna().sum():,}")
        print(f"Min date: {df['date_parsed'].min()}")
        print(f"Max date: {df['date_parsed'].max()}")
    else:
        print("No 'date' column found.")

    # -----------------------------
    # State checks
    # -----------------------------
    print("\n=== STATE CHECKS ===")
    if "state" in df.columns:
        print(f"Unique states: {df['state'].nunique()}")
        print("Sample states:")
        print(sorted(df["state"].dropna().unique())[:20])
    else:
        print("No 'state' column found.")

    # -----------------------------
    # State-date duplicate check
    # -----------------------------
    print("\n=== STATE-DATE DUPLICATE CHECK ===")
    if "state" in df.columns and "date_parsed" in df.columns:
        state_date_dupes = df.duplicated(subset=["state", "date_parsed"]).sum()
        print(f"Duplicate state-date rows: {state_date_dupes:,}")

        dupes_df = df[df.duplicated(subset=["state", "date_parsed"], keep=False)].copy()
        dupes_df.to_csv(output_dir / "duplicate_state_date_rows.csv", index=False)
    else:
        print("Could not run state-date duplicate check.")

    # -----------------------------
    # Numeric summary
    # -----------------------------
    print("\n=== NUMERIC SUMMARY ===")
    numeric_summary = df.describe(include="number").T
    print(numeric_summary.head(20))
    numeric_summary.to_csv(output_dir / "numeric_summary.csv")

    print("\nInspection complete.")
    print(f"Saved outputs in: {output_dir}")


if __name__ == "__main__":
    main()