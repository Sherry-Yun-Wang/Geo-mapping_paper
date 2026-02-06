"""
Need to change tableau_data to instead by based on active patients instead of patients who had their first prescription.
"""
import pandas as pd
import os
from datetime import timedelta
from tqdm import tqdm

# Step 1: Load all prescription files
def load_all_prescriptions(base_path, years):
    dfs = []
    for year in tqdm(years, desc="Loading prescription files"):
        fpath = os.path.join(base_path, f"iqvia_ndc_{year}.csv")
        df = pd.read_csv(fpath, parse_dates=["to_dt"])
        df["year_file"] = year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Step 2: Expand prescriptions into covered years
def expand_to_years(df):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Expanding prescriptions"):
        start = row["to_dt"]
        try:
            days = int(row["dayssup"]) if pd.notna(row["dayssup"]) else 0
        except ValueError:
            days = 0  # fallback if invalid value
        end = start + timedelta(days=days)
        for yr in range(start.year, end.year + 1):
            rows.append({"pat_id": row["pat_id"], "year": yr})
    return pd.DataFrame(rows).drop_duplicates()

# Step 3: Merge with tableau_data_final
def build_final_table(prescription_df, tableau_path, output_path):
    tableau = pd.read_csv(tableau_path)
    
    # Ensure index_date is datetime
    tableau["index_date"] = pd.to_datetime(tableau["index_date"], errors="coerce")

    # Extract pat_id from pat_key
    tableau["pat_id"] = tableau["pat_key"].str.split("_").str[0]

    merged = prescription_df.merge(tableau, on="pat_id", how="left")
    # --- Filtering step ---
    # keep rows if index_date is NA OR index_date.year == year
    mask = merged["index_date"].isna() | (merged["index_date"].dt.year == merged["year"])
    merged = merged[mask]

    # Drop unwanted columns
    drop_cols = [
        "population","age_median","age_over_65","male","female",
        "home_ownership","home_value","rent_median","rent_burden",
        "labor_force_participation","unemployment_rate","health_uninsured","disabled"
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    merged.to_csv(output_path, index=False)
    return merged

# Example usage
base_path = "/sharefolder/wanglab/merck_proposal/"
years = list(range(2010, 2023))  # adjust years available
all_rx = load_all_prescriptions(base_path, years)
expanded = expand_to_years(all_rx)

final = build_final_table(expanded, 
                          "tableau_data_final.csv",
                          "final_patient_year.csv")