import pandas as pd
import os

# -----------------------------
# Paths
# -----------------------------
payment_file = "/home/stofer@chapman.edu/federated_analysis/tableau/payment_type_filled_with_condition.csv"
zip_map_file = "/home/stofer@chapman.edu/federated_analysis/tableau/weighted_zip_by_zip3.csv"
us_zips_file = "/home/stofer@chapman.edu/federated_analysis/tableau/uszips.csv"
output_file = "/home/stofer@chapman.edu/federated_analysis/tableau/payment_almost_all_filled.csv"

# -----------------------------
# Load files
# -----------------------------
df = pd.read_csv(payment_file, dtype=str)
us_zips = pd.read_csv(us_zips_file, dtype={'zip': str})
zip_map = pd.read_csv(zip_map_file, dtype={'zip3': str, 'weighted_zip': str})

initial_rows = len(df)

# -----------------------------
# 1. Clean pat_zip3
# -----------------------------
before_clean = len(df)
df = df[df['pat_zip3'].notna()]
df = df[df['pat_zip3'] != '.']
after_clean = len(df)
print(f"Step 1 - Clean pat_zip3: Dropped {before_clean - after_clean} rows")

df['pat_zip3'] = df['pat_zip3'].astype(str).str.zfill(3)
zip_map['weighted_zip'] = zip_map['weighted_zip'].astype(str).str.zfill(5)

# -----------------------------
# 2. Merge ZIP3 → weighted ZIP
# -----------------------------
before_merge1 = len(df)
df = df.merge(
    zip_map[['zip3', 'weighted_zip']],
    left_on='pat_zip3',
    right_on='zip3',
    how='left',
    suffixes=("", "_dup")
)
# If weighted_zip already existed, prefer the new one
if 'weighted_zip_dup' in df.columns:
    df['weighted_zip'] = df['weighted_zip_dup'].fillna(df.get('weighted_zip'))
    df = df.drop(columns=['weighted_zip_dup'])

missing_zip3 = df['weighted_zip'].isna().sum()
print(f"Step 2 - Merge weighted_zip: {missing_zip3} rows have no match")

# -----------------------------
# 3. Merge weighted_zip → us_zips
# -----------------------------
df['weighted_zip'] = df['weighted_zip'].astype(str).str.zfill(5)
us_zips['zip'] = us_zips['zip'].astype(str).str.zfill(5)

selected_cols = [
    'zip',
    'income_household_median', 'income_individual_median', 'poverty',
    'education_less_highschool', 'education_highschool',
    'education_some_college', 'education_bachelors', 'education_graduate',
    'race_white', 'race_black', 'race_asian', 'hispanic'
]

us_zips = us_zips[selected_cols]

before_merge2 = len(df)
df = df.merge(us_zips, left_on='weighted_zip', right_on='zip', how='left', suffixes=("", "_dup"))

# If duplicate columns came in from us_zips merge, prefer the new values
for col in selected_cols:
    if f"{col}_dup" in df.columns:
        df[col] = df[f"{col}_dup"].fillna(df[col])
        df = df.drop(columns=[f"{col}_dup"])


# -----------------------------
# 4. Check if us_zips data merged correctly
# -----------------------------
missing_us_zips = df[['income_household_median', 'income_individual_median', 'poverty', 
                      'education_less_highschool', 'education_highschool', 
                      'education_some_college', 'education_bachelors', 'education_graduate', 
                      'race_white', 'race_black', 'race_asian', 'hispanic']].isna().sum().sum()

print(f"Step 3 - Merge us_zips: {missing_us_zips} missing neighborhood characteristics")

# -----------------------------
# 5. Drop rows with missing ZIP merges
# -----------------------------
before_dropna = len(df)
#df = df.dropna(subset=['weighted_zip', 'zip'])  # Drop rows where either ZIP or weighted ZIP is NA
after_dropna = len(df)
print(f"Step 4 - Drop NAs: Dropped {before_dropna - after_dropna} rows")

# -----------------------------
# 6. Select final columns
# -----------------------------
final_cols = [
    'pat_id', 'year', 'age', 'der_sex', 'pay_type', 'pat_state', 'condition', 'pat_zip3',
    'weighted_zip', 'zip',
    'income_household_median', 'income_individual_median', 'poverty',
    'education_less_highschool', 'education_highschool', 'education_some_college',
    'education_bachelors', 'education_graduate',
    'race_white', 'race_black', 'race_asian', 'hispanic'
]
df = df[final_cols]

# -----------------------------
# 7. Save final file
# -----------------------------
df.to_csv(output_file, index=False)

print(f"Initial rows: {initial_rows}")
print(f"Final dataset saved as {output_file} with shape {df.shape}")
