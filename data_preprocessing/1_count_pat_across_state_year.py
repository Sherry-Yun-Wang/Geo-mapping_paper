import pandas as pd
import os

# -----------------------------
# Paths
# -----------------------------
claims_folder = '/sharefolder/IQVIA'
year_folder = os.path.join(claims_folder, 'enroll_synth')
csv_in_parts_folder = os.path.join(year_folder, 'csv_in_parts')

data_folder = '/home/stofer@chapman.edu/federated_analysis/tableau/data'
output_file = os.path.join(data_folder, 'updated_state_counts.csv')

# Enrollment file headers
header_data = [
    'der_sex', 'der_yob', 'pat_id', 'pat_region', 'pat_state', 'pat_zip3',
    'grp_indv_cd', 'mh_cd', 'enr_rel'
]

# -----------------------------
# Collect patient IDs per year from condition files
# -----------------------------
patient_years = []
for year in range(2010, 2023):
    file_path = os.path.join(data_folder, f"iqvia_pat_{year}.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df_pat = pd.read_csv(file_path, dtype=str)
    if 'pat_id' not in df_pat.columns:
        raise ValueError(f"File {file_path} missing pat_id column")

    df_pat['year'] = year
    patient_years.append(df_pat[['pat_id', 'year']])

patient_years_df = pd.concat(patient_years, ignore_index=True)
patient_list = patient_years_df['pat_id'].unique().tolist()

# -----------------------------
# Read enrollment data in parts, filter to patient list
# -----------------------------
def read_enroll(header_data, patient_list):
    if not os.path.exists(csv_in_parts_folder):
        print(f"Folder not found: {csv_in_parts_folder}")
        return pd.DataFrame()

    csv_files = [file for file in os.listdir(csv_in_parts_folder) if file.endswith('.csv')]
    all_filtered_data = []

    for i, csv_file in enumerate(csv_files, start=1):
        file_path = os.path.join(csv_in_parts_folder, csv_file)
        data_part = pd.read_csv(file_path, sep='|', header=None, dtype=str)
        data_part.columns = header_data

        filtered_data = data_part[data_part['pat_id'].isin(patient_list)]
        all_filtered_data.append(filtered_data)
        print(f"Processed enroll file {i}/{len(csv_files)}", flush=True)

    if all_filtered_data:
        return pd.concat(all_filtered_data, ignore_index=True)
    else:
        return pd.DataFrame()

enroll_result = read_enroll(header_data, patient_list)

# -----------------------------
# Merge patient list with enrollment demographics
# -----------------------------
# Merge patient list with enrollment demographics
merged = pd.merge(patient_years_df, enroll_result, on='pat_id', how='inner')

# NEW: drop duplicate rows to ensure counts aren’t inflated
merged = merged.drop_duplicates(subset=['pat_id', 'year', 'pat_state'])
# Drop missing yob
merged = merged.dropna(subset=['der_yob'])
merged['der_yob'] = merged['der_yob'].astype(int)

# Compute age
merged['age'] = merged['year'] - merged['der_yob']

# Keep adults only (18–64)
merged = merged[(merged['age'] >= 18) & (merged['age'] <= 65)]

# -----------------------------
# Group by year + state
# -----------------------------
state_counts = (
    merged.groupby(['year', 'pat_state'])['pat_id']
    .nunique()
    .reset_index()
    .rename(columns={'pat_id': 'count'})
)

# -----------------------------
# Save
# -----------------------------
state_counts.to_csv(output_file, index=False)
print(f"Saved state counts to {output_file}")
