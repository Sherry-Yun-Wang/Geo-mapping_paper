import pandas as pd
import os
import concurrent.futures

# ------------------------------------------
# Enrollment Reader
# ------------------------------------------
def read_enroll(year_pat_list):
    """Read enrollment for one year, filtering only for needed pat_ids."""
    year, pat_list = year_pat_list
    claims_folder = '/sharefolder/IQVIA'
    year_folder = os.path.join(claims_folder, f'enroll2_{year}')
    csv_in_parts_folder = os.path.join(year_folder, 'csv_in_parts')

    if not os.path.isdir(csv_in_parts_folder):
        print(f"[{year}] Folder not found: {csv_in_parts_folder}")
        return pd.DataFrame(columns=['pat_id', 'pay_type', 'year'])

    csv_files = [f for f in os.listdir(csv_in_parts_folder) if f.endswith('.csv')]
    header_data = ['pat_id', 'mstr_enroll_cd', 'prd_type',
                   'pay_type', 'pcob_type', 'mcob_type', 'month_id']

    pat_set = set(pat_list)
    filtered_parts = []

    for i, csv_file in enumerate(csv_files, 1):
        path = os.path.join(csv_in_parts_folder, csv_file)
        try:
            df = pd.read_csv(path, sep='|', header=None, dtype=str)
        except Exception as e:
            print(f"[{year}] Skipping {csv_file}: {e}")
            continue

        if df.shape[1] < len(header_data):
            print(f"[{year}] Skipping {csv_file}: bad column count {df.shape[1]}")
            continue

        df = df.iloc[:, :len(header_data)]
        df.columns = header_data

        # filter only needed pat_ids
        df = df[df['pat_id'].isin(pat_set)]
        if not df.empty:
            filtered_parts.append(df)

        print(f"[{year}] Processed part {i}/{len(csv_files)}")

    if filtered_parts:
        result = pd.concat(filtered_parts, ignore_index=True)
        # keep one record per patient (latest month_id if multiple)
        result = (result.sort_values("month_id")
                         .drop_duplicates(subset=["pat_id"], keep="last"))
        # Only select pat_id and pay_type, then add year
        result = result[['pat_id', 'pay_type']].copy()
        result['year'] = year
        return result
    else:
        return pd.DataFrame(columns=['pat_id', 'pay_type', 'year'])


# ------------------------------------------
# Main script
# ------------------------------------------
print("Loading patient_year_filled.csv...")
pat_years = pd.read_csv("patient_year_filled.csv", dtype=str)

# normalize types
pat_years['year'] = pd.to_numeric(pat_years['year'], errors='coerce').astype('Int64')
pat_years['pat_id'] = pat_years['pat_id'].astype(str).str.strip()

# Find rows that still need filling
needs_fill = pat_years[pat_years['pay_type'].isna()].copy() if 'pay_type' in pat_years else pat_years.copy()
print(f"{len(needs_fill)} rows need pay_type filling out of {len(pat_years)}")

if needs_fill.empty:
    print("No rows need filling, writing cleaned file...")
    final = pat_years.copy()
else:
    # map year -> list of pat_ids
    year_to_patids = (
        needs_fill.groupby('year')['pat_id']
        .apply(lambda x: list(x.unique()))
        .to_dict()
    )

    print("Starting parallel read of enrollment files...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(read_enroll, year_to_patids.items()))

    # combine enrollment results
    enroll_df = pd.concat(results, ignore_index=True)

    # merge back into patient_year_filled
    merged = pat_years.merge(enroll_df, on=['pat_id', 'year'], how='left', suffixes=('', '_new'))

    # fill pay_type only if missing
    merged['pay_type'] = merged['pay_type'].fillna(merged['pay_type_new'])
    merged.drop(columns=['pay_type_new'], inplace=True)

    final = merged.copy()

# ------------------------------------------
# Handle missing pay_type as "U"
# ------------------------------------------
final['pay_type'] = final['pay_type'].fillna("U")

# Save result
out_path = "payment_type_filled.csv"
final.to_csv(out_path, index=False)
print(f"Saved {len(final)} rows to {out_path}")


