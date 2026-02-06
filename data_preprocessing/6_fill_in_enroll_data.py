import os
import re
import pandas as pd
from typing import List

# ---------------------------
# Helpers
# ---------------------------
ZIP3_RE = re.compile(r'^\d{3}$')

def normalize_zip3(val):
    """Return first 3 digits of val if possible, otherwise None."""
    if pd.isna(val):
        return None
    s = re.sub(r'\D', '', str(val))
    return s[:3] if len(s) >= 3 else None

def is_valid_zip3(val):
    return bool(val) and bool(ZIP3_RE.match(str(val)))

# ---------------------------
# Enrollment reader (robust)
# ---------------------------
def read_enroll(header_data: List[str], patient_list: List[str], claims_folder: str = '/sharefolder/IQVIA'):
    """
    Read enroll parts from /sharefolder/IQVIA/enroll_synth/csv_in_parts, return DataFrame
    only containing rows for patient_list. Handles files with >= expected cols; skips smaller.
    """
    year_folder = os.path.join(claims_folder, 'enroll_synth')
    csv_in_parts_folder = os.path.join(year_folder, 'csv_in_parts')

    if not os.path.isdir(csv_in_parts_folder):
        print(f"Enrollment folder not found: {csv_in_parts_folder}")
        return pd.DataFrame(columns=header_data)

    csv_files = [f for f in os.listdir(csv_in_parts_folder) if f.endswith('.csv')]
    parts = []
    for i, fname in enumerate(csv_files, start=1):
        path = os.path.join(csv_in_parts_folder, fname)
        try:
            part = pd.read_csv(path, sep='|', header=None, dtype=str, engine='python')
        except Exception as e:
            print(f"Skipping {fname} (read error): {e}")
            continue

        if part.shape[1] < len(header_data):
            print(f"Skipping {fname}: has {part.shape[1]} columns (expected >= {len(header_data)})")
            continue

        # keep only first N cols that correspond to header_data
        part = part.iloc[:, :len(header_data)]
        part.columns = header_data

        # normalize pat_id and keep only relevant patients to reduce memory
        part['pat_id'] = part['pat_id'].astype(str).str.strip()
        part = part[part['pat_id'].isin(patient_list)]

        if not part.empty:
            # normalize useful fields
            part['pat_zip3'] = part['pat_zip3'].apply(normalize_zip3)
            part['der_yob'] = pd.to_numeric(part['der_yob'], errors='coerce')
            parts.append(part)

        print(f"Processed {i}/{len(csv_files)}: {fname}  -> kept {len(part)} rows")

    if not parts:
        return pd.DataFrame(columns=header_data)

    combined = pd.concat(parts, ignore_index=True)
    # deduplicate by pat_id (enroll is per patient); keep first seen
    combined = combined.drop_duplicates(subset=['pat_id'], keep='first').reset_index(drop=True)
    return combined

# ---------------------------
# Main cleaning & filling
# ---------------------------

# Enrollment file headers (as you provided)
header_data = [
    'der_sex', 'der_yob', 'pat_id', 'pat_region', 'pat_state', 'pat_zip3',
    'grp_indv_cd', 'mh_cd', 'enr_rel'
]

# Read main file with conservative dtype (keep pat_id as string)
pat_years = pd.read_csv('patient_year.csv', dtype=str)

# Normalize types
pat_years['pat_id'] = pat_years['pat_id'].astype(str).str.strip()
pat_years['year'] = pd.to_numeric(pat_years['year'], errors='coerce')  # keep numeric for age calc
pat_years['age'] = pd.to_numeric(pat_years.get('age'), errors='coerce')  # may be NaN
pat_years['der_sex'] = pat_years['der_sex'].replace({'': pd.NA}).where(lambda s: s.notna(), pd.NA)
pat_years['pat_state'] = pat_years['pat_state'].replace({'': pd.NA}).where(lambda s: s.notna(), pd.NA)
# Normalize existing pat_zip3 so '12345' or '123-45' becomes '123', '.' or missing becomes None
pat_years['pat_zip3'] = pat_years['pat_zip3'].apply(normalize_zip3)

# Which rows need filling? (any of these missing/invalid)
mask_age_missing = pat_years['age'].isna()
mask_sex_missing = pat_years['der_sex'].isna()
mask_state_missing = pat_years['pat_state'].isna()
mask_zip3_invalid = ~pat_years['pat_zip3'].apply(is_valid_zip3)

needs_mask = mask_age_missing | mask_sex_missing | mask_state_missing | mask_zip3_invalid
needs_fill = pat_years[needs_mask].copy()

print(f"{len(needs_fill)} rows need filling out of {len(pat_years)} total rows")

if needs_fill.empty:
    print("Nothing to fill. Writing original file out.")
    pat_years.to_csv('patient_year_filled.csv', index=False)
else:
    # Read enroll only for the patients we need (reduces I/O)
    enroll = read_enroll(header_data, needs_fill['pat_id'].unique().tolist())

    if enroll.empty:
        print("No enrollment data found for the requested patients. No fills performed.")
        pat_years.to_csv('patient_year_filled.csv', index=False)
    else:
        # Ensure consistent types
        enroll['pat_id'] = enroll['pat_id'].astype(str).str.strip()
        # keep 'year' from pat_years when merging
        merge_left = needs_fill[['pat_id', 'year']].copy()
        merged = merge_left.merge(enroll, on='pat_id', how='left', validate='m:1')

        # compute age where possible
        merged['year'] = pd.to_numeric(merged['year'], errors='coerce')
        merged['der_yob'] = pd.to_numeric(merged['der_yob'], errors='coerce')
        merged['age'] = merged.apply(lambda r: (r['year'] - r['der_yob'])
                                    if pd.notna(r['year']) and pd.notna(r['der_yob']) else pd.NA, axis=1)
        merged['age'] = pd.to_numeric(merged['age'], errors='coerce')

        # Keep only rows with a calculable dob -> age, and age in [18,64] (exclude <18 or >=65)
        mask_valid_age_range = merged['age'].notna() & (merged['age'] >= 18) & (merged['age'] <= 65)
        valid_filled = merged[mask_valid_age_range].copy()
        print(f"From enroll data, {len(valid_filled)} patient-year rows have valid dob -> age in 18-65 range")

        if valid_filled.empty:
            print("No valid fills after age filtering. Writing original file out.")
            pat_years.to_csv('patient_year_filled.csv', index=False)
        else:
            # Prepare fill subset and de-duplicate on pat_id+year
            fill_cols = ['pat_id', 'year', 'age', 'der_sex', 'pat_state', 'pat_zip3']
            filled_subset = valid_filled[fill_cols].drop_duplicates(subset=['pat_id', 'year'], keep='first').copy()
            # Normalize zip3 in filled_subset
            filled_subset['pat_zip3'] = filled_subset['pat_zip3'].apply(normalize_zip3)

            # Build a key for mapping: "pat_id|year"
            filled_subset['year_int'] = filled_subset['year'].astype(int).astype(str)
            filled_subset['key'] = filled_subset['pat_id'].astype(str) + '|' + filled_subset['year_int']

            # Prepare mapping dictionaries
            mappings = {}
            for c in ['age', 'der_sex', 'pat_state', 'pat_zip3']:
                mappings[c] = dict(zip(filled_subset['key'], filled_subset[c]))

            # Add key column to pat_years (use int year -> str)
            # If year is missing in pat_years this will produce "pat|0"; those won't map and will be ignored
            pat_years['year_int'] = pat_years['year'].fillna(0).astype(int).astype(str)
            pat_years['key'] = pat_years['pat_id'].astype(str) + '|' + pat_years['year_int']

            # For each column, only update rows that are missing/invalid
            def update_column(col):
                if col == 'pat_zip3':
                    mask = ~pat_years['pat_zip3'].apply(is_valid_zip3)
                else:
                    mask = pat_years[col].isna()
                if mask.sum() == 0:
                    return 0
                mapped = pat_years.loc[mask, 'key'].map(mappings[col])
                # only assign when mapping has a non-null value
                assign_mask = mapped.notna()
                indices_to_assign = pat_years.loc[mask].index[assign_mask]
                pat_years.loc[indices_to_assign, col] = mapped.loc[assign_mask].values
                return len(indices_to_assign)

            filled_counts = {c: update_column(c) for c in ['age', 'der_sex', 'pat_state', 'pat_zip3']}
            print("Filled counts by column:", filled_counts)

            # Clean up temporary cols
            pat_years.drop(columns=['year_int', 'key'], inplace=True)

            # Build validity masks
            valid_zip_mask = pat_years['pat_zip3'].apply(is_valid_zip3)
            valid_age_mask = pat_years['age'].notna()
            valid_sex_mask = pat_years['der_sex'].notna()
            valid_state_mask = pat_years['pat_state'].notna()

            # Combine all
            valid_mask = valid_zip_mask & valid_age_mask & valid_sex_mask & valid_state_mask
            dropped_count = (~valid_mask).sum()

            # Apply filter
            pat_years = pat_years[valid_mask].copy()
            print(f"Dropped {dropped_count} rows due to missing/invalid age, sex, state, or ZIP3")

            # Write result
            pat_years.to_csv('patient_year_filled.csv', index=False)
            print("Wrote patient_year_filled.csv")