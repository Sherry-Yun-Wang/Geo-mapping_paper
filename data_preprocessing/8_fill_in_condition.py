import pandas as pd
import os
import glob

# Paths
data_folder = "/home/stofer@chapman.edu/federated_analysis/tableau/data"
payment_file = "/home/stofer@chapman.edu/federated_analysis/tableau/payment_type_filled.csv"
output_file = "/home/stofer@chapman.edu/federated_analysis/tableau/payment_type_filled_with_condition.csv"

# Load current payment_type_filled
payment_df = pd.read_csv(payment_file, dtype=str)

# Normalize key columns
payment_df['pat_id'] = payment_df['pat_id'].astype(str).str.strip()
payment_df['year'] = payment_df['year'].astype(str).str.strip()

# Define mask of rows that need condition filled (NaN or blank)
cond_missing_mask = payment_df['condition'].isna() | (payment_df['condition'].astype(str).str.strip() == '')
missing_count = cond_missing_mask.sum()
print(f"Rows needing condition filled: {missing_count}")

if missing_count == 0:
    print("No missing conditions. Saving copy and exiting.")
    payment_df.to_csv(output_file, index=False)
else:
    # Build master lookup across all iqvia files
    lookup_parts = []
    for iqvia_file in sorted(glob.glob(os.path.join(data_folder, "iqvia_pat_*.csv"))):
        print(f"Processing {iqvia_file}")
        try:
            iqvia_df = pd.read_csv(iqvia_file, dtype=str, usecols=['pat_id', 'condition'])
        except Exception as e:
            print(f"  ERROR reading {iqvia_file}: {e}. Skipping.")
            continue

        iqvia_df['pat_id'] = iqvia_df['pat_id'].astype(str).str.strip()
        iqvia_df['condition'] = iqvia_df['condition'].astype(str).str.strip().replace({'nan': ''})
        lookup_parts.append(iqvia_df)

    if not lookup_parts:
        print("⚠️ No lookup data was found. Saving original file.")
        payment_df.to_csv(output_file, index=False)
    else:
        # Concatenate all years together
        lookup = pd.concat(lookup_parts, ignore_index=True)

        # Prefer non-empty condition if multiple rows exist
        lookup['has_condition'] = lookup['condition'].astype(bool)
        lookup = lookup.sort_values(['pat_id', 'has_condition'], ascending=[True, False])
        lookup = lookup.drop_duplicates(subset=['pat_id'], keep='first').drop(columns=['has_condition'])

        # Convert to dictionary for fast lookup
        condition_dict = dict(zip(lookup['pat_id'], lookup['condition']))

        # Fill missing conditions in payment_df
        filled_before = payment_df['condition'].notna().sum()
        payment_df.loc[cond_missing_mask, 'condition'] = payment_df.loc[cond_missing_mask, 'pat_id'].map(condition_dict)
        filled_after = payment_df['condition'].notna().sum()

        filled_now = filled_after - filled_before
        print(f"Filled {filled_now} missing condition values from master lookup")

        # Save final file
        final = payment_df[payment_df['condition'].notna()].copy()
        final.to_csv(output_file, index=False)
        print(f"Saved updated file with condition: {output_file} (rows: {len(final)})")
