import pandas as pd

# Load both datasets
df_iqvia = pd.read_csv('final_dataset_from_IQVIA.csv')
df_zip3 = pd.read_excel('uszips_zip3_with_urbancore.xlsx')

print("IQVIA zip3 sample:", df_iqvia['zip3'].head(10).tolist())
print("ZIP3 characteristics zip3 sample:", df_zip3['zip3'].head(10).tolist())

# Convert both to same format (string with leading zeros)
df_iqvia['zip3'] = df_iqvia['zip3'].astype(str).str.zfill(3)
df_zip3['zip3'] = df_zip3['zip3'].astype(str).str.zfill(3)

print("\nAfter formatting:")
print("IQVIA zip3 sample:", df_iqvia['zip3'].head(10).tolist())
print("ZIP3 characteristics zip3 sample:", df_zip3['zip3'].head(10).tolist())

# Check overlap
iqvia_zips = set(df_iqvia['zip3'])
zip3_zips = set(df_zip3['zip3'])
print(f"\nIQVIA has {len(iqvia_zips)} unique zip3s")
print(f"ZIP3 characteristics has {len(zip3_zips)} unique zip3s")
print(f"Overlap: {len(iqvia_zips & zip3_zips)} zip3s")
print(f"IQVIA zip3s not in characteristics: {len(iqvia_zips - zip3_zips)}")

# Merge - keep all IQVIA records, add characteristics
merged = df_iqvia.merge(df_zip3, on='zip3', how='left')

print(f"\nMerged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
print(f"Rows with neighborhood data: {merged['population'].notna().sum()}")

# Show sample
print("\nSample of merged data:")
print(merged[['zip3', 'total_GLP_pat', 't2d_obesity_pat_count', 'state_id', 'population', 'income_household_median', 'UrbanCore']].head(15).to_string(index=False))

# Save
merged.to_csv('./IQVIA_with_zip3_characteristics.csv', index=False)
print("\nSaved to IQVIA_with_zip3_characteristics.csv")
