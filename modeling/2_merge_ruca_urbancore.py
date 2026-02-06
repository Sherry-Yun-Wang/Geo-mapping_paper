import pandas as pd
import numpy as np

# Load both datasets
df_zips = pd.read_csv('uszips.csv')
df_ruca = pd.read_excel('RUCA-codes-2020-tract.xlsx', sheet_name='RUCA2020 Tract Data', header=0, skiprows=1)

# Aggregate RUCA to county level - population-weighted UrbanCore
# UrbanCore = 1 if urban core, 0 if not
# This gives us % of county population living in urban core

county_ruca = df_ruca.groupby('CountyFIPS20').apply(
    lambda g: pd.Series({
        'urban_core_pct': np.average(g['UrbanCore'], weights=g['Population']) if g['Population'].sum() > 0 else np.nan,
        'tract_population': g['Population'].sum()
    }), include_groups=False
).reset_index()

county_ruca['CountyFIPS20'] = county_ruca['CountyFIPS20'].astype(float)
print(f"County-level RUCA data: {len(county_ruca)} counties")
print(county_ruca.head(10))

# Merge with ZIP data
df_zips['county_fips'] = df_zips['county_fips'].astype(float)
df_merged = df_zips.merge(county_ruca, left_on='county_fips', right_on='CountyFIPS20', how='left')
print(f"\nMerged ZIP data: {len(df_merged)} zips, {df_merged['urban_core_pct'].notna().sum()} with UrbanCore data")

# Create zip3
df_merged['zip3'] = df_merged['zip'].astype(str).str.zfill(5).str[:3]

# All columns to population-weight
weight_cols = [
    'density', 'age_median', 'age_under_10', 'age_10_to_19', 'age_20s', 'age_30s', 
    'age_40s', 'age_50s', 'age_60s', 'age_70s', 'age_over_80', 'age_over_65',
    'age_18_to_24', 'age_over_18', 'male', 'female', 'married', 'divorced', 
    'never_married', 'widowed', 'family_size', 'family_dual_income',
    'income_household_median', 'income_household_under_5', 'income_household_5_to_10',
    'income_household_10_to_15', 'income_household_15_to_20', 'income_household_20_to_25',
    'income_household_25_to_35', 'income_household_35_to_50', 'income_household_50_to_75',
    'income_household_75_to_100', 'income_household_100_to_150', 'income_household_150_over',
    'income_household_six_figure', 'income_individual_median', 'home_ownership',
    'home_value', 'rent_median', 'rent_burden', 'education_less_highschool',
    'education_highschool', 'education_some_college', 'education_bachelors',
    'education_graduate', 'education_college_or_above', 'education_stem_degree',
    'labor_force_participation', 'unemployment_rate', 'self_employed', 'farmer',
    'race_white', 'race_black', 'race_asian', 'race_native', 'race_pacific',
    'race_other', 'race_multiple', 'hispanic', 'disabled', 'poverty',
    'limited_english', 'commute_time', 'health_uninsured', 'veteran', 'charitable_givers',
    'lat', 'lng', 'urban_core_pct'  # Added UrbanCore
]

weight_cols = [c for c in weight_cols if c in df_merged.columns]

def pop_weighted_avg(group, col):
    mask = group[col].notna() & group['population'].notna() & (group['population'] > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(group.loc[mask, col], weights=group.loc[mask, 'population'])

# Aggregate to ZIP-3
result = df_merged.groupby('zip3').apply(lambda g: pd.Series({
    'population': g['population'].sum(),
    'housing_units': g['housing_units'].sum() if 'housing_units' in g.columns else np.nan,
    **{col: pop_weighted_avg(g, col) for col in weight_cols},
    'zip5_count': len(g),
    'state_id': g['state_id'].mode().iloc[0] if len(g['state_id'].mode()) > 0 else None,
    'state_name': g['state_name'].mode().iloc[0] if len(g['state_name'].mode()) > 0 else None,
}), include_groups=False).reset_index()

# Rename urban_core_pct to UrbanCore for clarity
result = result.rename(columns={'urban_core_pct': 'UrbanCore'})

# Reorder columns
priority_cols = ['zip3', 'state_id', 'state_name', 'population', 'housing_units', 'zip5_count', 'UrbanCore']
other_cols = [c for c in result.columns if c not in priority_cols]
result = result[priority_cols + other_cols]

print(f"\nFinal ZIP-3 data: {result.shape[0]} areas, {result.shape[1]} columns")
print(f"UrbanCore coverage: {result['UrbanCore'].notna().sum()} / {len(result)} ZIP-3 areas")
print("\nSample output:")
print(result[['zip3', 'state_id', 'population', 'UrbanCore', 'income_household_median']].head(15).to_string(index=False))

result.to_excel('./uszips_zip3_with_urbancore.xlsx', index=False)
print("\nSaved to uszips_zip3_with_urbancore.xlsx")
