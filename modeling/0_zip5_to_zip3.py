import pandas as pd
import numpy as np

df = pd.read_csv('uszips.csv')

# Create zip3 from first 3 digits
df['zip3'] = df['zip'].astype(str).str.zfill(5).str[:3]

# Identify column types for appropriate aggregation
# Population-weighted averages for rates/percentages/medians
# Sums for counts

# Columns to sum (counts)
sum_cols = ['population', 'housing_units']

# Columns to population-weight (rates, percentages, medians)
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
    'limited_english', 'commute_time', 'health_uninsured', 'veteran', 'charitable_givers'
]

# Geographic - use population-weighted centroid
geo_cols = ['lat', 'lng']

# Filter to only existing columns
sum_cols = [c for c in sum_cols if c in df.columns]
weight_cols = [c for c in weight_cols if c in df.columns]
geo_cols = [c for c in geo_cols if c in df.columns]

# Create aggregation
result = df.groupby('zip3').apply(lambda g: pd.Series({
    # Sums
    **{col: g[col].sum() for col in sum_cols},
    # Population-weighted averages
    **{col: np.average(g[col].dropna(), weights=g.loc[g[col].notna(), 'population']) 
       if g.loc[g[col].notna(), 'population'].sum() > 0 and len(g[col].dropna()) > 0 
       else np.nan 
       for col in weight_cols + geo_cols},
    # Count of zip5s in each zip3
    'zip5_count': len(g),
    # Mode for categorical - state
    'state_id': g['state_id'].mode().iloc[0] if len(g['state_id'].mode()) > 0 else None,
    'state_name': g['state_name'].mode().iloc[0] if len(g['state_name'].mode()) > 0 else None,
})).reset_index()

print(f"ZIP-3 aggregated data: {result.shape[0]} zip3 areas, {result.shape[1]} columns")
print("\nFirst 10 rows preview:")
print(result[['zip3', 'state_id', 'population', 'income_household_median', 'zip5_count']].head(10))

# Save to Excel
result.to_excel('./uszips_zip3.xlsx', index=False)
print("\nSaved to uszips_zip3.xlsx")
