import pandas as pd
import numpy as np

df = pd.read_csv('uszips.csv')

# Create zip3 from first 3 digits
df['zip3'] = df['zip'].astype(str).str.zfill(5).str[:3]

# All demographic percentages and medians to population-weight
weight_cols = [
    # Age demographics
    'age_median', 'age_under_10', 'age_10_to_19', 'age_20s', 'age_30s', 
    'age_40s', 'age_50s', 'age_60s', 'age_70s', 'age_over_80', 'age_over_65',
    'age_18_to_24', 'age_over_18',
    # Gender
    'male', 'female',
    # Marital status
    'married', 'divorced', 'never_married', 'widowed',
    # Family
    'family_size', 'family_dual_income',
    # Income
    'income_household_median', 'income_household_under_5', 'income_household_5_to_10',
    'income_household_10_to_15', 'income_household_15_to_20', 'income_household_20_to_25',
    'income_household_25_to_35', 'income_household_35_to_50', 'income_household_50_to_75',
    'income_household_75_to_100', 'income_household_100_to_150', 'income_household_150_over',
    'income_household_six_figure', 'income_individual_median',
    # Housing
    'home_ownership', 'home_value', 'rent_median', 'rent_burden', 'density',
    # Education
    'education_less_highschool', 'education_highschool', 'education_some_college', 
    'education_bachelors', 'education_graduate', 'education_college_or_above', 
    'education_stem_degree',
    # Employment
    'labor_force_participation', 'unemployment_rate', 'self_employed', 'farmer',
    # Race/ethnicity
    'race_white', 'race_black', 'race_asian', 'race_native', 'race_pacific',
    'race_other', 'race_multiple', 'hispanic',
    # Other demographics
    'disabled', 'poverty', 'limited_english', 'commute_time', 
    'health_uninsured', 'veteran', 'charitable_givers',
    # Geography
    'lat', 'lng'
]

# Filter to existing columns
weight_cols = [c for c in weight_cols if c in df.columns]

def pop_weighted_avg(group, col):
    """Calculate population-weighted average, handling NaN values"""
    mask = group[col].notna() & group['population'].notna() & (group['population'] > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(group.loc[mask, col], weights=group.loc[mask, 'population'])

# Aggregate
result = df.groupby('zip3').apply(lambda g: pd.Series({
    # Population sum
    'population': g['population'].sum(),
    'housing_units': g['housing_units'].sum() if 'housing_units' in g.columns else np.nan,
    # Population-weighted averages for all demographic variables
    **{col: pop_weighted_avg(g, col) for col in weight_cols},
    # Metadata
    'zip5_count': len(g),
    'state_id': g['state_id'].mode().iloc[0] if len(g['state_id'].mode()) > 0 else None,
    'state_name': g['state_name'].mode().iloc[0] if len(g['state_name'].mode()) > 0 else None,
}), include_groups=False).reset_index()

# Reorder columns for clarity
col_order = ['zip3', 'state_id', 'state_name', 'population', 'housing_units', 'zip5_count'] + \
            [c for c in weight_cols if c in result.columns]
result = result[[c for c in col_order if c in result.columns]]

print(f"ZIP-3 data: {result.shape[0]} areas, {result.shape[1]} columns")
print(f"\nAll {len(weight_cols)} demographic variables aggregated using population-weighted averages")
print("\nSample output (first 10 ZIP-3 areas):")
print(result[['zip3', 'state_id', 'population', 'age_median', 'income_household_median', 'race_white', 'education_bachelors']].head(10).to_string(index=False))

result.to_excel('./uszips_zip3_popweighted.xlsx', index=False)
print("\nSaved to uszips_zip3_popweighted.xlsx")
