import pandas as pd

# pat_key,age,der_sex,index_date,pat_state,pat_zip3,pay_type,condition

# "zip","lat","lng","city","state_id","state_name","zcta","parent_zcta","population","density","county_fips","county_name",
#"county_weights","county_names_all","county_fips_all","imprecise","military","timezone","age_median","age_under_10","age_10_to_19",
# "age_20s","age_30s","age_40s","age_50s","age_60s","age_70s","age_over_80","age_over_65","age_18_to_24","age_over_18","male","female","married",
# "divorced","never_married","widowed","family_size","family_dual_income","income_household_median","income_household_under_5","income_household_5_to_10",
# "income_household_10_to_15","income_household_15_to_20","income_household_20_to_25","income_household_25_to_35","income_household_35_to_50","income_household_50_to_75",
# "income_household_75_to_100","income_household_100_to_150","income_household_150_over","income_household_six_figure","income_individual_median","home_ownership",
# "housing_units","home_value","rent_median","rent_burden","education_less_highschool","education_highschool","education_some_college","education_bachelors",
# "education_graduate","education_college_or_above","education_stem_degree","labor_force_participation","unemployment_rate","self_employed","farmer",
# "race_white","race_black","race_asian","race_native","race_pacific","race_other","race_multiple","hispanic","disabled","poverty","limited_english",
# "commute_time","health_uninsured",
# "veteran","charitable_givers","cbsa_fips","cbsa_name","cbsa_metro","csa_fips","csa_name","metdiv_fips","metdiv_name"

#zip3, weighted_zip


# -----------------------------
# Load files
# -----------------------------
df = pd.read_csv('tableau_data.csv')
us_zips = pd.read_csv('uszips.csv', dtype={'zip': str})
zip_map = pd.read_csv('weighted_zip_by_zip3.csv', dtype={'zip3': str, 'weighted_zip': str})

initial_rows = len(df)

# -----------------------------
# 1. Clean tableau_data
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
df = df.merge(zip_map, left_on='pat_zip3', right_on='zip3', how='left')
after_merge1 = len(df)
print(df.columns)
missing_zip3 = df['weighted_zip'].isna().sum()
print(f"Step 2 - Merge weighted_zip: {missing_zip3} rows have no match")

# -----------------------------
# 3. Merge weighted_zip → us_zips
# -----------------------------
# Normalize keys
df['weighted_zip'] = df['weighted_zip'].astype(str).str.zfill(5)
us_zips['zip'] = us_zips['zip'].astype(str).str.zfill(5)

selected_cols = [
    'zip',
    'population',
    'age_median',
    'age_over_65',
    'male', 'female',
    'income_household_median', 'income_individual_median', 'poverty',
    'education_less_highschool', 'education_highschool', 'education_some_college',
    'education_bachelors', 'education_graduate',
    'home_ownership', 'home_value', 'rent_median', 'rent_burden',
    'labor_force_participation', 'unemployment_rate',
    'health_uninsured', 'disabled',
    'race_white', 'race_black', 'race_asian', 'hispanic'
]

us_zips = us_zips[selected_cols]

before_merge2 = len(df)
df = df.merge(us_zips, left_on='weighted_zip', right_on='zip', how='left')
after_merge2 = len(df)


# -----------------------------
# 4. Drop any rows with missing merges
# -----------------------------
before_dropna = len(df)
df = df.dropna()
after_dropna = len(df)
print(f"Step 4 - Drop NAs: Dropped {before_dropna - after_dropna} rows")

# -----------------------------
# 5. Save cleaned file
# -----------------------------
df.to_csv('tableau_data_final.csv', index=False)

print(f"Initial rows: {initial_rows}")
print(f"Final dataset saved as tableau_data_final.csv with shape {df.shape}")



