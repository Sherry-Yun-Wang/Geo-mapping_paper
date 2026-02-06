import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.special import gammaln, digamma
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('IQVIA_with_zip3_characteristics.csv')

print("="*90)
print("GENERALIZED LINEAR MODEL: NEGATIVE BINOMIAL REGRESSION")
print("Outcome: total_GLP_pat | Offset: log(population)")
print("="*90)

#=============================================================================
# PART 1: VARIABLE SELECTION METHODOLOGY
#=============================================================================

print("""
================================================================================
PART 1: VARIABLE SELECTION METHODOLOGY
================================================================================

PROBLEM: ZIP-3 neighborhood characteristics contain many compositional variables 
(categories that sum to ~100%) and derived variables, leading to multicollinearity.

STATISTICAL APPROACH:

1. COMPOSITIONAL VARIABLES (mutually exclusive categories summing to ~1):
   -------------------------------------------------------------------------
   For variables representing proportions of a whole (e.g., gender, race, 
   education levels), including ALL categories creates PERFECT MULTICOLLINEARITY
   because they sum to a constant.
   
   SOLUTION: Drop ONE category as the REFERENCE GROUP
   SELECTION CRITERION: Drop the LARGEST category (modal group)
   RATIONALE: 
   - Coefficients represent deviation from the most common group
   - Improves interpretability and numerical stability
   - Standard practice in categorical regression

2. DERIVED/REDUNDANT VARIABLES:
   -------------------------------------------------------------------------
   Some variables are mathematical combinations of others.
   Examples: 
   - age_over_65 = age_60s + age_70s + age_over_80
   - education_college_or_above = education_bachelors + education_graduate
   
   SOLUTION: Keep base variables, drop derived aggregates

3. VARIANCE INFLATION FACTOR (VIF) SCREENING:
   -------------------------------------------------------------------------
   After conceptual selection, compute VIF for remaining variables.
   VIF = 1/(1-R²) where R² is from regressing each predictor on all others.
   
   THRESHOLD: VIF > 10 indicates problematic multicollinearity
   PROCEDURE: Iteratively remove highest-VIF variable until all VIF < 10
""")

#=============================================================================
# PART 2: CATEGORY-BY-CATEGORY VARIABLE SELECTION
#=============================================================================

print("""
================================================================================
PART 2: CATEGORY-BY-CATEGORY VARIABLE SELECTION
================================================================================
""")

# --- GENDER ---
print("GENDER (male + female ≈ 100%):")
print(f"  male:   {df['male'].mean():.4f} (mean proportion)")
print(f"  female: {df['female'].mean():.4f} (mean proportion)")
print(f"  Correlation: r = {df['male'].corr(df['female']):.4f}")
print("  DECISION: DROP 'female' (larger category) as reference")
print("  INTERPRETATION: 'male' coefficient = effect of ↑male vs ↑female proportion\n")

# --- AGE ---
print("AGE DISTRIBUTION:")
age_detailed = ['age_under_10', 'age_10_to_19', 'age_20s', 'age_30s', 'age_40s', 
                'age_50s', 'age_60s', 'age_70s', 'age_over_80']
age_derived = ['age_median', 'age_over_65', 'age_over_18', 'age_18_to_24']
print(f"  Sum of detailed brackets: {df[age_detailed].sum(axis=1).mean():.4f}")
print("  Derived variables detected:")
print(f"    - age_over_65 corr with (60s+70s+80+): {df['age_over_65'].corr(df['age_60s']+df['age_70s']+df['age_over_80']):.4f}")
print(f"    - age_over_18 corr with adult sum: {df['age_over_18'].corr(1-df['age_under_10']-df['age_10_to_19']):.4f}")
print("  DECISION: KEEP detailed brackets; DROP derived (median, over_65, over_18, 18_to_24)")
print("  Note: Age brackets don't perfectly sum to 1, so no reference needed\n")

# --- RACE/ETHNICITY ---
print("RACE/ETHNICITY:")
race_cols = ['race_white', 'race_black', 'race_asian', 'race_native', 
             'race_pacific', 'race_other', 'race_multiple']
print(f"  Sum of race categories: {df[race_cols].sum(axis=1).mean():.4f}")
for col in race_cols:
    print(f"    {col}: {df[col].mean():.4f}")
print(f"  hispanic (cross-cutting, NOT mutually exclusive): {df['hispanic'].mean():.4f}")
print("  DECISION: DROP 'race_white' (largest at ~0.70) as reference")
print("            KEEP 'hispanic' (orthogonal - Hispanic ethnicity spans all races)")
print("  INTERPRETATION: race coefficients = effect vs. White reference group\n")

# --- MARITAL STATUS ---
print("MARITAL STATUS:")
marital_cols = ['married', 'divorced', 'never_married', 'widowed']
print(f"  Sum: {df[marital_cols].sum(axis=1).mean():.4f}")
for col in marital_cols:
    print(f"    {col}: {df[col].mean():.4f}")
print("  DECISION: DROP 'married' (largest) as reference\n")

# --- EDUCATION ---
print("EDUCATION:")
edu_base = ['education_less_highschool', 'education_highschool', 'education_some_college',
            'education_bachelors', 'education_graduate']
print(f"  Sum of base categories: {df[edu_base].sum(axis=1).mean():.4f}")
for col in edu_base:
    print(f"    {col}: {df[col].mean():.4f}")
print(f"  education_college_or_above (derived): {df['education_college_or_above'].mean():.4f}")
print(f"    Correlation with bachelors+graduate: {df['education_college_or_above'].corr(df['education_bachelors']+df['education_graduate']):.4f}")
print("  DECISION: DROP 'education_highschool' (largest) as reference")
print("            DROP 'education_college_or_above' (derived)")
print("            DROP 'education_stem_degree' (subset of college)\n")

# --- INCOME ---
print("INCOME:")
print("  Multiple representations available:")
print("    - income_household_median (continuous summary)")
print("    - Income brackets (compositional, sum to ~1)")
print("    - income_household_six_figure (derived threshold)")
income_brackets = ['income_household_under_5', 'income_household_5_to_10', 
                   'income_household_10_to_15', 'income_household_15_to_20',
                   'income_household_20_to_25', 'income_household_25_to_35',
                   'income_household_35_to_50', 'income_household_50_to_75',
                   'income_household_75_to_100', 'income_household_100_to_150',
                   'income_household_150_over']
print(f"  Income brackets sum: {df[income_brackets].sum(axis=1).mean():.4f}")
print(f"  Median corr with brackets: highly multicollinear")
print("  DECISION: KEEP 'income_household_median' only (parsimonious)")
print("            DROP all brackets and derived measures\n")

# --- GEOGRAPHY ---
print("GEOGRAPHY/URBANICITY:")
print(f"  UrbanCore: {df['UrbanCore'].mean():.4f} (prop. in urban core)")
print(f"  density: {df['density'].mean():.1f} (pop per sq mile)")
print(f"  Correlation: {df['UrbanCore'].corr(df['density']):.4f}")
print("  DECISION: KEEP both (moderate correlation, distinct constructs)")
print("            DROP lat, lng (coordinates not meaningful as linear terms)")
print("            DROP zip5_count (administrative artifact)\n")

#=============================================================================
# PART 3: FINAL VARIABLE LIST
#=============================================================================

final_vars = [
    # Key clinical predictor
    't2d_obesity_pat_count',
    
    # Gender (ref: female)
    'male',
    
    # Age distribution
    'age_under_10', 'age_10_to_19', 'age_20s', 'age_30s', 'age_40s', 
    'age_50s', 'age_60s', 'age_70s', 'age_over_80',
    
    # Race/Ethnicity (ref: race_white)
    'race_black', 'race_asian', 'race_native', 'race_pacific', 
    'race_other', 'race_multiple', 'hispanic',
    
    # Marital Status (ref: married)
    'divorced', 'never_married', 'widowed',
    
    # Education (ref: education_highschool)
    'education_less_highschool', 'education_some_college', 
    'education_bachelors', 'education_graduate',
    
    # Income
    'income_household_median',
    
    # Employment
    'labor_force_participation', 'unemployment_rate',
    
    # Housing
    'home_ownership', 'home_value', 'rent_median', 'rent_burden',
    
    # Family
    'family_size', 'family_dual_income',
    
    # Health & Vulnerability
    'disabled', 'poverty', 'health_uninsured', 'limited_english',
    
    # Other demographics
    'commute_time', 'veteran',
    
    # Geography
    'UrbanCore', 'density'
]

print("="*90)
print(f"FINAL SELECTED VARIABLES: {len(final_vars)} predictors")
print("="*90)
print("\nReference categories (dropped to avoid perfect multicollinearity):")
print("  - Gender: female")
print("  - Race: race_white")
print("  - Marital: married")
print("  - Education: education_highschool")

#=============================================================================
# PART 4: DATA PREPARATION
#=============================================================================

model_df = df[['total_GLP_pat', 'population'] + final_vars].copy()
print(f"\nData preparation:")
print(f"  Original rows: {len(model_df)}")
model_df = model_df.dropna()
print(f"  After dropping missing: {len(model_df)}")
model_df = model_df[model_df['population'] > 0]
print(f"  After removing zero population: {len(model_df)}")

#=============================================================================
# PART 5: VIF CALCULATION
#=============================================================================

print("\n" + "="*90)
print("PART 3: VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
print("="*90)

def calculate_vif(X):
    """Calculate VIF using OLS regression for each variable"""
    vif_values = []
    for i in range(X.shape[1]):
        y_i = X.iloc[:, i]
        X_others = X.drop(X.columns[i], axis=1)
        X_others = np.column_stack([np.ones(len(X_others)), X_others])
        
        # OLS: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_others, y_i, rcond=None)[0]
            y_pred = X_others @ beta
            ss_res = np.sum((y_i - y_pred)**2)
            ss_tot = np.sum((y_i - y_i.mean())**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except:
            vif = np.inf
        vif_values.append(vif)
    
    return pd.DataFrame({'Variable': X.columns, 'VIF': vif_values}).sort_values('VIF', ascending=False)

# Standardize predictors for VIF
X = model_df[final_vars].copy()
X_std = (X - X.mean()) / X.std()

print("\nInitial VIF values:")
vif_df = calculate_vif(X_std)
print(vif_df.head(20).to_string(index=False))

# Iteratively remove high VIF variables
current_vars = final_vars.copy()
removed_for_vif = []

while True:
    X_current = model_df[current_vars]
    X_std = (X_current - X_current.mean()) / X_current.std()
    vif_df = calculate_vif(X_std)
    
    max_vif = vif_df['VIF'].max()
    if max_vif <= 10 or len(current_vars) < 5:
        break
    
    var_to_remove = vif_df.iloc[0]['Variable']
    removed_for_vif.append((var_to_remove, max_vif))
    current_vars.remove(var_to_remove)

if removed_for_vif:
    print(f"\n\nVariables removed due to VIF > 10:")
    for var, vif_val in removed_for_vif:
        print(f"  - {var}: VIF = {vif_val:.2f}")

print(f"\nFinal model: {len(current_vars)} variables")

# Final VIF
X_final = model_df[current_vars]
X_std = (X_final - X_final.mean()) / X_final.std()
vif_final = calculate_vif(X_std)
print("\nFinal VIF values (all ≤ 10):")
print(vif_final.to_string(index=False))

# Save for model
model_vars = current_vars
