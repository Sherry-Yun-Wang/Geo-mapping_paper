import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.special import gammaln, digamma
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('IQVIA_with_zip3_characteristics.csv')

print("="*90)
print("ELASTIC NET REGULARIZED NEGATIVE BINOMIAL REGRESSION")
print("="*90)

#=============================================================================
# PART 1: METHODOLOGY
#=============================================================================

print("""
================================================================================
PART 1: METHODOLOGY - ELASTIC NET REGULARIZATION
================================================================================

RATIONALE FOR ELASTIC NET OVER MANUAL VARIABLE SELECTION:

1. HANDLES MULTICOLLINEARITY AUTOMATICALLY
   - LASSO (L1): Arbitrarily selects ONE variable from correlated groups
   - Ridge (L2): Keeps all variables but shrinks coefficients
   - Elastic Net (L1+L2): Keeps GROUPS of correlated predictors together
   
2. PENALTY FUNCTION:
   Loss = -log L(β) + λ[α||β||₁ + (1-α)||β||₂²/2]
   
   Where:
   - λ (lambda): Overall regularization strength
   - α (alpha): Mixing parameter (0=Ridge, 1=LASSO, 0<α<1=Elastic Net)
   
3. WORKFLOW:
   Step 1: Include ALL predictors (no manual exclusion of reference categories)
   Step 2: Standardize predictors (mean=0, SD=1)
   Step 3: Fit Elastic Net with cross-validation to select optimal λ
   Step 4: Identify non-zero coefficients (selected variables)
   Step 5: Refit unpenalized Negative Binomial GLM with selected variables

4. ADVANTAGES:
   - Data-driven variable selection (reduces researcher degrees of freedom)
   - Handles high-dimensional, correlated predictors
   - Grouping effect: correlated predictors selected/excluded together
   - Produces sparse, interpretable models
""")

#=============================================================================
# PART 2: DATA PREPARATION - INCLUDE ALL PREDICTORS
#=============================================================================

# Include ALL neighborhood characteristics (no manual reference category dropping)
all_predictors = [
    # Key clinical predictor
    't2d_obesity_pat_count',
    
    # Gender (include BOTH - elastic net will handle)
    'male', 'female',
    
    # Age (include ALL brackets including derived)
    'age_under_10', 'age_10_to_19', 'age_20s', 'age_30s', 'age_40s', 
    'age_50s', 'age_60s', 'age_70s', 'age_over_80', 'age_median',
    'age_over_65', 'age_over_18', 'age_18_to_24',
    
    # Race/Ethnicity (include ALL)
    'race_white', 'race_black', 'race_asian', 'race_native', 'race_pacific', 
    'race_other', 'race_multiple', 'hispanic',
    
    # Marital Status (include ALL)
    'married', 'divorced', 'never_married', 'widowed',
    
    # Education (include ALL)
    'education_less_highschool', 'education_highschool', 'education_some_college', 
    'education_bachelors', 'education_graduate', 'education_college_or_above',
    'education_stem_degree',
    
    # Income (include median AND brackets)
    'income_household_median', 'income_individual_median',
    'income_household_under_5', 'income_household_5_to_10',
    'income_household_10_to_15', 'income_household_15_to_20',
    'income_household_20_to_25', 'income_household_25_to_35',
    'income_household_35_to_50', 'income_household_50_to_75',
    'income_household_75_to_100', 'income_household_100_to_150',
    'income_household_150_over', 'income_household_six_figure',
    
    # Employment
    'labor_force_participation', 'unemployment_rate',
    
    # Housing
    'home_ownership', 'home_value', 'rent_median', 'rent_burden',
    'housing_units', 'density',
    
    # Family
    'family_size', 'family_dual_income',
    
    # Health & Vulnerability
    'disabled', 'poverty', 'health_uninsured', 'limited_english',
    
    # Other
    'commute_time', 'veteran',
    
    # Geography
    'UrbanCore'
]

# Filter to existing columns
all_predictors = [c for c in all_predictors if c in df.columns]
print(f"\nTotal candidate predictors: {len(all_predictors)}")

# Prepare data
model_df = df[['total_GLP_pat', 'population'] + all_predictors].copy()
model_df = model_df.dropna()
model_df = model_df[model_df['population'] > 0]
print(f"Observations after removing missing/zero population: {len(model_df)}")

y = model_df['total_GLP_pat'].values
population = model_df['population'].values
offset = np.log(population)
X = model_df[all_predictors].values

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nPredictor matrix shape: {X_scaled.shape}")

#=============================================================================
# PART 3: ELASTIC NET VARIABLE SELECTION
#=============================================================================

print("\n" + "="*90)
print("PART 2: ELASTIC NET VARIABLE SELECTION VIA CROSS-VALIDATION")
print("="*90)

# Transform to approximate Gaussian for Elastic Net
# Use log(rate) = log(y+0.5) - log(population) as response
y_transformed = np.log(y + 0.5) - offset

# Elastic Net with cross-validation
# Test multiple alpha values (mixing parameter)
alphas_to_test = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
best_alpha = None
best_score = -np.inf
best_model = None

print("\nTesting Elastic Net mixing parameters (α):")
print("-" * 60)

for alpha in alphas_to_test:
    enet = ElasticNetCV(
        l1_ratio=alpha,  # This is the mixing parameter
        alphas=np.logspace(-4, 1, 100),  # Lambda values to test
        cv=10,
        max_iter=10000,
        random_state=42
    )
    enet.fit(X_scaled, y_transformed)
    
    # Score on full data
    score = enet.score(X_scaled, y_transformed)
    n_selected = np.sum(enet.coef_ != 0)
    
    print(f"  α={alpha:.2f}: λ={enet.alpha_:.6f}, R²={score:.4f}, Variables selected={n_selected}")
    
    if score > best_score:
        best_score = score
        best_alpha = alpha
        best_model = enet

print(f"\nBest mixing parameter: α = {best_alpha}")
print(f"Best λ (regularization): {best_model.alpha_:.6f}")

# Get selected variables
selected_mask = best_model.coef_ != 0
selected_vars = [all_predictors[i] for i in range(len(all_predictors)) if selected_mask[i]]
selected_coefs = best_model.coef_[selected_mask]

print(f"\nVariables selected by Elastic Net: {len(selected_vars)} / {len(all_predictors)}")

# Show Elastic Net coefficients (standardized)
print("\n" + "-"*60)
print("Elastic Net Coefficients (standardized scale):")
print("-"*60)
enet_results = pd.DataFrame({
    'Variable': selected_vars,
    'EN_Coefficient': selected_coefs
}).sort_values('EN_Coefficient', key=abs, ascending=False)
print(enet_results.to_string(index=False))

# Show excluded variables
excluded_vars = [all_predictors[i] for i in range(len(all_predictors)) if not selected_mask[i]]
print(f"\n\nVariables EXCLUDED by Elastic Net ({len(excluded_vars)}):")
for v in excluded_vars:
    print(f"  - {v}")

#=============================================================================
# PART 4: REFIT UNPENALIZED NEGATIVE BINOMIAL WITH SELECTED VARIABLES
#=============================================================================

print("\n" + "="*90)
print("PART 3: UNPENALIZED NEGATIVE BINOMIAL GLM WITH SELECTED VARIABLES")
print("="*90)

# Prepare data with selected variables only
X_selected = model_df[selected_vars].values
X_selected_scaled = (X_selected - X_selected.mean(axis=0)) / X_selected.std(axis=0)
X_design = np.column_stack([np.ones(len(X_selected_scaled)), X_selected_scaled])

# Negative Binomial functions (same as before)
def neg_binomial_log_likelihood(params, y, X, offset):
    beta = params[:-1]
    alpha = np.exp(params[-1])
    eta = X @ beta + offset
    mu = np.exp(eta)
    mu = np.clip(mu, 1e-10, 1e10)
    alpha = np.clip(alpha, 1e-10, 1e10)
    ll = (gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1) +
          alpha * np.log(alpha / (alpha + mu)) +
          y * np.log(mu / (alpha + mu)))
    return -np.sum(ll)

def neg_binomial_gradient(params, y, X, offset):
    beta = params[:-1]
    log_alpha = params[-1]
    alpha = np.exp(log_alpha)
    eta = X @ beta + offset
    mu = np.exp(eta)
    mu = np.clip(mu, 1e-10, 1e10)
    grad_beta = -X.T @ (y - mu * (y + alpha) / (mu + alpha))
    grad_log_alpha = -np.sum(
        digamma(y + alpha) - digamma(alpha) + 
        np.log(alpha / (alpha + mu)) + 1 - (y + alpha) / (mu + alpha)
    ) * alpha
    return np.concatenate([grad_beta, [grad_log_alpha]])

# Initialize and optimize
eta_init = np.log((y + 0.5) / np.exp(offset))
beta_init = np.linalg.lstsq(X_design, eta_init, rcond=None)[0]
params_init = np.concatenate([beta_init, [np.log(1.0)]])

result = optimize.minimize(
    neg_binomial_log_likelihood,
    params_init,
    args=(y, X_design, offset),
    method='L-BFGS-B',
    jac=neg_binomial_gradient,
    options={'maxiter': 1000}
)

beta_hat = result.x[:-1]
alpha_hat = np.exp(result.x[-1])
log_lik = -result.fun

print(f"\nModel specification:")
print(f"  Outcome: total_GLP_pat (count)")
print(f"  Distribution: Negative Binomial")
print(f"  Link function: log")
print(f"  Offset: log(population)")
print(f"  Observations: {len(y)}")
print(f"  Parameters: {len(selected_vars) + 1} (intercept + {len(selected_vars)} predictors)")
print(f"\n  Convergence: {result.success}")
print(f"  Log-likelihood: {log_lik:.2f}")
print(f"  Dispersion (α): {alpha_hat:.4f}")

# Compute standard errors
def compute_hessian(params, y, X, offset, eps=1e-5):
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad_plus = neg_binomial_gradient(params_plus, y, X, offset)
        grad_minus = neg_binomial_gradient(params_minus, y, X, offset)
        hessian[i, :] = (grad_plus - grad_minus) / (2 * eps)
    return (hessian + hessian.T) / 2

hessian = compute_hessian(result.x, y, X_design, offset)
try:
    cov_matrix = np.linalg.inv(hessian)
    se_scaled = np.sqrt(np.diag(cov_matrix)[:-1])  # Exclude alpha SE
except:
    se_scaled = np.full(len(beta_hat), np.nan)

# Transform back to original scale
X_mean = X_selected.mean(axis=0)
X_std = X_selected.std(axis=0)
beta_original = beta_hat.copy()
beta_original[0] = beta_hat[0] - np.sum(beta_hat[1:] * X_mean / X_std)
beta_original[1:] = beta_hat[1:] / X_std
se_original = np.concatenate([[se_scaled[0]], se_scaled[1:] / X_std])

# Results table
var_names = ['(Intercept)'] + selected_vars
results_df = pd.DataFrame({
    'Variable': var_names,
    'Coefficient': beta_original,
    'Std.Error': se_original,
    'z-value': beta_original / se_original,
    'p-value': 2 * (1 - stats.norm.cdf(np.abs(beta_original / se_original))),
    'IRR': np.exp(beta_original),
    'IRR_95%_LCI': np.exp(beta_original - 1.96 * se_original),
    'IRR_95%_UCI': np.exp(beta_original + 1.96 * se_original)
})

def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '.'
    return ''

results_df['Sig'] = results_df['p-value'].apply(sig_stars)

print("\n" + "-"*90)
print("NEGATIVE BINOMIAL REGRESSION RESULTS (ELASTIC NET SELECTED VARIABLES)")
print("-"*90)

pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 120)
print(results_df[['Variable', 'Coefficient', 'Std.Error', 'z-value', 'p-value', 'Sig', 'IRR', 'IRR_95%_LCI', 'IRR_95%_UCI']].to_string(index=False))

print("\n" + "-"*90)
print("Significance codes: '***' p<0.001, '**' p<0.01, '*' p<0.05, '.' p<0.1")

#=============================================================================
# PART 5: MODEL FIT STATISTICS
#=============================================================================

print("\n" + "="*90)
print("PART 4: MODEL FIT STATISTICS")
print("="*90)

eta_pred = X_design @ beta_hat + offset
mu_pred = np.exp(eta_pred)

# Null model
mu_null = np.mean(y)
ll_null = np.sum(gammaln(y + alpha_hat) - gammaln(alpha_hat) - gammaln(y + 1) +
                 alpha_hat * np.log(alpha_hat / (alpha_hat + mu_null)) +
                 y * np.log(mu_null / (alpha_hat + mu_null)))

# Deviance
def nb_deviance_residual(y, mu, alpha):
    dev = np.where(
        y == 0,
        2 * alpha * np.log(alpha / (alpha + mu)),
        2 * (y * np.log(y / mu) - (y + alpha) * np.log((y + alpha) / (alpha + mu)))
    )
    return np.sign(y - mu) * np.sqrt(np.abs(dev))

dev_resid = nb_deviance_residual(y, mu_pred, alpha_hat)
deviance = np.sum(dev_resid**2)

# Fit statistics
n_params = len(beta_hat) + 1  # +1 for alpha
aic = -2 * log_lik + 2 * n_params
bic = -2 * log_lik + np.log(len(y)) * n_params
pseudo_r2 = 1 - (log_lik / ll_null)

print(f"\nNull deviance: {-2*ll_null:.2f} on {len(y)-1} df")
print(f"Residual deviance: {deviance:.2f} on {len(y)-len(beta_hat)} df")
print(f"\nAIC: {aic:.2f}")
print(f"BIC: {bic:.2f}")
print(f"McFadden's Pseudo R²: {pseudo_r2:.4f}")

#=============================================================================
# SAVE RESULTS
#=============================================================================

results_df.to_csv('./glm_nb_elasticnet_results.csv', index=False)

# Save summary of variable selection
selection_summary = pd.DataFrame({
    'Variable': all_predictors,
    'Selected': ['Yes' if v in selected_vars else 'No' for v in all_predictors],
    'EN_Coefficient': [best_model.coef_[i] if best_model.coef_[i] != 0 else 0 for i in range(len(all_predictors))]
})
selection_summary.to_csv('./elasticnet_variable_selection.csv', index=False)

print(f"\nResults saved to:")
print(f"  - glm_nb_elasticnet_results.csv (final model coefficients)")
print(f"  - elasticnet_variable_selection.csv (variable selection details)")
