import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.special import gammaln, digamma
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('IQVIA_with_zip3_characteristics.csv')

print("="*90)
print("DIMENSION REDUCTION: PCA + FACTOR ANALYSIS + NEGATIVE BINOMIAL GLM")
print("="*90)

print("""
================================================================================
METHODOLOGY
================================================================================
We used dimension reduction via factor analysis to address multicollinearity 
among ZIP-3 neighborhood characteristics.

APPROACH:
1. PCA to determine number of factors (Kaiser criterion: eigenvalue > 1)
2. Factor Analysis with Varimax rotation for interpretability  
3. Name factors based on high-loading variables
4. Use orthogonal factor scores as predictors (eliminates multicollinearity)
5. Include t2d_obesity_pat_count and UrbanCore as separate key predictors

ADVANTAGES:
- Reduces 66 correlated variables to ~10 orthogonal factors
- Factors are uncorrelated → no multicollinearity
- Each factor represents a meaningful neighborhood dimension
- More parsimonious model with better interpretability
""")

# Prepare data
exclude = ['zip3', 'total_GLP_pat', 'population', 'state_id', 'state_name', 
           'zip5_count', 'lat', 'lng', 'housing_units']
all_vars = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

model_df = df[['total_GLP_pat', 'population', 't2d_obesity_pat_count', 'UrbanCore'] + all_vars].dropna()
model_df = model_df[model_df['population'] > 0].reset_index(drop=True)

key_preds = ['t2d_obesity_pat_count', 'UrbanCore']
factor_vars = [v for v in all_vars if v not in key_preds]

X_factor = model_df[factor_vars].values
y = model_df['total_GLP_pat'].values
offset = np.log(model_df['population'].values)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_factor)
n_obs = len(y)

print(f"Observations: {n_obs}")
print(f"Variables for factor analysis: {len(factor_vars)}")

# PCA for determining number of factors
pca = PCA().fit(X_scaled)
eig = pca.explained_variance_
var_ratio = pca.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)

print("\n" + "="*90)
print("PCA SCREE ANALYSIS")
print("="*90)
print(f"\n{'PC':<5} {'Eigenvalue':<12} {'% Variance':<12} {'Cumulative %':<12} {'Retain (>1)':<10}")
print("-"*55)
for i in range(15):
    keep = "Yes" if eig[i] > 1 else "No"
    print(f"PC{i+1:<3} {eig[i]:<12.3f} {var_ratio[i]*100:<12.2f} {cum_var[i]*100:<12.2f} {keep}")

n_factors = sum(eig > 1)
n_factors = min(n_factors, 10)  # Cap at 10 for interpretability
print(f"\n→ Kaiser criterion: {sum(eig>1)} factors with eigenvalue > 1")
print(f"→ Extracting {n_factors} factors (explaining {cum_var[n_factors-1]*100:.1f}% of variance)")

# Factor Analysis with Varimax rotation
print("\n" + "="*90)
print("FACTOR ANALYSIS WITH VARIMAX ROTATION")
print("="*90)

fa = FactorAnalysis(n_components=n_factors, rotation='varimax', random_state=42)
F_scores = fa.fit_transform(X_scaled)

loadings_df = pd.DataFrame(fa.components_.T, index=factor_vars, 
                           columns=[f'F{i+1}' for i in range(n_factors)])

# Interpret and name factors
factor_names = []
factor_interpretations = []

print("\nFactor Loadings and Interpretation:")
print("-"*80)

for i in range(n_factors):
    col = f'F{i+1}'
    high_loadings = loadings_df[col][abs(loadings_df[col]) > 0.35].sort_values(key=abs, ascending=False)
    
    print(f"\n{col}:")
    for var, ld in high_loadings.head(6).items():
        sign = "+" if ld > 0 else "-"
        print(f"  {sign} {var}: {ld:.3f}")
    
    # Name based on top loading variables
    top_vars = high_loadings.head(3).index.tolist() if len(high_loadings) >= 3 else high_loadings.index.tolist()
    
    if any('income' in v.lower() for v in top_vars):
        if any('150' in v or 'six_figure' in v or 'median' in v for v in top_vars):
            name = "High_SES"
        else:
            name = "Middle_SES"
    elif any('age' in v.lower() for v in top_vars):
        if any('over_6' in v or 'over_8' in v or '70' in v for v in top_vars):
            name = "Elderly_Pop"
        elif any('under' in v or '20s' in v or '18_to' in v for v in top_vars):
            name = "Young_Pop"
        elif any('30s' in v or '40s' in v for v in top_vars):
            name = "Working_Age"
        else:
            name = "Age_Structure"
    elif any('race' in v.lower() or 'hispanic' in v.lower() for v in top_vars):
        if any('hispanic' in v.lower() or 'limited_english' in v.lower() for v in top_vars):
            name = "Hispanic_Immigrant"
        else:
            name = "Racial_Composition"
    elif any('married' in v.lower() or 'family' in v.lower() for v in top_vars):
        name = "Family_Structure"
    elif any('male' in v.lower() or 'female' in v.lower() for v in top_vars):
        name = "Gender_Composition"
    else:
        name = f"Dimension_{i+1}"
    
    # Ensure unique names
    base_name = name
    counter = 2
    while name in factor_names:
        name = f"{base_name}_{counter}"
        counter += 1
    
    factor_names.append(name)
    factor_interpretations.append({
        'factor': name,
        'top_vars': "; ".join([f"{v} ({high_loadings[v]:.2f})" for v in high_loadings.head(5).index])
    })
    print(f"  → Interpretation: {name}")

print("\n" + "-"*80)
print("Factor Summary:")
for i, name in enumerate(factor_names):
    print(f"  F{i+1}: {name}")

# Negative Binomial GLM
print("\n" + "="*90)
print("NEGATIVE BINOMIAL GLM WITH FACTOR SCORES")
print("="*90)

# Prepare design matrix
X_key = model_df[key_preds].values
key_scaler = StandardScaler()
X_key_scaled = key_scaler.fit_transform(X_key)

X_design = np.column_stack([np.ones(n_obs), X_key_scaled, F_scores])
predictor_names = ['(Intercept)'] + key_preds + factor_names

print(f"\nModel specification:")
print(f"  Outcome: total_GLP_pat")
print(f"  Distribution: Negative Binomial")
print(f"  Link: log")
print(f"  Offset: log(population)")
print(f"  Predictors: {len(predictor_names)} (2 key variables + {n_factors} factors)")

# NB GLM functions
def nb_negloglik(params, y, X, offset):
    beta = params[:-1]
    alpha = np.exp(params[-1])
    eta = X @ beta + offset
    mu = np.clip(np.exp(eta), 1e-10, 1e10)
    ll = np.sum(gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1) +
                alpha * np.log(alpha / (alpha + mu)) + y * np.log(mu / (alpha + mu)))
    return -ll

def nb_gradient(params, y, X, offset):
    beta = params[:-1]
    alpha = np.exp(params[-1])
    eta = X @ beta + offset
    mu = np.clip(np.exp(eta), 1e-10, 1e10)
    grad_beta = -X.T @ (y - mu * (y + alpha) / (mu + alpha))
    grad_alpha = -np.sum(digamma(y + alpha) - digamma(alpha) + 
                         np.log(alpha / (alpha + mu)) + 1 - (y + alpha) / (mu + alpha)) * alpha
    return np.concatenate([grad_beta, [grad_alpha]])

# Initialize
eta_init = np.log((y + 0.5) / np.exp(offset))
beta_init = np.linalg.lstsq(X_design, eta_init, rcond=None)[0]
params_init = np.concatenate([beta_init, [0.0]])

# Fit
result = optimize.minimize(nb_negloglik, params_init, args=(y, X_design, offset),
                          method='L-BFGS-B', jac=nb_gradient, 
                          options={'maxiter': 2000, 'ftol': 1e-10})

beta = result.x[:-1]
alpha = np.exp(result.x[-1])
loglik = -result.fun

print(f"\n  Convergence: {result.success}")
print(f"  Log-likelihood: {loglik:.2f}")
print(f"  Dispersion (α): {alpha:.4f}")

# Compute standard errors via bootstrap (more robust than Hessian)
print("\n  Computing standard errors via bootstrap (100 iterations)...")
np.random.seed(42)
n_boot = 100
boot_betas = []

for b in range(n_boot):
    idx = np.random.choice(n_obs, n_obs, replace=True)
    y_b = y[idx]
    X_b = X_design[idx]
    off_b = offset[idx]
    
    try:
        res_b = optimize.minimize(nb_negloglik, result.x, args=(y_b, X_b, off_b),
                                  method='L-BFGS-B', jac=nb_gradient,
                                  options={'maxiter': 500, 'ftol': 1e-6})
        if res_b.success:
            boot_betas.append(res_b.x[:-1])
    except:
        pass

boot_betas = np.array(boot_betas)
se = np.std(boot_betas, axis=0)

# Results
z_vals = beta / se
p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))
irr = np.exp(beta)
irr_lo = np.exp(beta - 1.96 * se)
irr_hi = np.exp(beta + 1.96 * se)

def sig_stars(p):
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '.'
    return ''

print("\n" + "-"*100)
print("MODEL RESULTS")
print("-"*100)
print(f"{'Variable':<25} {'Coef':>10} {'SE':>10} {'z':>8} {'p':>10} {'':>4} {'IRR':>8} {'95% CI':>20}")
print("-"*100)

for i in range(len(predictor_names)):
    ci_str = f"[{irr_lo[i]:.3f}, {irr_hi[i]:.3f}]"
    p_str = f"{p_vals[i]:.4f}" if p_vals[i] >= 0.0001 else "<0.0001"
    print(f"{predictor_names[i]:<25} {beta[i]:>10.4f} {se[i]:>10.4f} {z_vals[i]:>8.2f} {p_str:>10} {sig_stars(p_vals[i]):>4} {irr[i]:>8.4f} {ci_str:>20}")

print("-"*100)
print("Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")

# Model fit statistics
mu_null = np.mean(y)
ll_null = np.sum(gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1) +
                 alpha * np.log(alpha / (alpha + mu_null)) + y * np.log(mu_null / (alpha + mu_null)))

n_params = len(beta) + 1
aic = -2 * loglik + 2 * n_params
bic = -2 * loglik + np.log(n_obs) * n_params
pseudo_r2 = 1 - loglik / ll_null

print("\n" + "="*90)
print("MODEL FIT STATISTICS")
print("="*90)
print(f"Log-likelihood: {loglik:.2f}")
print(f"AIC: {aic:.2f}")
print(f"BIC: {bic:.2f}")
print(f"McFadden's Pseudo R²: {pseudo_r2:.4f}")

# Save results
results_list = []
for i in range(len(predictor_names)):
    results_list.append({
        'Variable': predictor_names[i],
        'Coefficient': beta[i],
        'Std_Error': se[i],
        'z_value': z_vals[i],
        'p_value': p_vals[i],
        'Significance': sig_stars(p_vals[i]),
        'IRR': irr[i],
        'IRR_95_LCI': irr_lo[i],
        'IRR_95_UCI': irr_hi[i]
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv('glm_nb_factor_results.csv', index=False)

# Save factor loadings
loadings_df.columns = factor_names
loadings_df.to_csv('factor_loadings_matrix.csv')

# Save factor interpretation
pd.DataFrame(factor_interpretations).to_csv('factor_interpretation_summary.csv', index=False)

print("\nResults saved:")
print("  - glm_nb_factor_results.csv")
print("  - factor_loadings_matrix.csv")
print("  - factor_interpretation_summary.csv")
