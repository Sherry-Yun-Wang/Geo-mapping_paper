import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.special import gammaln, digamma, polygamma
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('IQVIA_with_zip3_characteristics.csv')

# Final variables after VIF screening
model_vars = [
    't2d_obesity_pat_count',
    'male',
    'age_under_10', 'age_10_to_19', 'age_30s', 'age_40s', 
    'age_50s', 'age_60s', 'age_over_80',
    'race_black', 'race_asian', 'race_native', 'race_pacific', 
    'race_other', 'race_multiple',
    'divorced',
    'education_less_highschool', 'education_some_college', 'education_graduate',
    'home_ownership', 'home_value', 'rent_burden',
    'family_size', 'family_dual_income',
    'disabled', 'health_uninsured', 'limited_english',
    'commute_time', 'veteran',
    'UrbanCore', 'density',
    'unemployment_rate'
]

# Prepare data
model_df = df[['total_GLP_pat', 'population'] + model_vars].copy().dropna()
model_df = model_df[model_df['population'] > 0]

y = model_df['total_GLP_pat'].values
offset = np.log(model_df['population'].values)  # Log population offset
X = model_df[model_vars].values

# Standardize predictors for numerical stability
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1  # Avoid division by zero
X_scaled = (X - X_mean) / X_std

# Add intercept
X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
n_params = X_design.shape[1]

print("="*90)
print("PART 4: NEGATIVE BINOMIAL GLM ESTIMATION")
print("="*90)
print(f"\nModel specification:")
print(f"  Outcome: total_GLP_pat (count)")
print(f"  Distribution: Negative Binomial")
print(f"  Link function: log")
print(f"  Offset: log(population)")
print(f"  Observations: {len(y)}")
print(f"  Parameters: {n_params} (including intercept)")

#=============================================================================
# NEGATIVE BINOMIAL GLM IMPLEMENTATION
#=============================================================================

def neg_binomial_log_likelihood(params, y, X, offset):
    """
    Negative log-likelihood for NB2 model (variance = mu + mu^2/alpha)
    params = [beta_0, beta_1, ..., beta_p, log_alpha]
    """
    beta = params[:-1]
    alpha = np.exp(params[-1])  # Dispersion parameter (ensure positive)
    
    # Linear predictor: eta = X @ beta + offset
    eta = X @ beta + offset
    mu = np.exp(eta)  # Mean (inverse link)
    
    # Clip for numerical stability
    mu = np.clip(mu, 1e-10, 1e10)
    alpha = np.clip(alpha, 1e-10, 1e10)
    
    # NB2 log-likelihood
    # f(y|mu,alpha) = Gamma(y+alpha) / (Gamma(alpha) * Gamma(y+1)) * 
    #                 (alpha/(alpha+mu))^alpha * (mu/(alpha+mu))^y
    
    ll = (gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1) +
          alpha * np.log(alpha / (alpha + mu)) +
          y * np.log(mu / (alpha + mu)))
    
    return -np.sum(ll)  # Return negative for minimization

def neg_binomial_gradient(params, y, X, offset):
    """Gradient of negative log-likelihood"""
    beta = params[:-1]
    log_alpha = params[-1]
    alpha = np.exp(log_alpha)
    
    eta = X @ beta + offset
    mu = np.exp(eta)
    mu = np.clip(mu, 1e-10, 1e10)
    
    # Gradient w.r.t. beta
    grad_beta = -X.T @ (y - mu * (y + alpha) / (mu + alpha))
    
    # Gradient w.r.t. log_alpha (chain rule)
    grad_log_alpha = -np.sum(
        digamma(y + alpha) - digamma(alpha) + 
        np.log(alpha / (alpha + mu)) + 1 - (y + alpha) / (mu + alpha)
    ) * alpha
    
    return np.concatenate([grad_beta, [grad_log_alpha]])

# Initial values: Poisson estimates for beta, alpha=1
print("\nEstimation procedure:")
print("  1. Initialize with Poisson regression estimates")
print("  2. Optimize via L-BFGS-B algorithm")

# Simple Poisson initialization
eta_init = np.log((y + 0.5) / np.exp(offset))
beta_init = np.linalg.lstsq(X_design, eta_init, rcond=None)[0]
alpha_init = 1.0
params_init = np.concatenate([beta_init, [np.log(alpha_init)]])

# Optimize
result = optimize.minimize(
    neg_binomial_log_likelihood,
    params_init,
    args=(y, X_design, offset),
    method='L-BFGS-B',
    jac=neg_binomial_gradient,
    options={'maxiter': 1000, 'disp': False}
)

if not result.success:
    print(f"  Warning: Optimization may not have fully converged: {result.message}")

# Extract estimates
beta_hat = result.x[:-1]
alpha_hat = np.exp(result.x[-1])
log_lik = -result.fun

print(f"  Convergence: {result.success}")
print(f"  Log-likelihood: {log_lik:.2f}")
print(f"  Dispersion parameter (alpha): {alpha_hat:.4f}")

#=============================================================================
# STANDARD ERRORS (via observed Fisher Information)
#=============================================================================

print("\n  Computing standard errors via numerical Hessian...")

def compute_hessian(params, y, X, offset, eps=1e-5):
    """Compute Hessian numerically"""
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
    
    return (hessian + hessian.T) / 2  # Symmetrize

hessian = compute_hessian(result.x, y, X_design, offset)

try:
    cov_matrix = np.linalg.inv(hessian)
    se_scaled = np.sqrt(np.diag(cov_matrix))
    
    # Transform SE back to original scale
    se_beta = se_scaled[:-1] / np.concatenate([[1], X_std])  # Intercept + scaled vars
    se_alpha = se_scaled[-1] * alpha_hat  # Delta method for exp transform
except:
    print("  Warning: Could not compute standard errors (singular Hessian)")
    se_beta = np.full(len(beta_hat), np.nan)

# Transform coefficients back to original scale
beta_original = beta_hat.copy()
beta_original[0] = beta_hat[0] - np.sum(beta_hat[1:] * X_mean / X_std)  # Adjust intercept
beta_original[1:] = beta_hat[1:] / X_std

#=============================================================================
# RESULTS TABLE
#=============================================================================

print("\n" + "="*90)
print("PART 5: MODEL RESULTS")
print("="*90)

# Create results dataframe
var_names = ['(Intercept)'] + model_vars
results_df = pd.DataFrame({
    'Variable': var_names,
    'Coefficient': beta_original,
    'Std.Error': se_beta,
    'z-value': beta_original / se_beta,
    'p-value': 2 * (1 - stats.norm.cdf(np.abs(beta_original / se_beta))),
    'IRR': np.exp(beta_original),
    'IRR_95%_LCI': np.exp(beta_original - 1.96 * se_beta),
    'IRR_95%_UCI': np.exp(beta_original + 1.96 * se_beta)
})

# Add significance stars
def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '.'
    return ''

results_df['Sig'] = results_df['p-value'].apply(sig_stars)

print("\nNegative Binomial Regression Results")
print("-" * 90)
print(f"Dependent variable: total_GLP_pat")
print(f"Offset: log(population)")
print(f"Observations: {len(y)}")
print(f"Dispersion (alpha): {alpha_hat:.4f}")
print(f"Log-likelihood: {log_lik:.2f}")
print("-" * 90)

# Format for display
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', '{:.4f}'.format)

print("\nCoefficients (log scale) and Incidence Rate Ratios:")
print(results_df[['Variable', 'Coefficient', 'Std.Error', 'z-value', 'p-value', 'Sig', 'IRR', 'IRR_95%_LCI', 'IRR_95%_UCI']].to_string(index=False))

print("\n" + "-"*90)
print("Significance codes: '***' p<0.001, '**' p<0.01, '*' p<0.05, '.' p<0.1")

#=============================================================================
# MODEL FIT STATISTICS
#=============================================================================

print("\n" + "="*90)
print("PART 6: MODEL FIT STATISTICS")
print("="*90)

# Predicted values
eta_pred = X_design @ beta_hat + offset
mu_pred = np.exp(eta_pred)

# Pearson residuals
pearson_resid = (y - mu_pred) / np.sqrt(mu_pred + mu_pred**2 / alpha_hat)
pearson_chi2 = np.sum(pearson_resid**2)
pearson_dispersion = pearson_chi2 / (len(y) - n_params)

# Deviance residuals
def nb_deviance_residual(y, mu, alpha):
    """Calculate deviance residuals for NB model"""
    dev = np.where(
        y == 0,
        2 * alpha * np.log(alpha / (alpha + mu)),
        2 * (y * np.log(y / mu) - (y + alpha) * np.log((y + alpha) / (alpha + mu)))
    )
    return np.sign(y - mu) * np.sqrt(np.abs(dev))

dev_resid = nb_deviance_residual(y, mu_pred, alpha_hat)
deviance = np.sum(dev_resid**2)

# Null model (intercept only)
mu_null = np.mean(y)
ll_null = np.sum(gammaln(y + alpha_hat) - gammaln(alpha_hat) - gammaln(y + 1) +
                 alpha_hat * np.log(alpha_hat / (alpha_hat + mu_null)) +
                 y * np.log(mu_null / (alpha_hat + mu_null)))

# Pseudo R-squared (McFadden)
pseudo_r2 = 1 - (log_lik / ll_null)

# AIC and BIC
aic = -2 * log_lik + 2 * (n_params + 1)  # +1 for alpha
bic = -2 * log_lik + np.log(len(y)) * (n_params + 1)

print(f"\nNull deviance: {-2*ll_null:.2f} on {len(y)-1} df")
print(f"Residual deviance: {deviance:.2f} on {len(y)-n_params} df")
print(f"\nPearson Chi-squared: {pearson_chi2:.2f}")
print(f"Pearson dispersion: {pearson_dispersion:.4f}")
print(f"\nAIC: {aic:.2f}")
print(f"BIC: {bic:.2f}")
print(f"McFadden's Pseudo R²: {pseudo_r2:.4f}")

#=============================================================================
# SAVE RESULTS
#=============================================================================

results_df.to_csv('./glm_nb_results.csv', index=False)
print(f"\nResults saved to glm_nb_results.csv")
