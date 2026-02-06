import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.special import gammaln, digamma
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('IQVIA_with_zip3_characteristics.csv')

print("="*90)
print("GENERALIZED ADDITIVE MODELS (GAMs) FOR GLP-1 PRESCRIBING")
print("="*90)

print("""
================================================================================
METHODOLOGY - GENERALIZED ADDITIVE MODELS (GAMs)
================================================================================

GAMs extend GLMs by allowing non-linear relationships through smooth functions:

    g(E[Y]) = β₀ + f₁(X₁) + f₂(X₂) + ... + fₚ(Xₚ)

Where each f() is a smooth function (typically spline) estimated from the data.

KEY ADVANTAGES:
1. Captures non-linear relationships while maintaining interpretability
2. Can visualize how each predictor affects the outcome
3. Automatic smoothness selection via cross-validation
4. More interpretable than tree-based methods
5. Handles multicollinearity better than standard GLM

IMPLEMENTATION:
Since pygam is not available, we implement GAM using:
- B-spline basis expansion for each predictor
- Ridge regularization to prevent overfitting
- Cross-validation for smoothness parameter selection

This is equivalent to a penalized regression GAM.
""")

# Data prep
exclude = ['zip3', 'total_GLP_pat', 'population', 'state_id', 'state_name', 
           'zip5_count', 'lat', 'lng', 'housing_units']
feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

model_df = df[['total_GLP_pat', 'population'] + feature_cols].dropna()
model_df = model_df[model_df['population'] > 0].reset_index(drop=True)
model_df['log_rate'] = np.log((model_df['total_GLP_pat'] + 0.5) / model_df['population'])

X = model_df[feature_cols].values
y = model_df['log_rate'].values
feature_names = feature_cols
n_obs, n_features = X.shape

print(f"Observations: {n_obs}, Features: {n_features}")

#=============================================================================
# SPLINE-BASED GAM IMPLEMENTATION
#=============================================================================

print("\n" + "="*90)
print("GAM MODEL FITTING")
print("="*90)

# Select top features based on previous analyses to keep model manageable
# Using top 20 features from RF/GB importance
top_features = [
    'health_uninsured', 'education_bachelors', 'home_value', 'UrbanCore',
    'race_native', 't2d_obesity_pat_count', 'labor_force_participation',
    'self_employed', 'education_highschool', 'hispanic', 'farmer',
    'charitable_givers', 'veteran', 'race_black', 'rent_burden',
    'age_over_18', 'disabled', 'commute_time', 'density', 'unemployment_rate'
]

# Filter to available features
top_features = [f for f in top_features if f in feature_names]
print(f"\nUsing top {len(top_features)} features for GAM")

X_gam = model_df[top_features].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_gam)

# Create spline basis for each feature
print("\nCreating spline basis functions...")
n_knots = 5  # Number of knots per feature (df = n_knots + degree - 1)
degree = 3   # Cubic splines

# Build spline-expanded design matrix
spline_transformer = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)

X_spline_list = []
feature_spline_names = []

for i, feat in enumerate(top_features):
    x_feat = X_scaled[:, i:i+1]
    spline_trans = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
    x_spline = spline_trans.fit_transform(x_feat)
    X_spline_list.append(x_spline)
    
    for j in range(x_spline.shape[1]):
        feature_spline_names.append(f"{feat}_s{j+1}")

X_spline = np.hstack(X_spline_list)
print(f"Spline basis matrix shape: {X_spline.shape}")
print(f"  (Original: {len(top_features)} features → {X_spline.shape[1]} spline basis functions)")

#=============================================================================
# CROSS-VALIDATION FOR REGULARIZATION PARAMETER
#=============================================================================

print("\n" + "="*90)
print("CROSS-VALIDATION FOR SMOOTHING PARAMETER")
print("="*90)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv_scores = []

print(f"\n{'Alpha':<12} {'CV R²':>12} {'CV Std':>12}")
print("-" * 40)

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_spline, y, cv=cv, scoring='r2')
    cv_scores.append((alpha, scores.mean(), scores.std()))
    print(f"{alpha:<12} {scores.mean():>12.4f} {scores.std():>12.4f}")

best_alpha = max(cv_scores, key=lambda x: x[1])[0]
best_r2 = max(cv_scores, key=lambda x: x[1])[1]
print(f"\nBest alpha: {best_alpha} (CV R² = {best_r2:.4f})")

#=============================================================================
# FIT FINAL GAM MODEL
#=============================================================================

print("\n" + "="*90)
print("FINAL GAM MODEL")
print("="*90)

gam_model = Ridge(alpha=best_alpha)
gam_model.fit(X_spline, y)

# Predictions
y_pred = gam_model.predict(X_spline)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_train = 1 - ss_res / ss_tot

cv_r2 = cross_val_score(gam_model, X_spline, y, cv=cv, scoring='r2')

print(f"\nModel Performance:")
print(f"  Training R²: {r2_train:.4f}")
print(f"  CV R² (5-fold): {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")

#=============================================================================
# EXTRACT SMOOTH FUNCTION EFFECTS
#=============================================================================

print("\n" + "="*90)
print("SMOOTH FUNCTION EFFECTS (PARTIAL EFFECTS)")
print("="*90)

# For each feature, compute the partial effect curve
n_spline_per_feat = X_spline.shape[1] // len(top_features)

partial_effects = {}

print(f"\n{'Feature':<30} {'Effect Range':>15} {'Direction':>12} {'Nonlinearity':>12}")
print("-" * 75)

for i, feat in enumerate(top_features):
    # Get spline coefficients for this feature
    start_idx = i * n_spline_per_feat
    end_idx = (i + 1) * n_spline_per_feat
    
    # Create grid of values for this feature
    x_grid = np.linspace(X_scaled[:, i].min(), X_scaled[:, i].max(), 100)
    
    # Create spline basis for grid
    spline_trans = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
    spline_trans.fit(X_scaled[:, i:i+1])
    x_grid_spline = spline_trans.transform(x_grid.reshape(-1, 1))
    
    # Compute partial effect
    coefs = gam_model.coef_[start_idx:end_idx]
    partial_effect = x_grid_spline @ coefs
    
    # Center the effect
    partial_effect = partial_effect - partial_effect.mean()
    
    # Store
    partial_effects[feat] = {
        'x_grid': x_grid,
        'effect': partial_effect,
        'effect_range': partial_effect.max() - partial_effect.min()
    }
    
    # Determine direction
    low_effect = partial_effect[:30].mean()
    high_effect = partial_effect[-30:].mean()
    diff = high_effect - low_effect
    
    if diff > 0.05:
        direction = "Positive ↑"
    elif diff < -0.05:
        direction = "Negative ↓"
    else:
        direction = "Weak"
    
    # Assess nonlinearity (compare to linear fit)
    linear_fit = np.polyfit(x_grid, partial_effect, 1)
    linear_pred = np.polyval(linear_fit, x_grid)
    nonlin_residual = np.std(partial_effect - linear_pred)
    
    if nonlin_residual > 0.05:
        nonlin = "Nonlinear"
    else:
        nonlin = "Linear"
    
    partial_effects[feat]['direction'] = direction
    partial_effects[feat]['nonlinearity'] = nonlin
    partial_effects[feat]['diff'] = diff
    
    print(f"{feat:<30} {partial_effects[feat]['effect_range']:>15.4f} {direction:>12} {nonlin:>12}")

#=============================================================================
# FEATURE IMPORTANCE BASED ON EFFECT RANGE
#=============================================================================

print("\n" + "="*90)
print("FEATURE IMPORTANCE (BY EFFECT RANGE)")
print("="*90)

importance_df = pd.DataFrame([
    {
        'Feature': feat,
        'Effect_Range': partial_effects[feat]['effect_range'],
        'Direction': partial_effects[feat]['direction'],
        'Nonlinearity': partial_effects[feat]['nonlinearity'],
        'Effect_Diff': partial_effects[feat]['diff']
    }
    for feat in top_features
]).sort_values('Effect_Range', ascending=False)

print(f"\n{'Rank':<6} {'Feature':<30} {'Effect Range':>12} {'Direction':>12} {'Shape':>12}")
print("-" * 80)
for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{rank:<6} {row['Feature']:<30} {row['Effect_Range']:>12.4f} {row['Direction']:>12} {row['Nonlinearity']:>12}")

#=============================================================================
# COMPARISON WITH LINEAR AND TREE MODELS
#=============================================================================

print("\n" + "="*90)
print("MODEL COMPARISON")
print("="*90)

# Linear model on same features
X_linear = model_df[top_features].values
X_linear_scaled = StandardScaler().fit_transform(X_linear)
ridge_linear = Ridge(alpha=1.0)
linear_cv_r2 = cross_val_score(ridge_linear, X_linear_scaled, y, cv=cv, scoring='r2')

# GAM (spline)
gam_cv_r2 = cv_r2

print(f"\n{'Model':<30} {'CV R²':>12} {'±Std':>10} {'Features':>10}")
print("-" * 70)
print(f"{'Linear (Ridge)':<30} {linear_cv_r2.mean():>12.4f} {linear_cv_r2.std():>10.4f} {len(top_features):>10}")
print(f"{'GAM (Spline + Ridge)':<30} {gam_cv_r2.mean():>12.4f} {gam_cv_r2.std():>10.4f} {len(top_features):>10}")

improvement = (gam_cv_r2.mean() - linear_cv_r2.mean()) * 100
print(f"\nGAM improvement over linear: {improvement:+.1f} percentage points")
print("(Improvement indicates non-linear relationships in the data)")

#=============================================================================
# IDENTIFY MOST NONLINEAR RELATIONSHIPS
#=============================================================================

print("\n" + "="*90)
print("FEATURES WITH STRONGEST NONLINEAR EFFECTS")
print("="*90)

nonlinear_features = importance_df[importance_df['Nonlinearity'] == 'Nonlinear'].head(10)
print("\nFeatures with detected nonlinear relationships:")
for _, row in nonlinear_features.iterrows():
    print(f"  - {row['Feature']}: effect range = {row['Effect_Range']:.3f}, {row['Direction']}")

#=============================================================================
# SAVE RESULTS
#=============================================================================

# Save feature importance
importance_df.to_csv('gam_feature_importance.csv', index=False)

# Save model summary
summary = [
    ['N_observations', n_obs],
    ['N_features', len(top_features)],
    ['N_spline_basis', X_spline.shape[1]],
    ['N_knots', n_knots],
    ['Spline_degree', degree],
    ['Best_alpha', best_alpha],
    ['Train_R2', f"{r2_train:.4f}"],
    ['CV_R2_mean', f"{gam_cv_r2.mean():.4f}"],
    ['CV_R2_std', f"{gam_cv_r2.std():.4f}"],
    ['Linear_CV_R2', f"{linear_cv_r2.mean():.4f}"],
    ['GAM_improvement', f"{improvement:.2f}%"]
]
pd.DataFrame(summary, columns=['Metric', 'Value']).to_csv('gam_model_summary.csv', index=False)

# Save partial effects data for plotting
effects_data = []
for feat in top_features:
    pe = partial_effects[feat]
    for x, eff in zip(pe['x_grid'], pe['effect']):
        effects_data.append({'Feature': feat, 'X_scaled': x, 'Partial_Effect': eff})
pd.DataFrame(effects_data).to_csv('gam_partial_effects.csv', index=False)

print("\nSaved: gam_feature_importance.csv, gam_model_summary.csv, gam_partial_effects.csv")
