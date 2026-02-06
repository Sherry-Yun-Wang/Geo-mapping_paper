import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('IQVIA_with_zip3_characteristics.csv')

print("="*90)
print("RANDOM FOREST ANALYSIS FOR GLP-1 PRESCRIBING")
print("="*90)

print("""
================================================================================
METHODOLOGY - RANDOM FOREST
================================================================================

Random Forest is an ensemble method that:
1. Builds many decision trees on random subsets of data and features
2. Averages predictions across trees (reduces overfitting vs single tree)
3. Handles multicollinearity naturally
4. Captures non-linear relationships and interactions
5. Provides robust feature importance measures

KEY DIFFERENCES FROM GRADIENT BOOSTING:
- RF: Trees built independently (parallel) → less prone to overfitting
- GB: Trees built sequentially (boosting) → often better accuracy
- RF: More robust, fewer hyperparameters to tune
- GB: Can achieve better performance with careful tuning

We model log(rate) = log(GLP_count / population) as the target.
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

print(f"Observations: {len(model_df)}, Features: {len(feature_cols)}")

#=============================================================================
# RANDOM FOREST WITH CROSS-VALIDATION
#=============================================================================

print("\n" + "="*90)
print("RANDOM FOREST MODEL FITTING")
print("="*90)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nPerforming randomized search with 5-fold CV (50 iterations)...")
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    rf_base, param_grid, n_iter=50, cv=cv, 
    scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)
random_search.fit(X, y)

best_params = random_search.best_params_
print(f"\nBest parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"\nBest CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")

# Fit best model
rf_best = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
rf_best.fit(X, y)

cv_r2 = cross_val_score(rf_best, X, y, cv=cv, scoring='r2')
print(f"Cross-validation R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")

#=============================================================================
# FEATURE IMPORTANCE - MULTIPLE METHODS
#=============================================================================

print("\n" + "="*90)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*90)

# 1. Impurity-based (MDI - Mean Decrease in Impurity)
print("\n1. IMPURITY-BASED IMPORTANCE (Mean Decrease in Impurity):")
print("-" * 60)

imp_mdi = pd.DataFrame({
    'Feature': feature_names,
    'MDI_Importance': rf_best.feature_importances_
}).sort_values('MDI_Importance', ascending=False)

for _, row in imp_mdi.head(15).iterrows():
    bar = "█" * int(row['MDI_Importance'] * 80)
    print(f"{row['Feature']:<35} {row['MDI_Importance']:.4f} {bar}")

# 2. Permutation importance (more reliable)
print("\n2. PERMUTATION IMPORTANCE (more reliable):")
print("-" * 60)
print("Computing permutation importance...")

perm = permutation_importance(rf_best, X, y, n_repeats=30, random_state=42, n_jobs=-1)
imp_perm = pd.DataFrame({
    'Feature': feature_names,
    'Perm_Mean': perm.importances_mean,
    'Perm_Std': perm.importances_std
}).sort_values('Perm_Mean', ascending=False)

print(f"\n{'Feature':<35} {'Mean':>12} {'Std':>10}")
print("-" * 60)
for _, row in imp_perm.head(20).iterrows():
    print(f"{row['Feature']:<35} {row['Perm_Mean']:>12.4f} {row['Perm_Std']:>10.4f}")

#=============================================================================
# DIRECTION OF EFFECTS
#=============================================================================

print("\n" + "="*90)
print("DIRECTION OF EFFECTS (PARTIAL DEPENDENCE)")
print("="*90)

top_features = imp_perm.head(15)['Feature'].tolist()
print(f"\n{'Feature':<35} {'Direction':>12} {'Effect Size':>12}")
print("-" * 65)

directions = []
for feat in top_features:
    idx = feature_names.index(feat)
    pdp = partial_dependence(rf_best, X, features=[idx], grid_resolution=20)
    avg = pdp['average'][0]
    
    low = avg[:len(avg)//3].mean()
    high = avg[-len(avg)//3:].mean()
    diff = high - low
    
    if diff > 0.02:
        direction = "Positive ↑"
    elif diff < -0.02:
        direction = "Negative ↓"
    else:
        direction = "Weak/Nonlin"
    
    directions.append({'Feature': feat, 'Direction': direction, 'Effect': diff})
    print(f"{feat:<35} {direction:>12} {diff:>+12.4f}")

#=============================================================================
# COMPARISON: RANDOM FOREST VS GRADIENT BOOSTING
#=============================================================================

print("\n" + "="*90)
print("COMPARISON: RANDOM FOREST VS GRADIENT BOOSTING VS LINEAR")
print("="*90)

# Gradient Boosting (best from previous)
gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    min_samples_leaf=10, subsample=0.8, random_state=42
)
gb_cv_r2 = cross_val_score(gb, X, y, cv=cv, scoring='r2')

# Ridge
X_scaled = StandardScaler().fit_transform(X)
ridge_cv_r2 = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=cv, scoring='r2')

print(f"\n{'Model':<25} {'CV R² Mean':>12} {'CV R² Std':>12}")
print("-" * 55)
print(f"{'Ridge Regression':<25} {ridge_cv_r2.mean():>12.4f} {ridge_cv_r2.std():>12.4f}")
print(f"{'Random Forest':<25} {cv_r2.mean():>12.4f} {cv_r2.std():>12.4f}")
print(f"{'Gradient Boosting':<25} {gb_cv_r2.mean():>12.4f} {gb_cv_r2.std():>12.4f}")

#=============================================================================
# OUT-OF-BAG ERROR (UNIQUE TO RF)
#=============================================================================

print("\n" + "="*90)
print("OUT-OF-BAG (OOB) ANALYSIS")
print("="*90)

rf_oob = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1, oob_score=True)
rf_oob.fit(X, y)
print(f"\nOOB R² Score: {rf_oob.oob_score_:.4f}")
print("(OOB uses samples not included in each tree's bootstrap - natural validation)")

#=============================================================================
# MODEL PERFORMANCE SUMMARY
#=============================================================================

print("\n" + "="*90)
print("MODEL PERFORMANCE SUMMARY")
print("="*90)

y_pred = rf_best.predict(X)
r2_train = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
rmse_train = np.sqrt(np.mean((y - y_pred)**2))

print(f"\nTraining: R² = {r2_train:.4f}, RMSE = {rmse_train:.4f}")
print(f"OOB R²: {rf_oob.oob_score_:.4f}")
print(f"CV R² (5-fold): {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")

#=============================================================================
# FEATURE IMPORTANCE CONSISTENCY CHECK
#=============================================================================

print("\n" + "="*90)
print("FEATURE IMPORTANCE CONSISTENCY (RF vs GB)")
print("="*90)

# Load GB importance
gb.fit(X, y)
gb_perm = permutation_importance(gb, X, y, n_repeats=30, random_state=42, n_jobs=-1)
gb_imp = pd.DataFrame({
    'Feature': feature_names,
    'GB_Importance': gb_perm.importances_mean
}).sort_values('GB_Importance', ascending=False)

# Merge
comparison = imp_perm[['Feature', 'Perm_Mean']].merge(
    gb_imp, on='Feature'
).rename(columns={'Perm_Mean': 'RF_Importance'})

comparison['RF_Rank'] = comparison['RF_Importance'].rank(ascending=False)
comparison['GB_Rank'] = comparison['GB_Importance'].rank(ascending=False)
comparison = comparison.sort_values('RF_Importance', ascending=False)

print(f"\n{'Feature':<30} {'RF Imp':>10} {'RF Rank':>8} {'GB Imp':>10} {'GB Rank':>8}")
print("-" * 75)
for _, row in comparison.head(15).iterrows():
    print(f"{row['Feature']:<30} {row['RF_Importance']:>10.4f} {int(row['RF_Rank']):>8} {row['GB_Importance']:>10.4f} {int(row['GB_Rank']):>8}")

# Rank correlation
from scipy.stats import spearmanr
corr, pval = spearmanr(comparison['RF_Rank'], comparison['GB_Rank'])
print(f"\nSpearman rank correlation between RF and GB importance: {corr:.3f} (p={pval:.4f})")

#=============================================================================
# SAVE RESULTS
#=============================================================================

# Combine all importance measures
final_imp = imp_mdi.merge(imp_perm, on='Feature')
final_imp = final_imp.merge(gb_imp, on='Feature')
final_imp['Direction'] = final_imp['Feature'].map({d['Feature']: d['Direction'] for d in directions})
final_imp['Effect_Size'] = final_imp['Feature'].map({d['Feature']: d['Effect'] for d in directions})
final_imp = final_imp.sort_values('Perm_Mean', ascending=False)
final_imp.to_csv('rf_feature_importance.csv', index=False)

# Model summary
summary = [
    ['N_observations', len(model_df)],
    ['N_features', len(feature_cols)],
    ['Best_n_estimators', best_params['n_estimators']],
    ['Best_max_depth', best_params['max_depth']],
    ['Best_min_samples_leaf', best_params['min_samples_leaf']],
    ['Best_max_features', best_params['max_features']],
    ['CV_R2_mean', f"{cv_r2.mean():.4f}"],
    ['CV_R2_std', f"{cv_r2.std():.4f}"],
    ['OOB_R2', f"{rf_oob.oob_score_:.4f}"],
    ['Train_R2', f"{r2_train:.4f}"],
    ['GB_CV_R2', f"{gb_cv_r2.mean():.4f}"],
    ['Ridge_CV_R2', f"{ridge_cv_r2.mean():.4f}"],
    ['RF_GB_Rank_Correlation', f"{corr:.4f}"]
]
pd.DataFrame(summary, columns=['Metric', 'Value']).to_csv('rf_model_summary.csv', index=False)

print("\nSaved: rf_feature_importance.csv, rf_model_summary.csv")
