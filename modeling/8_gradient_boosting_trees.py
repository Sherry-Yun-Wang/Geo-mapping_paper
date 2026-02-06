import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('IQVIA_with_zip3_characteristics.csv')

print("="*90)
print("GRADIENT BOOSTING TREES FOR GLP-1 PRESCRIBING ANALYSIS")
print("="*90)

print("""
================================================================================
METHODOLOGY
================================================================================
Gradient boosting is a powerful ML approach that:
- Handles multicollinearity naturally (tree-based, no coefficient estimation)
- Captures non-linear relationships and interactions automatically
- Provides feature importance rankings
- No need for manual variable selection

We model log(rate) = log(GLP_count / population) as the target variable.
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

# Grid search
print("\n" + "="*90)
print("MODEL FITTING WITH CROSS-VALIDATION")
print("="*90)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'min_samples_leaf': [5, 10],
    'subsample': [0.8]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, 
                           cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")
print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

gb_best = GradientBoostingRegressor(**best_params, random_state=42)
gb_best.fit(X, y)

cv_r2 = cross_val_score(gb_best, X, y, cv=cv, scoring='r2')
print(f"Cross-validation R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")

# Feature importance
print("\n" + "="*90)
print("FEATURE IMPORTANCE")
print("="*90)

# Impurity-based
imp_df = pd.DataFrame({'Feature': feature_names, 'Impurity_Importance': gb_best.feature_importances_})
imp_df = imp_df.sort_values('Impurity_Importance', ascending=False)

print("\nTop 20 by Impurity-Based Importance:")
print("-" * 55)
for _, row in imp_df.head(20).iterrows():
    bar = "█" * int(row['Impurity_Importance'] * 100)
    print(f"{row['Feature']:<35} {row['Impurity_Importance']:.4f} {bar}")

# Permutation importance
print("\nComputing permutation importance...")
perm = permutation_importance(gb_best, X, y, n_repeats=30, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    'Feature': feature_names,
    'Perm_Mean': perm.importances_mean,
    'Perm_Std': perm.importances_std
}).sort_values('Perm_Mean', ascending=False)

print("\nTop 20 by Permutation Importance:")
print("-" * 65)
print(f"{'Feature':<35} {'Mean':>12} {'Std':>10}")
print("-" * 65)
for _, row in perm_df.head(20).iterrows():
    print(f"{row['Feature']:<35} {row['Perm_Mean']:>12.4f} {row['Perm_Std']:>10.4f}")

# Direction of effects
print("\n" + "="*90)
print("DIRECTION OF EFFECTS (PARTIAL DEPENDENCE)")
print("="*90)

top_features = perm_df.head(15)['Feature'].tolist()
print(f"\n{'Feature':<35} {'Direction':>12} {'Effect':>12}")
print("-" * 65)

directions = []
for feat in top_features:
    idx = feature_names.index(feat)
    pdp_result = partial_dependence(gb_best, X, features=[idx], grid_resolution=20)
    avg = pdp_result['average'][0]
    
    low = avg[:len(avg)//3].mean()
    high = avg[-len(avg)//3:].mean()
    diff = high - low
    
    if diff > 0.02:
        direction = "Positive ↑"
    elif diff < -0.02:
        direction = "Negative ↓"
    else:
        direction = "Weak/Nonlinear"
    
    directions.append({'Feature': feat, 'Direction': direction, 'Effect': diff})
    print(f"{feat:<35} {direction:>12} {diff:>+12.4f}")

# Comparison with linear
print("\n" + "="*90)
print("COMPARISON WITH LINEAR MODEL")
print("="*90)

X_scaled = StandardScaler().fit_transform(X)
ridge_r2 = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=cv, scoring='r2')

print(f"\nRidge Regression CV R²: {ridge_r2.mean():.4f} (±{ridge_r2.std():.4f})")
print(f"Gradient Boosting CV R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")
print(f"Improvement: {(cv_r2.mean() - ridge_r2.mean())*100:+.1f} percentage points")

# Model performance
print("\n" + "="*90)
print("MODEL PERFORMANCE SUMMARY")
print("="*90)

y_pred = gb_best.predict(X)
r2_train = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
rmse_train = np.sqrt(np.mean((y - y_pred)**2))

print(f"\nTraining: R² = {r2_train:.4f}, RMSE = {rmse_train:.4f}")
print(f"CV (5-fold): R² = {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")

# Save results
combined = imp_df.merge(perm_df, on='Feature').sort_values('Perm_Mean', ascending=False)
combined['Direction'] = combined['Feature'].map({d['Feature']: d['Direction'] for d in directions})
combined['Effect_Size'] = combined['Feature'].map({d['Feature']: d['Effect'] for d in directions})
combined.to_csv('gb_feature_importance.csv', index=False)

summary_data = [
    ['N_observations', len(model_df)],
    ['N_features', len(feature_cols)],
    ['Best_n_estimators', best_params['n_estimators']],
    ['Best_max_depth', best_params['max_depth']],
    ['Best_learning_rate', best_params['learning_rate']],
    ['CV_R2_mean', f"{cv_r2.mean():.4f}"],
    ['CV_R2_std', f"{cv_r2.std():.4f}"],
    ['CV_RMSE', f"{np.sqrt(-grid_search.best_score_):.4f}"],
    ['Train_R2', f"{r2_train:.4f}"],
    ['Ridge_CV_R2', f"{ridge_r2.mean():.4f}"]
]
pd.DataFrame(summary_data, columns=['Metric', 'Value']).to_csv('gb_model_summary.csv', index=False)

print("\nSaved: gb_feature_importance.csv, gb_model_summary.csv")
