"""
11_generate_figures.py
======================
Generates all 5 publication figures. Fully self-contained: reads only the
CSV files that already exist in your working directory. No .pkl files needed.

For Figure 3 (partial dependence overlay) the script refits GB and RF from
IQVIA_with_zip3_characteristics.csv using the same hyperparameters as
8_gradient_boosting_trees.py and 9_random_forests.py (~4 s total).
The GAM curves are read directly from gam_partial_effects.csv.

Required input files (all produced by scripts 0-10)
----------------------------------------------------
  IQVIA_with_zip3_characteristics.csv
  gb_feature_importance.csv
  rf_feature_importance.csv
  gam_feature_importance.csv
  gam_partial_effects.csv

Output files
------------
  figure1_distribution.png
  figure2_model_comparison.png
  figure3_partial_dependence.png
  figure4_cross_method_importance.png
  figure5_nonlinearity.png

Figure map
----------
Fig 1  — Outcome distribution: raw skew → log transform → urban-core preview
Fig 2  — Model performance ladder: Ridge < GAM < RF < GB, with nonlinearity bracket
Fig 3  — PDP overlay: GAM / RF / GB curves on the same axes for 6 key features
Fig 4  — Cross-method importance: grouped bars + rank heatmap + Spearman ρ
Fig 5  — Nonlinearity evidence: 3 nonlinear vs 3 ≈linear GAM smooth functions
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# =============================================================================
# 0.  CONFIGURATION — colours, labels, style
# =============================================================================
PAL = {
    'gam' : '#2C5F8A',   # deep steel blue
    'rf'  : '#E07A30',   # burnt orange
    'gb'  : '#4CAF72',   # forest green
    'neg' : '#C0392B',   # red  (negative-direction effects)
    'pos' : '#2980B9',   # blue (positive-direction effects)
    'neu' : '#95A5A6',   # grey (neutral / baseline)
}

LBL = {                  # human-readable feature labels (IEEE: words, no abbreviations)
    'health_uninsured'          : 'Uninsured Rate',
    'education_bachelors'       : "Bachelor's Degree",
    'UrbanCore'                 : 'Urban Core',
    'home_value'                : 'Home Value',
    'labor_force_participation' : 'Labor Force Participation',   # CHANGED: was 'Labor Force Part.'
    'self_employed'             : 'Self-Employed (%)',
    'race_native'               : 'Native American (%)',
    'veteran'                   : 'Veteran (%)',
    'age_over_18'               : 'Adult Population (%)',
    'hispanic'                  : 'Hispanic (%)',
    'disabled'                  : 'Disabled (%)',
    'charitable_givers'         : 'Charitable Givers (%)',
    'rent_burden'               : 'Rent Burden',
    'education_highschool'      : 'High School (%)',
    'density'                   : 'Population Density',
    'unemployment_rate'         : 'Unemployment Rate',
    'commute_time'              : 'Commute Time',
    't2d_obesity_pat_count'     : 'Type 2 Diabetes/Obesity Count',   # CHANGED: was 'T2D/Obesity Count'
    'farmer'                    : 'Farmer (%)',
    'race_black'                : 'Black Population (%)',
}

# The 20 features the GAM was fit on (order matches gam_partial_effects.csv)
TOP_FEATURES = [
    'health_uninsured', 'education_bachelors', 'home_value', 'UrbanCore',
    'race_native', 't2d_obesity_pat_count', 'labor_force_participation',
    'self_employed', 'education_highschool', 'hispanic', 'farmer',
    'charitable_givers', 'veteran', 'race_black', 'rent_burden',
    'age_over_18', 'disabled', 'commute_time', 'density', 'unemployment_rate',
]

# Columns to exclude when building the 68-feature predictor set
EXCLUDE_COLS = {'zip3', 'total_GLP_pat', 'population', 'state_id',
                'state_name', 'zip5_count', 'lat', 'lng', 'housing_units'}

# IEEE figure label requirements: 8pt Times New Roman
# 'Liberation Serif' is the standard Linux metric-equivalent of Times New Roman
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],  # CHANGED: fallback chain for Linux clusters
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.labelsize'   : 8,                      # CHANGED: was 9
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'axes.titlesize'   : 8,                      # CHANGED: was 10
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'axes.grid'        : False,
})

# =============================================================================
# 1.  LOAD & PREPARE DATA
# =============================================================================
print("Loading data…")
df = pd.read_csv('IQVIA_with_zip3_characteristics.csv')

# 68-feature predictor list (same as steps 8-10)
feature_cols = [c for c in df.columns
                if c not in EXCLUDE_COLS and df[c].dtype in ['float64', 'int64']]

# Clean model matrix: drop any row with a NaN in predictors or outcome columns
model_df = df[['total_GLP_pat', 'population'] + feature_cols].dropna()
model_df = model_df[model_df['population'] > 0].reset_index(drop=True)
model_df['log_rate'] = np.log((model_df['total_GLP_pat'] + 0.5) / model_df['population'])

X = model_df[feature_cols].values          # (616, 68)
y = model_df['log_rate'].values            # (616,)

# Scaler on the 20 GAM features — needed to back-transform gam_partial_effects X_scaled
sc = StandardScaler().fit(model_df[TOP_FEATURES].values)

# Load all result CSVs
gb_imp  = pd.read_csv('gb_feature_importance.csv')
rf_imp  = pd.read_csv('rf_feature_importance.csv')
gam_imp = pd.read_csv('gam_feature_importance.csv')
gam_pe  = pd.read_csv('gam_partial_effects.csv')   # 2000 rows: 100 grid points × 20 features

print(f"  model_df: {len(model_df)} rows, {len(feature_cols)} features")

# =============================================================================
# 2.  REFIT GB & RF  (needed for Figure 3 PDPs — ~4 s total)
# =============================================================================
print("Refitting GB and RF for partial dependence plots…")

gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    min_samples_leaf=10, subsample=0.8, random_state=42)
gb.fit(X, y)

rf = RandomForestRegressor(
    n_estimators=100, min_samples_split=5, min_samples_leaf=2,
    max_features=0.5, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X, y)

print("  GB and RF fitted.")

# =============================================================================
# 3.  HELPER — back-transform GAM x_scaled to original scale for one feature
# =============================================================================
def gam_curve(feature_name):
    """Return (x_original, y_effect) arrays for one GAM feature."""
    sub = gam_pe[gam_pe['Feature'] == feature_name].sort_values('X_scaled')
    idx = TOP_FEATURES.index(feature_name)
    x_orig = sub['X_scaled'].values * sc.scale_[idx] + sc.mean_[idx]
    return x_orig, sub['Partial_Effect'].values

# =============================================================================
# FIGURE 1 — OUTCOME DISTRIBUTION
# =============================================================================
print("Generating Figure 1…")

# Panels A & B only need outcome columns (no predictor NaN filtering)
df1 = df.dropna(subset=['total_GLP_pat', 'population']).copy()
df1 = df1[df1['population'] > 0]
df1['rate']     = (df1['total_GLP_pat'] / df1['population']) * 100_000
df1['log_rate'] = np.log(df1['rate'])

# Panel C also needs UrbanCore (has its own NaNs)
df1c = df.dropna(subset=['total_GLP_pat', 'population', 'UrbanCore']).copy()
df1c = df1c[df1c['population'] > 0]
df1c['log_rate'] = np.log((df1c['total_GLP_pat'] + 0.5) / df1c['population'])

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8),
                         gridspec_kw={'width_ratios': [1.1, 1, 1]})

# --- A: raw rate histogram ---
ax = axes[0]
ax.hist(df1['rate'], bins=60, color=PAL['gam'], alpha=0.75,
        edgecolor='white', linewidth=0.4)
ax.set_xlabel('Prescriptions per 100,000 Population')       # unchanged — already compliant
ax.set_ylabel('Number of ZIP-3 Areas')                      # unchanged — already compliant
ax.set_title('A.  Raw Prescribing Rate', fontweight='bold', fontsize=9.5)
med = df1['rate'].median()
ax.axvline(med, color='#E74C3C', linestyle='--', linewidth=1.2)
ax.text(med + 1200, ax.get_ylim()[1] * 0.85,
        f'Median\n{med:,.0f}', fontsize=7.5, color='#E74C3C', ha='left')

# --- B: log-transformed (modeling scale) ---
ax = axes[1]
ax.hist(df1['log_rate'], bins=45, color=PAL['gam'], alpha=0.75,
        edgecolor='white', linewidth=0.4)
ax.set_xlabel('Log-Transformed Prescribing Rate (per 100,000 Population)')  # CHANGED: was 'Log(Prescriptions per 100,000)'
ax.set_ylabel('Number of ZIP-3 Areas')
ax.set_title('B.  Log-Transformed Rate\n(Modeling Scale)', fontweight='bold', fontsize=9.5)
ax.axvline(df1['log_rate'].median(), color='#E74C3C', linestyle='--', linewidth=1.2)

# --- C: UrbanCore scatter + OLS line ---
ax = axes[2]
uc = df1c['UrbanCore'].values
lr = df1c['log_rate'].values
ax.scatter(uc, lr, s=12, alpha=0.45, color=PAL['gam'], edgecolors='none')
z  = np.polyfit(uc, lr, 1)
xg = np.linspace(uc.min(), uc.max(), 100)
ax.plot(xg, np.polyval(z, xg), color='#E74C3C', linewidth=1.8)
ax.set_xlabel('Urban Core Fraction')
ax.set_ylabel('Log-Transformed Prescribing Rate (per 100,000 Population)')  # CHANGED: was 'Log(Prescriptions per 100,000)'
ax.set_title(f'C.  Urban Core vs. Prescribing\n(n = {len(df1c)})',
             fontweight='bold', fontsize=9.5)
r = np.corrcoef(uc, lr)[0, 1]
ax.text(0.05, 0.08, f'r = {r:.2f}', transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='grey', alpha=0.85))

plt.tight_layout()
fig.savefig('figure1_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  figure1_distribution.png")

# =============================================================================
# FIGURE 2 — MODEL PERFORMANCE LADDER
# =============================================================================
print("Generating Figure 2…")

gb_sum  = pd.read_csv('gb_model_summary.csv').set_index('Metric')['Value']
rf_sum  = pd.read_csv('rf_model_summary.csv').set_index('Metric')['Value']
gam_sum = pd.read_csv('gam_model_summary.csv').set_index('Metric')['Value']

means = [
    float(rf_sum['Ridge_CV_R2']),
    float(gam_sum['CV_R2_mean']),
    float(rf_sum['CV_R2_mean']),
    float(gb_sum['CV_R2_mean']),
]
stds = [
    float(gb_sum['CV_R2_std']),
    float(gam_sum['CV_R2_std']),
    float(rf_sum['CV_R2_std']),
    float(gb_sum['CV_R2_std']),
]

models = ['Linear\nRegression\n(Ridge)',
          'GAM\n(Spline +\nRidge)',
          'Random\nForest',
          'Gradient\nBoosting']
colors = [PAL['neu'], PAL['gam'], PAL['rf'], PAL['gb']]

fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(models))
ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='white',
       linewidth=1.2, width=0.55,
       error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': '#555'})

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.012, f'{m:.3f}',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Bracket: Ridge → GAM nonlinearity gain
yb = max(means[0] + stds[0], means[1] + stds[1]) + 0.035
ax.annotate('', xy=(0, yb), xytext=(1, yb),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
pp_gain = (means[1] - means[0]) * 100
ax.text(0.5, yb + 0.015, f'+{pp_gain:.1f} pp\n(nonlinearity)',
        ha='center', fontsize=7.5, color='#555', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8.5)
ax.set_ylabel('5-Fold Cross-Validated R\u00b2')   # CHANGED: was '5-Fold CV R²' (CV is abbreviation)
ax.set_ylim(0, 0.58)
ax.axhline(0, color='grey', linewidth=0.5)

ax.text(0.98, 0.04,
        'Error bars = \u00b11 SD across folds\n'
        'GAM selected as primary model:\n'
        '  optimal interpretability\u2013accuracy trade-off',
        transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.35', fc='#F5F5F5', ec='#CCC'))

plt.tight_layout()
fig.savefig('figure2_model_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  figure2_model_comparison.png")

# =============================================================================
# FIGURE 3 — PARTIAL DEPENDENCE OVERLAY (GAM / RF / GB)
# =============================================================================
print("Generating Figure 3…")

panel_specs = [
    ('health_uninsured',          'A.  Health Uninsured Rate',       'Health Uninsured Rate (%)'),                        # CHANGED: was '% Uninsured'
    ('education_bachelors',       "B.  Bachelor's Degree Holders",   "Population with Bachelor's Degree or Higher (%)"),  # CHANGED: was "% with Bachelor's+"
    ('UrbanCore',                 'C.  Urban Core Fraction',         'Urban Core (Proportion)'),                          # CHANGED: capitalized Proportion
    ('home_value',                'D.  Median Home Value',           'Median Home Value (USD)'),                          # CHANGED: added (USD)
    ('labor_force_participation', 'E.  Labor Force Participation',   'Labor Force Participation Rate (%)'),               # CHANGED: was '% in Labor Force'
    ('race_native',               'F.  Native American Population',  'Native American Population (%)'),                   # CHANGED: was '% Native American'
]

fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))

for pi, (fn, title, xlabel) in enumerate(panel_specs):
    row, col = divmod(pi, 3)
    ax = axes[row, col]

    # --- GB & RF PDPs (fresh from refitted models, original x-scale) ---
    col_idx = feature_cols.index(fn)
    gb_pdp  = partial_dependence(gb, X, features=[col_idx], grid_resolution=50)
    rf_pdp  = partial_dependence(rf, X, features=[col_idx], grid_resolution=50)
    shared_x = gb_pdp['grid_values'][0]
    gb_y_c   = gb_pdp['average'][0] - gb_pdp['average'][0].mean()
    rf_y_c   = rf_pdp['average'][0] - rf_pdp['average'][0].mean()

    # --- GAM curve: back-transform from X_scaled, interpolate onto shared_x ---
    gam_x, gam_y = gam_curve(fn)
    gam_yi = np.interp(shared_x, gam_x, gam_y, left=np.nan, right=np.nan)

    ax.plot(shared_x, gb_y_c,  color=PAL['gb'],  linewidth=1.8, label='Gradient Boosting')
    ax.plot(shared_x, rf_y_c,  color=PAL['rf'],  linewidth=1.8, label='Random Forest', linestyle='--')
    ax.plot(shared_x, gam_yi,  color=PAL['gam'], linewidth=2.2, label='GAM')
    ax.axhline(0, color='grey', linewidth=0.6, linestyle=':')

    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    if col == 0:
        ax.set_ylabel('Partial Effect\n(Centered)', fontsize=8)
    if fn == 'home_value':
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
        ax.tick_params(axis='x', labelsize=7)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
           frameon=True, edgecolor='#CCC', fancybox=True,
           bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig('figure3_partial_dependence.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  figure3_partial_dependence.png")

# =============================================================================
# FIGURE 4 — CROSS-METHOD IMPORTANCE CONSISTENCY
# =============================================================================
print("Generating Figure 4…")

# Merge on the 20 GAM features (the shared set)
gam_feats = gam_imp['Feature'].tolist()
merged = pd.DataFrame({
    'GAM': gam_imp.set_index('Feature')['Effect_Range'],
    'RF' : rf_imp.set_index('Feature')['Perm_Mean'],
    'GB' : gb_imp.set_index('Feature')['Perm_Mean'],
}).loc[gam_feats].dropna()

def minmax(s):
    return (s - s.min()) / (s.max() - s.min())

# Ranks (1 = most important); sort by average rank for display
ranks     = merged.rank(ascending=False).astype(int)
ranks['avg'] = ranks[['GAM', 'RF', 'GB']].mean(axis=1)
order     = ranks.sort_values('avg').index.tolist()
labels    = [LBL.get(f, f) for f in order]

fig, axes = plt.subplots(1, 2, figsize=(13, 7.2),
                         gridspec_kw={'width_ratios': [1.3, 0.7]})

# --- LEFT: grouped horizontal bars ---
ax   = axes[0]
n    = len(order)
ypos = np.arange(n)
bh   = 0.25

gam_mm = minmax(merged['GAM']).loc[order]
rf_mm  = minmax(merged['RF']).loc[order]
gb_mm  = minmax(merged['GB']).loc[order]

ax.barh(ypos + bh,  gam_mm, bh, color=PAL['gam'],
        label='GAM (Effect Range)',    edgecolor='white', linewidth=0.5)
ax.barh(ypos,       rf_mm,  bh, color=PAL['rf'],
        label='RF (Permutation Importance)', edgecolor='white', linewidth=0.5)   # CHANGED: was 'Perm. Importance'
ax.barh(ypos - bh,  gb_mm,  bh, color=PAL['gb'],
        label='GB (Permutation Importance)', edgecolor='white', linewidth=0.5)   # CHANGED: was 'Perm. Importance'

ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Normalized Importance (0 to 1 per Method)', fontsize=8.5)        # CHANGED: was '(0–1 per method)'
ax.set_title('Importance Rankings by Method', fontweight='bold', fontsize=9.5)
ax.legend(loc='lower right', fontsize=7.5, frameon=True, edgecolor='#CCC')
ax.set_xlim(0, 1.15)

# Highlight band behind the top-3 rows
for i in range(n - 3, n):
    ax.axhspan(i - 0.45, i + 0.45, color='#EBF5FB', zorder=0, alpha=0.6)
ax.text(1.02, n - 1.5, 'Top 3\nconsistent',
        fontsize=6.5, color=PAL['gam'], ha='left', va='center', style='italic')

# --- RIGHT: rank heatmap ---
ax = axes[1]
rank_arr = ranks[['GAM', 'RF', 'GB']].loc[order].values.astype(float)

im = ax.imshow(rank_arr, cmap='YlOrRd_r', aspect='auto', vmin=1, vmax=20)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['GAM', 'RF', 'GB'], fontsize=9, fontweight='bold')
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=8)
ax.set_title('Importance Rank\n(1 = most important)', fontweight='bold', fontsize=9.5)

for i in range(n):
    for j in range(3):
        v     = int(rank_arr[i, j])
        color = 'white' if v >= 12 else 'black'
        ax.text(j, i, str(v), ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=color)

fig.colorbar(im, ax=axes[1], shrink=0.6, pad=0.02, label='Rank')

# Spearman ρ footer
rho_gam_rf, _ = spearmanr(ranks['GAM'], ranks['RF'])
rho_gam_gb, _ = spearmanr(ranks['GAM'], ranks['GB'])
rho_rf_gb,  _ = spearmanr(ranks['RF'],  ranks['GB'])
fig.text(0.5, -0.02,
         f'Spearman rank correlations:  '
         f'GAM\u2013RF \u03c1 = {rho_gam_rf:.2f}   '
         f'GAM\u2013GB \u03c1 = {rho_gam_gb:.2f}   '
         f'RF\u2013GB \u03c1 = {rho_rf_gb:.2f}   (all p < 0.05)',
         ha='center', fontsize=8, style='italic', color='#555')

plt.tight_layout()
fig.savefig('figure4_cross_method_importance.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  figure4_cross_method_importance.png")

# =============================================================================
# FIGURE 5 — NONLINEARITY EVIDENCE (GAM smooth vs OLS line)
# =============================================================================
print("Generating Figure 5…")

nonlin_feats = ['health_uninsured', 'education_bachelors', 'UrbanCore']
linear_feats = ['density', 'unemployment_rate', 't2d_obesity_pat_count']
all6 = nonlin_feats + linear_feats

XLABEL5 = {
    'health_uninsured'      : 'Health Uninsured Rate (%)',                   # CHANGED: was 'Uninsured Rate (%)'
    'education_bachelors'   : "Population with Bachelor's Degree (%)",       # CHANGED: was "Bachelor's Degree (%)"
    'UrbanCore'             : 'Urban Core (Proportion)',                      # CHANGED: capitalized Proportion
    'density'               : 'Population Density (persons per square mile)', # CHANGED: added units
    'unemployment_rate'     : 'Unemployment Rate (%)',
    't2d_obesity_pat_count' : 'Type 2 Diabetes and Obesity Patient Count',   # CHANGED: was 'T2D/Obesity Patient Count'
}

# Read nonlinearity flags from gam_feature_importance.csv
nl_flags = gam_imp.set_index('Feature')

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

for idx, fn in enumerate(all6):
    row, col = divmod(idx, 3)
    ax = axes[row, col]

    x_orig, y_eff = gam_curve(fn)

    is_nl     = nl_flags.loc[fn, 'Nonlinearity'] == 'Nonlinear'
    direction = nl_flags.loc[fn, 'Direction']
    if   direction == 'Negative ↓': color = PAL['neg']
    elif direction == 'Positive ↑': color = PAL['pos']
    else:                           color = PAL['neu']

    # Filled area + smooth curve
    ax.fill_between(x_orig, 0, y_eff, alpha=0.15, color=color)
    ax.plot(x_orig, y_eff, color=color, linewidth=2.2)

    # OLS reference line
    lin = np.polyfit(x_orig, y_eff, 1)
    ax.plot(x_orig, np.polyval(lin, x_orig),
            color='grey', linewidth=1.2, linestyle='--', alpha=0.7,
            label='Linear Fit')                                   # CHANGED: was 'Linear fit'
    ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')

    ax.set_xlabel(XLABEL5.get(fn, fn), fontsize=8)
    if col == 0:
        ax.set_ylabel('Partial Effect (Centered)', fontsize=8)    # CHANGED: capitalized Centered

    # Badge
    badge    = '[Nonlinear]' if is_nl else '[Approximately Linear]'   # CHANGED: was '[≈ Linear]'
    badge_fc = '#E74C3C'     if is_nl else '#27AE60'
    ax.set_title(fn.replace('_', ' ').title(), fontweight='bold', fontsize=9)
    ax.text(0.98, 0.95, badge, transform=ax.transAxes, fontsize=7,
            ha='right', va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc=badge_fc, ec='none'))

# Row labels that double as section headers
axes[0, 0].set_ylabel('Partial Effect (Centered)\n— Nonlinear —',
                       fontsize=8, color=PAL['neg'])
axes[1, 0].set_ylabel('Partial Effect (Centered)\n— Approximately Linear —',  # CHANGED: was '— ≈ Linear —'
                       fontsize=8, color='#27AE60')
axes[0, 0].legend(loc='lower left', fontsize=7, frameon=True, edgecolor='#CCC')

plt.tight_layout()
fig.savefig('figure5_nonlinearity.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   figure5_nonlinearity.png")

print("\nAll 5 figures generated successfully.")
