# Characterizing Geographic Variation in GLP-1 Receptor Agonist Prescribing Using Interpretable Machine Learning

Machine learning analysis of neighborhood-level predictors of GLP-1RA prescription patterns across 667 three-digit ZIP code areas in the United States.

## Overview

This repository contains a complete analysis pipeline examining geographic variation in glucagon-like peptide-1 receptor agonist (GLP-1RA) prescribing using IQVIA claims data (2010–2022) merged with neighborhood sociodemographic characteristics. The analysis compares generalized additive models (GAMs) with tree-based ensemble methods (random forest, gradient boosting) to identify and rank the most important predictors of prescription volume.

**Key Finding**: Health insurance coverage emerged as the dominant predictor across all methods, with areas in the highest uninsurance quartile having rates approximately half those in the lowest quartile. The GAM explained 33.5% of variance (CV R² = 0.335), balancing interpretability with predictive performance.

## Repository Structure

```
data_preprocessing: please see at https://github.com/Sherry-Yun-Wang/Geographic_and_Sociodemographic_Variation_in_GLP1RA.git
modeling
├── 0_zip5_to_zip3.py                          # ZIP-5 → ZIP-3 aggregation
├── 1_zip3_population_weighted_averages.py     # Population-weighted demographic aggregation
├── 2_merge_ruca_urbancore.py                  # Urban-rural classification (RUCA codes)
├── 3_merge_zip3_iqvia.py                      # Merge IQVIA claims with ZIP-3 characteristics
├── 4_vif.py                                   # Multicollinearity diagnostics (VIF screening)
├── 5_negative_binomial_glm.py                 # Negative binomial GLM (VIF-screened predictors)
├── 6_elastic_net_negative_binomial.py         # Elastic net variable selection + NB GLM
├── 7_pca_glm.py                               # Factor analysis + NB GLM
├── 8_gradient_boosting_trees.py               # Gradient boosting regressor
├── 9_random_forests.py                        # Random forest regressor
├── 10_generalized_additive_models.py          # GAM via cubic B-splines + Ridge
└── 11_generate_figures.py                     # Generate all 5 publication figures
```

## Pipeline Steps

### Step 0: ZIP-5 to ZIP-3 Aggregation
**Script**: `0_zip5_to_zip3.py`  
**Input**: `uszips.csv` (ZIP-5 level data with lat/lng, population, demographics)  
**Output**: `uszips_zip3.xlsx`  
**Purpose**: Aggregate 33,000+ five-digit ZIP codes into 900+ three-digit ZIP code areas for area-level analysis.

**Method**: Group by first 3 digits of ZIP code, compute population-weighted centroids (lat/lng).

---

### Step 1: Population-Weighted Demographic Averages
**Script**: `1_zip3_population_weighted_averages.py`  
**Input**: `uszips.csv`, `uszips_zip3.xlsx`  
**Output**: `uszips_zip3_popweighted.xlsx`  
**Purpose**: Compute population-weighted means for all demographic variables at the ZIP-3 level.

**Method**: For each ZIP-3 area, weight each ZIP-5's demographic value by its population share, then sum. Handles 68 sociodemographic variables spanning age structure, race/ethnicity, income, education, employment, housing, disability, and health insurance coverage.

---

### Step 2: Merge RUCA Urban-Core Classification
**Script**: `2_merge_ruca_urbancore.py`  
**Input**: `uszips_zip3_popweighted.xlsx`, `RUCA-codes-2020-tract.xlsx`  
**Output**: `uszips_zip3_with_urbancore.xlsx`  
**Purpose**: Add urban-rural classification via RUCA (Rural-Urban Commuting Area) codes.

**Method**: 
- Merge census tract-level RUCA codes to ZIP-5s via lat/lng spatial join
- Classify tracts as "Urban Core" (RUCA primary code = 1) or non-core
- Aggregate to ZIP-3: `UrbanCore` = fraction of population in urban core tracts

**RUCA Definition**: Primary code 1 = metropolitan area core (≥50K population, high commuting integration). This is a continuous 0–1 variable, not a binary urban/rural flag.

---

### Step 3: Merge IQVIA Claims Data
**Script**: `3_merge_zip3_iqvia.py`  
**Input**: `uszips_zip3_with_urbancore.xlsx`, `final_dataset_from_IQVIA.csv`  
**Output**: `IQVIA_with_zip3_characteristics.csv`  
**Purpose**: Merge GLP-1RA prescription counts with neighborhood characteristics.

**IQVIA Data**:
- `total_GLP_pat`: Total unique patients with ≥1 GLP-1RA prescription (2010–2022)
- `t2d_obesity_pat_count`: Patients with type 2 diabetes or obesity diagnosis
- Coverage: Commercially insured population only (excludes Medicaid-only, uninsured, Medicare-only)

**Final Dataset**: 667 ZIP-3 areas × 77 variables (outcome + 68 predictors + geographic identifiers)

---

### Step 4: Multicollinearity Diagnostics
**Script**: `4_vif.py`  
**Input**: `IQVIA_with_zip3_characteristics.csv`  
**Output**: Console report of VIF values  
**Purpose**: Identify collinear predictors using Variance Inflation Factor.

**Method**: Iteratively compute VIF for all predictors; flag any VIF > 10 (severe multicollinearity). No predictors are automatically dropped — this is a diagnostic step only.

**Key Finding**: Several income brackets, age groups, and education variables show high VIF (expected due to compositional constraints where categories sum to 100%). These are handled via variable selection in subsequent steps.

---

### Step 5: Negative Binomial GLM (VIF-Screened)
**Script**: `5_negative_binomial_glm.py`  
**Input**: `IQVIA_with_zip3_characteristics.csv`  
**Output**: 
- `glm_nb_results.csv` (coefficients, IRR, p-values)
- `step6_state.pkl` (fitted model object)

**Purpose**: Baseline interpretable model using manually curated predictor set (32 variables) chosen to minimize VIF.

**Method**: 
- Outcome: GLP-1RA count (offset: log(population))
- Model: Negative binomial GLM with log link
- Standard errors: Hessian-based
- Predictor selection: Dropped redundant income/age/education categories based on VIF > 10

**Model Fit**: 
- N = 658 ZIP-3 areas
- AIC = 13,723.7, BIC = 13,876.3
- Dispersion α = 0.868

**Significant Predictors** (p < 0.05):
- **Negative**: `race_black`, `race_native`, `race_pacific`, `home_ownership`, `family_dual_income`, `disabled`, `health_uninsured`, `density`
- **Positive**: `male`, `age_30s`, `age_50s`, `age_60s`, `age_over_80`, `education_graduate`, `unemployment_rate`
- **Strong effect**: `UrbanCore` (IRR = 0.23, p < 0.001) — urban areas have 77% lower prescribing rates

---

### Step 6: Elastic Net Variable Selection + NB GLM
**Script**: `6_elastic_net_negative_binomial.py`  
**Input**: `IQVIA_with_zip3_characteristics.csv`  
**Output**: 
- `glm_nb_elasticnet_results.csv` (coefficients for selected variables)
- `elasticnet_variable_selection.csv` (selection summary)
- `step7_state.pkl` (fitted model object)

**Purpose**: Data-driven variable selection via elastic net regularization, then refit unregularized NB GLM on selected variables.

**Method**:
1. Fit elastic net Poisson GLM on all 68 predictors (α = 0.5, λ chosen via 5-fold CV)
2. Extract 31 non-zero coefficients
3. Refit standard NB GLM on these 31 variables with Hessian SEs

**Model Fit**:
- N = 658 ZIP-3 areas
- AIC = 13,477.1 (**better than VIF model**)
- BIC = 13,625.2
- Dispersion α = 0.834

**Variables Selected**: Mix of demographic (age, race), socioeconomic (income, education, employment), and health access (insurance) predictors. Elastic net dropped many collinear income brackets and compositionally redundant categories.

---

### Step 7: Factor Analysis + NB GLM
**Script**: `7_pca_glm.py`  
**Input**: `IQVIA_with_zip3_characteristics.csv`  
**Output**: 
- `glm_nb_factor_results.csv` (coefficients on factor scores)
- `factor_loadings_matrix.csv` (variable → factor loadings)
- `factor_interpretation_summary.csv` (named factors with top loadings)
- `step8_state.pkl` (fitted model object)

**Purpose**: Reduce multicollinearity via factor analysis; interpret latent sociodemographic dimensions.

**Method**:
1. Fit factor analysis (10 factors, varimax rotation) on 68 predictors
2. Compute factor scores for each ZIP-3 area
3. Fit NB GLM with outcome ~ factor scores + `t2d_obesity_pat_count` + `UrbanCore`
4. Use **bootstrap standard errors** (100 replicates) instead of Hessian for robust inference

**Model Fit**:
- N = 616 ZIP-3 areas (dropped 51 rows with missing factor-relevant variables)
- **AIC = 12,928.3 (best among Steps 5–7)**
- **BIC = 12,990.2 (best among Steps 5–7)**
- Dispersion α = 0.808

**Identified Factors** (labeled by dominant loadings):
1. **High_SES**: Income medians, top income brackets
2. **Elderly_Pop**: Age 60+, 70+, 80+
3. **Racial_Composition**: % Black, unmarried, unemployed
4. **Young_Pop**: Age 18–29
5. **Working_Age**: Age 30s
6. **Hispanic_Immigrant**: % Hispanic, limited English, < HS education
7. **High_SES_2**: Middle-upper income brackets
8. **Dimension_8**: (No clear interpretation — mixed loadings)
9. **Working_Age_2**: Age 40s–50s, commute time
10. **Middle_SES**: % Veteran, some college

**Significant Factor Effects** (8 of 10 factors, p < 0.05):
- **Positive** (higher prescribing): `Elderly_Pop` (IRR 1.26***), `Racial_Composition` (1.45***), `Hispanic_Immigrant` (1.15*), `High_SES_2` (1.35***)
- **Negative**: `Dimension_8` (IRR 0.75***), `Middle_SES` (0.75***)
- `UrbanCore` remains significant (IRR 0.75*, p = 0.015)
- `t2d_obesity_pat_count` **negative** (IRR 0.87*, p = 0.026) — counterintuitive, likely a methodological artifact of the offset structure

---

### Step 8: Gradient Boosting Regressor
**Script**: `8_gradient_boosting_trees.py`  
**Input**: `IQVIA_with_zip3_characteristics.csv`  
**Output**: 
- `gb_feature_importance.csv` (permutation importance + effect directions)
- `gb_model_summary.csv` (hyperparameters, CV R², train R²)
- `step9_state.pkl` (fitted model object)

**Purpose**: Predictive benchmark; feature importance via permutation; partial dependence analysis.

**Method**:
- Outcome: `log(rate)` = `log((GLP_count + 0.5) / population)` — **regression on log-transformed rate**, not count modeling
- All 68 features (no variable selection)
- Hyperparameter tuning: Grid search (5-fold CV) over learning rate, max depth, n_estimators, subsample
- Feature importance: Permutation importance (how much CV R² drops when each feature is shuffled)
- Effect direction: Sign of mean partial dependence slope

**Best Hyperparameters**:
- `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `min_samples_leaf=10`, `subsample=0.8`

**Performance**:
- **CV R² = 0.408** (5-fold, ±0.070 SD)
- Train R² = 0.978 (evidence of overfitting)
- Ridge baseline (68 features): CV R² = 0.248
- **Improvement: +16.0 pp over Ridge**

**Top 5 Features (Permutation Importance)**:
1. `health_uninsured` (0.149) — **dominant**, 2× the next feature
2. `education_bachelors` (0.068)
3. `UrbanCore` (0.067)
4. `race_native` (0.051)
5. `t2d_obesity_pat_count` (0.046)

**Effect Directions**: Almost all top features show **negative** effects (higher value → lower prescribing): `health_uninsured` (−0.61), `education_bachelors` (−0.45), `UrbanCore` (−0.40), `race_native` (−0.34). Positive: `home_value` (+0.27).

---

### Step 9: Random Forest Regressor
**Script**: `9_random_forests.py`  
**Input**: `IQVIA_with_zip3_characteristics.csv`  
**Output**: 
- `rf_feature_importance.csv` (permutation + MDI importance, with GB comparison)
- `rf_model_summary.csv` (hyperparameters, CV R², OOB R²)
- `step10_state.pkl` (fitted model object)

**Purpose**: Second predictive benchmark; cross-method importance consistency check.

**Method**:
- Outcome: `log(rate)` (same as GB)
- All 68 features
- Hyperparameter tuning: Randomized search (50 iterations, 5-fold CV)
- Feature importance: Both MDI (mean decrease impurity) and permutation importance computed
- Cross-method comparison: Spearman rank correlation between RF and GB importance

**Best Hyperparameters**:
- `n_estimators=100`, `max_depth=None` (no limit), `min_samples_leaf=2`, `max_features=0.5`, `min_samples_split=5`

**Performance**:
- **CV R² = 0.383** (5-fold, ±0.066 SD)
- **OOB R² = 0.402** (out-of-bag, natural validation)
- Train R² = 0.892 (less overfitting than GB)
- GB slightly edges RF on CV R² (0.408 vs 0.383)

**Top 5 Features (Permutation Importance)**:
1. `health_uninsured` (0.149) — **identical rank to GB**
2. `education_bachelors` (0.081)
3. `home_value` (0.075)
4. `labor_force_participation` (0.052)
5. `self_employed` (0.044)

**RF vs GB Consistency**:
- Spearman ρ = **0.773** (p < 0.0001) — strong agreement
- Top 3 features identical across both methods
- Notable rank swaps:
  - `UrbanCore`: rank **3** in GB, rank **15** in RF
  - `t2d_obesity_pat_count`: rank **5** in GB, rank **10** in RF
  - → GB more sensitive to these predictors

**Effect Directions**: Match GB on all shared top features. `health_uninsured`, `education_bachelors`, `labor_force_participation`, `race_native`, `UrbanCore` all negative. `home_value`, `education_highschool` positive.

---

### Step 10: Generalized Additive Model (GAM)
**Script**: `10_generalized_additive_models.py`  
**Input**: `IQVIA_with_zip3_characteristics.csv`  
**Output**: 
- `gam_feature_importance.csv` (effect range + nonlinearity flags)
- `gam_model_summary.csv` (CV R², linear baseline comparison)
- `gam_partial_effects.csv` (100 grid points × 20 features for plotting)
- `step11_state.pkl` (fitted model object)

**Purpose**: Primary interpretable model. Captures nonlinear effects while remaining communicable to clinical audiences.

**Method**:
- Outcome: `log(rate)` (regression, same as GB/RF)
- **20 features** selected based on importance from GB/RF + domain knowledge
- Cubic B-splines (5 knots per feature, degree 3) → 6 basis functions per feature → 120-column design matrix
- Ridge regularization (α = 10, selected via 5-fold CV)
- Linear baseline: Ridge regression on the same 20 features (no splines) for comparison

**Selected Features**: `health_uninsured`, `education_bachelors`, `home_value`, `UrbanCore`, `race_native`, `t2d_obesity_pat_count`, `labor_force_participation`, `self_employed`, `education_highschool`, `hispanic`, `farmer`, `charitable_givers`, `veteran`, `race_black`, `rent_burden`, `age_over_18`, `disabled`, `commute_time`, `density`, `unemployment_rate`

**Performance**:
- **CV R² = 0.335** (5-fold, ±0.061 SD)
- Train R² = 0.421
- **Linear Ridge (same 20 features): CV R² = 0.208**
- **GAM improvement: +12.7 pp over linear** ← confirms substantial nonlinearity

**Nonlinearity Detection**: 15 of 20 features are **nonlinear** (residual std > 0.05 after removing linear trend). Only 5 approximately linear: `t2d_obesity_pat_count`, `farmer`, `race_black`, `density`, `unemployment_rate`.

**Top 5 Features (Effect Range = max − min of smooth function)**:
1. `health_uninsured` (1.04, **Negative ↓**, Nonlinear)
2. `education_bachelors` (0.92, **Negative ↓**, Nonlinear)
3. `UrbanCore` (0.70, **Negative ↓**, Nonlinear)
4. `self_employed` (0.66, **Negative ↓**, Nonlinear)
5. `home_value` (0.61, **Positive ↑**, Nonlinear)

**Direction Discrepancy to Investigate**: `race_native` shows **Positive ↑** in GAM (effect range 0.44) but was **Negative ↓** in RF/GB partial dependence. The spline is likely capturing a different segment of the curve — examine the partial effect plot to diagnose.

---

### Step 11: Generate Publication Figures
**Script**: `11_generate_figures.py`  
**Input**: 
- `IQVIA_with_zip3_characteristics.csv`
- `gb_feature_importance.csv`, `rf_feature_importance.csv`, `gam_feature_importance.csv`
- `gam_partial_effects.csv`
- `gb_model_summary.csv`, `rf_model_summary.csv`, `gam_model_summary.csv`

**Output**: 5 publication-ready figures (PNG, 200 DPI)

**Requirements**: 
- No `.pkl` files needed — script refits GB and RF from CSV
- GAM curves read directly from `gam_partial_effects.csv`

**Figures**:

#### Figure 1 — Outcome Distribution
Three panels showing raw prescribing rate distribution → log transformation → urban-core bivariate relationship.
- **Panel A**: Raw rate histogram (heavy right skew, >10× range)
- **Panel B**: Log-transformed rate (approximately normal, modeling scale)
- **Panel C**: UrbanCore vs log(rate) scatter + OLS line (r = −0.22, negative relationship)

**Purpose**: Motivates log transformation and previews the key urban-rural finding.

#### Figure 2 — Model Performance Ladder
Bar chart comparing 5-fold CV R² across 4 methods: Ridge (0.248) → GAM (0.335) → RF (0.383) → GB (0.408).
- Annotated bracket: Ridge → GAM = **+8.8 pp** gain (computed from CSV values)
- Error bars = ±1 SD across folds

**Purpose**: Shows GAM closes most of the gap to tree methods while remaining interpretable. The nonlinearity gain is quantified and made explicit.

#### Figure 3 — Partial Dependence Overlay (6 panels, 3 curves each)
Six key features: `health_uninsured`, `education_bachelors`, `UrbanCore`, `home_value`, `labor_force_participation`, `race_native`.

Each panel overlays:
- **GAM** smooth function (solid blue)
- **RF** partial dependence (dashed orange)
- **GB** partial dependence (solid green)

All three curves on the same x-axis (original scale) for direct comparison.

**Purpose**: Core evidence figure. Shows (a) agreement on effect direction across methods, (b) smooth GAM curves vs stepwise tree curves, (c) complex relationships like the U-shaped `home_value` effect that linear models miss.

#### Figure 4 — Cross-Method Importance Consistency
Two-panel figure:
- **Left**: Grouped horizontal bars showing normalized importance (0–1 per method) for all 20 GAM features, sorted by average rank. Top 3 highlighted with blue band.
- **Right**: Rank heatmap (1 = most important) with numbers annotated in each cell.

**Footer**: Spearman rank correlations: GAM–RF ρ = 0.55, GAM–GB ρ = 0.57, RF–GB ρ = 0.51 (all P < 0.05).

**Purpose**: Demonstrates robustness — the top predictors are consistent across methods despite different mathematical approaches.

#### Figure 5 — Nonlinearity Evidence (6 panels)
**Top row**: 3 features with strongest **nonlinear** effects (`health_uninsured`, `education_bachelors`, `UrbanCore`)  
**Bottom row**: 3 features with approximately **linear** effects (`density`, `unemployment_rate`, `t2d_obesity_pat_count`)

Each panel:
- GAM smooth function (solid colored curve + shaded area)
- OLS reference line (dashed grey)
- Badge: [Nonlinear] (red) or [≈ Linear] (green)

**Purpose**: Justifies GAM over linear regression. Shows the smooth functions deviate substantially from OLS lines for nonlinear features, but **track** the OLS line for linear features — the GAM isn't over-smoothing, it finds linearity where it exists.

---

## Cross-Model Summary

| Model | CV R² | N obs | N predictors | Approach |
|-------|-------|-------|--------------|----------|
| Ridge (all features) | 0.248 | 658 | 68 | Linear regression baseline |
| NB GLM (VIF) | N/A | 658 | 32 | Count model, manual selection |
| NB GLM (Elastic Net) | N/A | 658 | 31 | Count model, elastic net selection |
| NB GLM (Factor Analysis) | N/A | 616 | 12 | Count model, factor scores (**best AIC/BIC**) |
| Ridge (top 20) | 0.208 | 616 | 20 | Linear regression on selected features |
| GAM | **0.335** | 616 | 20 | Splines + Ridge (**primary model**) |
| Random Forest | 0.383 | 616 | 68 | Ensemble trees |
| Gradient Boosting | **0.408** | 616 | 68 | Ensemble trees (**best predictive**) |

**Best predictive model**: Gradient Boosting (CV R² = 0.408)  
**Best parsimonious model**: Factor Analysis NB GLM (AIC = 12,928, 12 predictors)  
**Primary interpretable model**: GAM (optimal balance of accuracy + interpretability)

---

## Consistent Top Predictors (Across All Methods)

1. **`health_uninsured`** — Dominant in all models
   - RF/GB permutation importance: ~0.15 (highest)
   - GAM effect range: 1.04 (highest)
   - Direction: **Negative** across all methods (higher uninsurance → lower prescribing)

2. **`education_bachelors`** — Consistently rank 2–3
   - GAM effect range: 0.92
   - Direction: **Negative** (more education → lower prescribing)

3. **`UrbanCore`** — Highly significant in all NB GLMs, top 3 in GB, rank 15 in RF
   - NB GLM IRR: 0.23–0.75 (urban areas have dramatically lower rates)
   - Direction: **Negative**

4. **`home_value`** — Positive effect across all methods
   - GAM/RF/GB: Nonlinear U-shaped relationship
   - Direction: **Positive**

5. **`race_native`**, **`labor_force_participation`**, **`self_employed`** — Round out top 10 in most models

---

## Key Substantive Findings

1. **Insurance coverage is the dominant predictor** of GLP-1RA prescribing variation, consistent across all methods. Areas with higher uninsurance rates have substantially lower prescription volumes, suggesting access barriers as a primary driver of geographic disparities.

2. **Urban areas have lower prescribing rates** than rural areas after adjusting for all other characteristics. This persists across all models and contradicts common assumptions about rural healthcare access. The drivers remain unclear but may reflect higher disease burden in rural areas or differences in prescriber practice patterns.

3. **Nonlinearity is substantial**. The GAM improves on linear regression by +12.7 pp (CV R²: 0.335 vs 0.208), with 15 of 20 features showing nonlinear relationships. This justifies the added complexity of smooth functions.

4. **Tree methods achieve only modest gains over GAM** (+4.8–7.3 pp), suggesting the interpretability cost of black-box models may not be justified for policy communication.

5. **Feature importance rankings are robust**. Spearman rank correlations between GAM, RF, and GB range from 0.51–0.57 (all p < 0.05), indicating findings are not method-dependent.

6. **Substantial unexplained variance** (~66% in best model) highlights limitations of area-level analysis. Individual-level factors (patient preferences, comorbidities, prescriber behavior) likely account for much of the heterogeneity.

---

## Requirements

### Python Packages
```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl
```

### Data Files (Not Included)
Due to data use agreements, the following files are **not** included in this repository:
- `uszips.csv` — ZIP-5 level demographics (SimpleMaps or similar commercial database)
- `RUCA-codes-2020-tract.xlsx` — USDA Rural-Urban Commuting Area codes
- `final_dataset_from_IQVIA.csv` — IQVIA PharMetrics® Plus claims data (requires data use agreement with IQVIA)

### Intermediate Files (Generated by Pipeline)
All intermediate CSV/XLSX files are generated by the pipeline. If starting from Step 3 (assuming you have the merged geographic file), you need:
- `uszips_zip3_with_urbancore.xlsx`
- `final_dataset_from_IQVIA.csv`

---

## Usage

### Full Pipeline (Steps 0–11)
Run scripts in order. Each step reads the output of the previous step:

```bash
python 0_zip5_to_zip3.py
python 1_zip3_population_weighted_averages.py
python 2_merge_ruca_urbancore.py
python 3_merge_zip3_iqvia.py
python 4_vif.py                              # Diagnostic only — no output files
python 5_negative_binomial_glm.py
python 6_elastic_net_negative_binomial.py
python 7_pca_glm.py
python 8_gradient_boosting_trees.py
python 9_random_forests.py
python 10_generalized_additive_models.py
python 11_generate_figures.py                # Generates all 5 figures
```

### Generate Figures Only (Step 11)
If you already have all the summary CSVs from Steps 8–10, you can generate figures without re-running models:

```bash
python 11_generate_figures.py
```

**Required files**:
- `IQVIA_with_zip3_characteristics.csv`
- `gb_feature_importance.csv`, `rf_feature_importance.csv`, `gam_feature_importance.csv`
- `gam_partial_effects.csv`
- `gb_model_summary.csv`, `rf_model_summary.csv`, `gam_model_summary.csv`

**Output**: `figure1_distribution.png`, `figure2_model_comparison.png`, `figure3_partial_dependence.png`, `figure4_cross_method_importance.png`, `figure5_nonlinearity.png`

---

## Methods Notes

### Why Three Different Modeling Approaches?

1. **Negative Binomial GLM (Steps 5–7)**: Traditional count modeling approach. Directly models prescription **counts** with population as an offset. Provides interpretable incidence rate ratios (IRR). Used for exploratory variable selection and factor analysis.

2. **Tree-Based Regression (Steps 8–9)**: Models log-transformed **rates** directly (`log((count + 0.5) / population)`). Chosen for predictive benchmarking and feature importance ranking. The +0.5 constant handles zero counts.

3. **GAM (Step 10)**: Also models log-transformed rates. Selected as the **primary model** because it balances predictive performance with interpretability. Smooth functions can be visualized and communicated to clinical audiences.

### Why Did We Switch from Count Modeling to Rate Regression?

Steps 5–7 use NB GLMs because they:
- Handle overdispersion in count data (variance > mean)
- Provide interpretable IRRs
- Are standard in epidemiologic literature

Steps 8–10 switch to regression on log(rate) because:
- Tree methods (RF/GB) and GAMs are **not naturally suited to count data** — they're designed for continuous outcomes
- Log-transforming the rate gives a continuous outcome while preserving the relative scale
- The +0.5 constant avoids log(0) for zero-count areas (only 1 area had zero counts)
- This is a pragmatic choice to enable comparison across methods

**Important**: The two approaches are **not directly comparable** in terms of model fit statistics. NB GLMs report AIC/BIC (lower is better); regression models report R² (higher is better). We report both to characterize each model family on its own terms.

### Cross-Validation Strategy

All models use **5-fold cross-validation**:
- Data split into 5 equal folds
- Each fold used once as test set (remaining 4 folds = training set)
- Performance metrics averaged across 5 test folds
- Standard deviation across folds reported as uncertainty

This gives unbiased estimates of out-of-sample prediction error.

---

## Reproducibility Notes

### Random Seeds
All models use `random_state=42` for reproducibility. Re-running the pipeline will produce identical results.

---

## Acknowledgments

This analysis uses IQVIA PharMetrics® Plus data under a data use agreement. RUCA codes are provided by the USDA Economic Research Service.
