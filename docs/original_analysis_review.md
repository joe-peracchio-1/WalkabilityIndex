# Review of the Original Walkability Analysis

Notebook under review: [`archived/Walkability ML.ipynb`](../archived/Walkability%20ML.ipynb)
(kept unchanged as a record of the original work).

Scope: this review was done by reading the notebook's code and saved outputs. The
source CSV (`epadatabase.csv`) was not in the repository, so no numbers were re-run.
Items that depend on the EPA documentation cite the
[National Walkability Index Methodology and User Guide (June 2021)](https://www.epa.gov/sites/default/files/2021-06/documents/national_walkability_index_methodology_and_user_guide_june2021.pdf).

## Summary

The notebook trains regressors to predict the EPA National Walkability Index (`NatWalkInd`)
from the other columns of the EPA Smart Location Database. The index is a deterministic
formula over four of those columns, so the high scores measure how well a model recovers
that formula. They say little about what makes a community walkable.

## 1. Target leakage (the central failure)

EPA defines the index as follows. Each of four Smart Location Database variables is placed
into 20 equal-sized quantiles across all U.S. block groups and given a rank score of 1 to 20:

| Symbol | SLD variable | Meaning | Weight |
|---|---|---|---|
| w | `D3B` | Intersection density | 1/3 |
| x | `D4A` | Distance from population center to nearest transit stop (m) | 1/3 |
| y | `D2B_E8MIXA` | Employment mix | 1/6 |
| z | `D2A_EPHHM` | Employment and household mix | 1/6 |

`NatWalkInd = w/3 + x/3 + y/6 + z/6`, on a 1 to 20 scale.

In cell 1 the notebook drops the four `*_Ranked` columns but keeps the raw `D3B`, `D4A`,
`D2B_E8MIXA` and `D2A_EPHHM` columns they are computed from. Because a rank is a monotone
step function of its raw value, the target is still fully determined by the features.

Evidence in the saved outputs:

- Tree models on raw features reach R2 about 0.998 (Random Forest, cell 13) and 0.997
  (Extra Trees, cell 16). Trees are well suited to learning quantile step functions.
- The same models after PCA drop to about 0.93 (cells 12 and 15). PCA mixes the four
  inputs with about 40 others, which blurs the thresholds.
- The feature-importance plots (cells 14 and 17) therefore mostly show the formula's inputs.
  They are not findings about walkability.

**Improvement avenue:** treat `NatWalkInd` and its four inputs as the definition of
walkability, not as something to predict. Use the remaining variables (density, demographics,
land use, jobs, auto ownership, regional context) to explain why block groups differ. Either
exclude all inputs and ranks from the features, or model something else (see section 6).

## 2. Data cleaning errors

| # | Cell | Problem | Why it matters |
|---|---|---|---|
| 2.1 | 1 | `-99999` replaced with `0` | `-99999` is a no-data code. For `D4A` (distance to transit), 0 means the best possible value, so places without transit data are treated as ideal. Other columns get fabricated zeros. |
| 2.2 | 3 | Mean imputation of remaining nulls, fit on the full dataset | Uses the national mean for local gaps and uses information from rows that later land in test folds. Neighboring block groups would be a better source. |
| 2.3 | 3 | `get_dummies` on `Region` keeps every level | With an intercept this is perfectly collinear. The non-PCA linear model returns coefficients near 1e11 (cell 9). |
| 2.4 | 1 to 3 | Redundant columns kept, such as `Ac_Total = Ac_Water + Ac_Land` | More collinearity, and it inflates the apparent number of "independent" features. |
| 2.5 | 3 | `CBSA_POP`, `CBSA_EMP` and `CBSA_WRK` kept | These are identical for every block group in a metro, so they act as a metro ID that tree models can memorize. |
| 2.6 | 1 to 2 | Duplicate and constant-column checks run after the zero replacement | What counts as constant or duplicate depends on the earlier error. |
| 2.7 | none | No geographic identifier (block group GEOID) used | Results cannot be mapped or tied to a community. |

## 3. Evaluation errors

| # | Cell | Problem |
|---|---|---|
| 3.1 | 4 to 6 | `StandardScaler` and `PCA` are fit on all data before cross-validation, so test folds influence preprocessing. |
| 3.2 | 8 to 19 | `KFold(shuffle=True)` on spatial data. Block groups in the same metro are near-copies, so neighbors sit in both train and test. This overstates generalization. Grouped CV by CBSA or state is the right check. |
| 3.3 | 8 to 23 | MASE is computed with `np.mean(np.abs(np.diff(y_train)))` as the denominator. That is the naive one-step error for an ordered time series. Rows here have no order, so the value is arbitrary. The plot titles also expand it as "Mean Absolute *Squared* Error". |
| 3.4 | 20, 22 | Averages are divided by 3 afterwards ("Forgetting to add average to list") and the same lists are reused across cells, which is fragile. |
| 3.5 | 10 to 17 | Hyperparameters are hardcoded (`alpha=0.01`, 75 trees). The grid searches are commented out and were written for a different split. |
| 3.6 | 18 to 19 | The voting ensemble includes plain linear regression and does worse than Extra Trees alone. The result is reported without discussion. |
| 3.7 | 3 | `iteritems` was removed in pandas 2.0, and `clone = epa_df` aliases the DataFrame instead of copying it. The notebook will not run on a current environment. |
| 3.8 | 24 | The final cell is empty and there is no written conclusion. |

**Improvement avenue:** put preprocessing inside a `Pipeline`, use grouped spatial
cross-validation, report R2, RMSE and MAE, and use permutation importance or SHAP on
held-out folds.

## 4. Documentation notes

- Cell 1 describes the dropped `*_Ranked` columns as "averages of groupings". They are
  quantile rank scores of the raw inputs.
- The EPA overview page's field table describes `D4a` as predicted carpool mode split.
  The methodology guide and the Smart Location Database define it as distance to the nearest
  transit stop. The methodology guide is treated as authoritative here and should be confirmed
  against the data dictionary.

## 5. What the original work does support

- The EPA index is mostly driven by intersection density and transit proximity, which carry
  two thirds of the weight. This follows from the formula and needs no model.
- The descriptive Tableau map of scores by community remains valid for its purpose, as long
  as it was built from the unmodified EPA scores and not from model predictions. This has not
  been checked.

## 6. Proposed direction for the revamp

1. **Re-acquire the data** from EPA with block group GEOIDs retained, and treat `-99999` as
   missing.
2. **Describe, then explain.** Start with a correct descriptive layer: scores by state, metro
   and county, population-weighted, with the four EPA categories.
3. **Ask answerable questions** with the index and its inputs held out of the features:
   what built-environment, demographic and economic factors distinguish high and low
   walkability areas, and which communities are walkable relative to similar peers?
4. **Feed an improvement layer.** For a given community, identify which of the four inputs
   holds its score back (for example, low intersection density versus poor transit proximity)
   and use that to recommend resources and interventions.
5. **Reuse the code** as tested modules so the UI and GenAI components share one
   implementation.
