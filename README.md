# Credit Risk Scoring & Expected Loss Analysis

A portfolio project demonstrating end-to-end credit risk analytics: data preparation, SQL-based portfolio analysis, PD modelling (Logistic Regression and Random Forest), probability calibration, and Expected Loss scenario analysis under a simplified Basel-style EL scenario.

---

## Dataset

| Field | Detail |
|---|---|
| **Source** | [UCI Machine Learning Repository — Default of Credit Card Clients (ID 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| **Citation** | Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480. |
| **Instances** | 30,000 credit card clients (Taiwan, April–September 2005) |
| **Features** | 23 predictors: demographics, repayment history, bill amounts, payment amounts |
| **Target** | `DEFAULT_NEXT_MONTH` — binary (1 = defaulted, 0 = did not default) |
| **Default rate** | 22.12% (class imbalance: ~3.5:1 non-default : default) |

---

## Methodology & Findings

### 1. SQL Risk Analysis
Beyond basic aggregations, the project utilises SQL to conduct repayment-status default-rate analysis, portfolio segmentation, CTE + NTILE quartile analysis, and exposure concentration analysis.

### 2. PD Modelling (Probability of Default)
The project compares a Logistic Regression baseline with a Random Forest model. Unweighted models were deliberately chosen to preserve the natural probability scale, ensuring the Mean Predicted PD (22.19%) closely matches the observed default rate (22.12%).

**Test Set Performance:**
| Metric | Logistic Regression | Random Forest (Unweighted) |
|
| **ROC-AUC** | 0.7084 | 0.7748 |
| **Gini** | 0.4168 | 0.5496 |
| **KS Statistic** | 0.2850 | 0.4237 |
| **Brier Score** | 0.1685 | 0.1356 |

### 3. Risk Segmentation (5-Grade Illustrative Master Scale)
Clients are segmented into a 5-grade scale based on fixed PD cutoffs. The model demonstrates monotonic rank-ordering capabilities.

| Grade | Clients | Avg_PD | Obs_Default | Expected_Def | Actual_Def | EL (NTD) |
|---|
| **1. Very Low (<5%)** | 431 | 3.5% | 4.9% | 15.08 | 21 | 1,997,425 |
| **2. Low (5-15%)** | 2,509 | 10.3% | 10.5% | 258.32 | 264 | 21,835,470 |
| **3. Medium (15-30%)** | 1,691 | 20.3% | 18.6% | 344.00 | 314 | 20,973,880 |
| **4. High (30-60%)** | 889 | 42.8% | 43.5% | 380.21 | 387 | 20,298,420 |
| **5. Very High (>=60%)** | 480 | 69.5% | 71.0% | 333.64 | 341 | 14,277,730 |

### 4. Expected Loss Scenario & Backtesting
A simplified Basel-style Expected Loss formula is applied:
`EL = 1-month PD × LGD × Exposure Proxy`

**Default-based Loss Proxy Backtest:**
Loss proxy = `LIMIT_BAL` × `actual_default` × `45%`. Because the same 45% LGD enters both sides, it cancels out; this backtest tests exposure-weighted PD calibration only, not loss severity. 

| Metric | Value (NTD) | Difference vs Loss Proxy |
|
| **Realized Loss Proxy** | 82,155,456 | 0.0% |
| **Model EL (RF)** | 79,382,918 | -3.4% |
| **Model EL (LR)** | 76,643,926 | -6.7% |

The model's EL is 3.4% below the proxy. The bootstrap 95% CI of the difference `[-7.72M, +2.21M]` includes zero, so the gap cannot be distinguished from sampling noise; the point estimate errs slightly on the non-conservative side (under the proxy).

---

## Limitations
1. **Horizon mismatch:** The target variable is 1-month payment default. Basel/IRB PDs are 12-month models. The EL calculated here is a 1-month scenario.
2. **EAD proxy:** `LIMIT_BAL` (approved credit limit) overstates actual outstanding exposure for under-utilised accounts. 
3. **LGD assumption:** An illustrative 45% LGD was applied (inspired by the Basel II F-IRB senior unsecured corporate baseline). Actual retail LGDs are bank-estimated and subject to different floors. The Loss Proxy backtest cannot validate this LGD.
4. **Validation Scope:** Evaluated on a single 6,000-record test split. 
5. **Point-in-Time:** The dataset is restricted to Taiwan (April-September 2005), reflecting a Point-in-Time (PiT) economic environment rather than a Through-the-Cycle (TTC) model.

---

## Review Notes: The Impact of `class_weight` on PD Calibration
A self-review of v1.1 revealed that training Random Forest with `class_weight='balanced'` artificially inflated the predicted probabilities well above the base rate. For context, a naive base-rate prediction yields a Brier score of ~0.1723. The v1.1 balanced model performed worse than this naive baseline (OOF Brier ~0.1758). 

I performed a 5-fold out-of-fold CV ablation study to select the best calibration strategy. Results showed the optimal solution was simply removing `class_weight`.

| Strategy | OOF Brier (Train) | Held-out Brier (Test) | Mean Predicted - Observed (Test) |
|---|
| **v1.1 (RF, class_weight='balanced')** | ~0.1758 | 0.1764 | +19.63 pp |
| **v1.2 (RF, Unweighted / No Calibration)** | 0.1338 | 0.1356 | +0.07 pp |


| **v1.1 (RF, `class_weight='balanced'`)** | 0.1758 | +19.63 pp |
| **v1.2 (RF, Unweighted / No Calibration)** | 0.1356 | +0.07 pp |

---

## Reproducibility
All reported metrics are generated from the codebase and are fully reproducible by executing the notebooks 01 through 05 in order. Please ensure the raw dataset `default_of_credit_card_clients.csv` is placed in the `data/raw/` directory before running.