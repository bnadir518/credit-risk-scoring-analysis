# Credit Risk Scoring & Expected Loss Analysis

A portfolio project demonstrating end-to-end credit risk analytics: data preparation, SQL-based portfolio analysis, PD modelling (Logistic Regression and Random Forest), probability calibration, and Expected Loss scenario analysis under the Basel standard EL framework.

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

### 1. SQL Risk Analysis (Delinquency Migration)
Beyond basic aggregations, the project utilises advanced SQL (CTEs, Window Functions like `LAG()`) to build a **Delinquency Migration (Roll-rate) matrix**, tracking how accounts transition across repayment statuses month-over-month.

### 2. PD Modelling (Probability of Default)
The project compares a Logistic Regression baseline with a Random Forest model. Unweighted models were deliberately chosen to preserve the natural probability scale, ensuring the Mean Predicted PD closely matches the observed default rate (22.1%).

**Test Set Performance:**
| Metric | Logistic Regression | Random Forest (Unweighted) |
|---|---|---|
| **ROC-AUC** | 0.7084 | 0.7734 |
| **Gini** | 0.4168 | 0.5468 |
| **KS Statistic** | 0.2850 | 0.4237 |
| **Brier Score** | 0.1685 | 0.1354 |

### 3. Risk Segmentation (5-Grade Master Scale)
Clients are segmented into a 5-grade scale based on fixed PD cutoffs. The model demonstrates strong rank-ordering capabilities.

| Grade | Clients | Avg_PD | Obs_Default | Expected_Def | Actual_Def | EL (NTD) |
|---|---|---|---|---|---|---|
| **1. Very Low (<5%)** | 431 | 3.5% | 4.9% | 15.08 | 21 | 1,997,425 |
| **2. Low (5-15%)** | 2,509 | 10.3% | 10.5% | 258.32 | 264 | 21,835,470 |
| **3. Medium (15-30%)** | 1,691 | 20.3% | 18.6% | 344.00 | 314 | 20,973,880 |
| **4. High (30-60%)** | 889 | 42.8% | 43.5% | 380.21 | 387 | 20,298,420 |
| **5. Very High (>=60%)** | 480 | 69.5% | 71.0% | 333.64 | 341 | 14,277,730 |

### 4. Expected Loss Scenario & Backtesting
The Basel-standard Expected Loss formula is applied as a simplified scenario:
`EL = 1-month PD × LGD × Exposure Proxy`

**Default-based Loss Proxy Backtest:**
Loss proxy = `LIMIT_BAL` × `actual_default` × `45%`. Because the same 45% LGD enters both sides, it cancels out; this backtest tests exposure-weighted PD calibration only, not loss severity. 

| Metric | Value (NTD) | Difference vs Loss Proxy |
|---|---|---|
| **Realized Loss Proxy** | 82,155,456 | 0.0% |
| **Model EL (RF)** | 79,382,918 | -3.4% |
| **Model EL (LR)** | 76,643,926 | -6.7% |

The model's EL is 3.4% below the proxy. The bootstrap 95% CI of the difference `[-7.72M, +2.21M]` includes zero, so the gap cannot be statistically distinguished from sampling noise, though the point estimate errs slightly on the non-conservative side (underestimation in the Very Low tier).

---

## Limitations
1. **Horizon mismatch:** The target variable is 1-month payment default. Basel/IRB PDs are 12-month models. The EL calculated here is a 1-month scenario, which results in a monthly EL/Exposure ratio (~7.9%) that would be extremely high if interpreted as an annualized figure.
2. **EAD proxy:** `LIMIT_BAL` (approved credit limit) overstates actual outstanding exposure for under-utilised accounts. True EAD modelling requires outstanding balance data.
3. **LGD assumption:** An illustrative 45% LGD was applied (inspired by Basel II/CRR2 senior unsecured corporate baselines). Actual retail/QRRE LGDs are bank-estimated and subject to different floors (e.g., 50%).
4. **Validation Scope:** Evaluated on a single 6,000-record test split. 
5. **Point-in-Time:** The dataset is restricted to Taiwan (April-September 2005), reflecting a Point-in-Time (PiT) economic environment rather than a Through-the-Cycle (TTC) model.

---

## Review Notes: The Impact of `class_weight` on PD Calibration
Initial self-review (v1.1) revealed that training Random Forest with `class_weight='balanced'` artificially inflated the predicted probabilities well above the base rate. 

We performed a nested cross-validation ablation study to select the best calibration strategy. Results showed that applying Isotonic or Sigmoid calibration over the unweighted RF yielded no material Brier score gain (< 0.0005). The optimal solution was simply removing `class_weight`.

| Metric | v1.1 (RF, class_weight='balanced') | v1.2 (RF, Unweighted / No Calibration) |
|---|---|---|
| **Mean Predicted PD** | 41.74% | 22.17% |
| **Mean predicted - observed** | +19.63 pp | +0.05 pp |
| **Brier Score** | 0.1764 (Worse than 0.172 base rate) | 0.1354 |
| **Model Expected Loss** | 160,805,361 NTD | 79,382,918 NTD |

---

## Reproducibility
All reported metrics are generated from the codebase and are fully reproducible by executing the notebooks 01 through 05 in order.