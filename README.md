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

## Project Structure

```
credit-risk-scoring-analysis/
│
├── data/
│   ├── raw/                           # Original UCI dataset (gitignored)
│   └── processed/                     # Cleaned CSV + model predictions (gitignored)
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda_and_data_cleaning.ipynb
│   ├── 03_sql_risk_analysis.ipynb
│   ├── 04_pd_modeling.ipynb
│   └── 05_expected_loss_analysis.ipynb
│
├── figures/                           # Saved charts referenced in notebooks
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Methodology & Findings

### v1.1 -> v1.2 Validation Notes
An external audit of v1.1 identified that models were trained with `class_weight='balanced'`, which inflated predicted PDs well above the true base rate and thus inflated Expected Loss. 

**Nested CV Calibration Ablation (Train Folds Only):**
I evaluated probability calibration strictly via 5-fold nested cross-validation on the training set to prevent data leakage. The calibration selection rule required an out-of-fold Brier score improvement of at least 0.0005.

| Variant | OOF Brier | Mean Predicted - Observed |
|---|---|---|
| Raw Unweighted RF | 0.13379 | 0.000 |
| RF + Isotonic | 0.13368 | 0.000 |
| RF + Sigmoid | 0.13390 | 0.000 |

*Result:* Calibration provided no material gain; the true fix was removing `class_weight`. The simplest model (Raw Unweighted RF) was selected.

**Before vs After (Test Set):**
| Metric | v1.1 (RF Raw, class_weight='balanced') | v1.2 (RF Selected, unweighted) |
|---|---|---|
| Mean Predicted PD | 41.74% | 22.19% |
| CITL (Bias) | +19.63 pp | +0.07 pp |
| ROC-AUC | 0.7750 | 0.7748 |
| Brier Score | 0.1758 | 0.1356 |
| Total Expected Defaults | 2,504 | 1,331.3 |
| Model Expected Loss | 160,805,361 NTD | 79,382,918 NTD |

### 1. SQL Roll-Rate Analysis
A Delinquency Migration matrix was constructed using window functions (`LAG()`) to track month-over-month account status. The transition rate from 1-month to 2-month delinquency is 32.0%, identifying the critical point of portfolio deterioration where early intervention is most effective.

### 2. Expected Loss Scenario & Backtesting

The expected loss is calculated using a simplified Basel-style EL scenario:
`EL = PD × LGD × Exposure Proxy`

#### Loss Proxy Backtest (Test Set)
Loss Proxy = `LIMIT_BAL` × `Actual_Default` × `45%`. This backtest is consistent with the PD calibration and Exposure proxy. It cannot validate LGD, as the same 45% assumption enters both sides. Note that all variants fall slightly below the realized loss proxy, and their estimates are highly correlated.

| Metric | Value (NTD) | Difference vs Loss Proxy |
|---|---|---|
| **Loss Proxy** | 82,155,456 | 0.0% |
| **Model EL (LR)** | 76,643,926 | -6.7% |
| **Model EL (RF Selected)** | 79,382,918 | -3.4% |

**95% Bootstrap CI of Difference (RF EL - Loss Proxy):** [-7,720,421, 2,208,886] NTD

The model demonstrates monotonic rank-ordering capability, properly segmenting risk across the 5-grade master scale.

---

## Limitations

1. **Horizon mismatch**: The target variable in this dataset is 1-month payment default. Basel/IRB PDs are 12-month models. The EL calculated here is a 1-month expected loss scenario by design.
2. **Default definition**: The dataset defines default as "payment default next month", not the standard Basel 90 Days Past Due (DPD) or Unlikely to Pay (UTP) definitions.
3. **EAD proxy**: `LIMIT_BAL` (approved credit limit) overstates actual outstanding exposure for under-utilised accounts. True EAD modelling requires outstanding balance data.
4. **LGD assumption**: An illustrative 45% LGD was applied (Basel II F-IRB senior unsecured corporate baseline, for scenario purposes only; actual retail/QRRE LGDs are bank-estimated and subject to different floors). The realized-loss backtest cannot validate this LGD, since the same 45% enters both sides.

---

## Reproducibility

All reported metrics are generated from the codebase and are fully reproducible by executing the notebooks 01 through 05 in order. Ensure the raw CSV file `default_of_credit_card_clients.csv` is placed in `data/raw/` before execution.
