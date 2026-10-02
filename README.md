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

**Before vs After Calibration:**
| Metric | v1.1 (RF Raw, class_weight='balanced') | v1.2 (RF Isotonic, unweighted) |
|---|---|---|
| Mean Predicted PD | 41.74% | 22.19% |
| CITL (Bias) | +19.63 pp | +0.07 pp |
| ROC-AUC | 0.7750 | 0.7748 |
| Brier Score | 0.1758 | 0.1356 |
| Model Expected Loss | 160,805,361 NTD | 79,382,918 NTD |
| Realized Loss | 82,155,456 NTD | 82,155,456 NTD |

### 1. Expected Loss Scenario & Backtesting

The Basel-standard Expected Loss formula is applied as a simplified scenario:
`EL = PD × LGD × Exposure Proxy`

#### Realized Loss Backtest (Test Set)
Realized Loss = `LIMIT_BAL` × `Actual_Default` × `45%`. This backtest is consistent with the PD calibration and Exposure proxy. It cannot validate LGD, as the same 45% assumption enters both sides. Note that all variants fall slightly below realized loss, and their estimates are highly correlated.

| Metric | Value (NTD) | Difference vs Realized |
|---|---|---|
| **Realized Loss** | 82,155,456 | 0.0% |
| **Model EL (LR)** | 76,643,926 | -6.7% |
| **Model EL (RF Calibrated)** | 79,382,918 | -3.4% |

---

## Limitations

1. **Horizon mismatch**: The target variable in this dataset is 1-month payment default. Basel/IRB PDs are 12-month models. The EL calculated here is a 1-month expected loss scenario by design.
2. **Default definition**: The dataset defines default as "payment default next month", not the standard Basel 90 Days Past Due (DPD) or Unlikely to Pay (UTP) definitions.
3. **EAD proxy**: `LIMIT_BAL` (approved credit limit) overstates actual outstanding exposure for under-utilised accounts. True EAD modelling requires outstanding balance data.
4. **LGD assumption**: An illustrative 45% LGD was applied (inspired by F-IRB senior unsecured corporate baselines, for scenario purposes only; actual retail/QRRE LGDs are bank-estimated and subject to different floors). The realized-loss backtest cannot validate this LGD, since the same 45% enters both sides.

---

## Reproducibility

All reported metrics are generated from the codebase and are fully reproducible by executing the notebooks 01 through 05 in order.
