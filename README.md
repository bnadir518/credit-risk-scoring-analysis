# Credit Risk Scoring & Expected Loss Analysis

A portfolio project demonstrating end-to-end credit risk analytics: data preparation, SQL-based portfolio analysis, PD modelling (Logistic Regression and Random Forest), and Expected Loss scenario analysis under the Basel standard EL framework.

---

## Dataset

| Field | Detail |
|---|---|
| **Source** | [UCI Machine Learning Repository — Default of Credit Card Clients (ID 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| **Citation** | Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480. |
| **Instances** | 30,000 credit card clients (Taiwan, April–September 2005) |
| **Features** | 23 predictors: demographics, repayment history, bill amounts, payment amounts |
| **Target** | `DEFAULT_NEXT_MONTH` — binary (1 = defaulted, 0 = did not default) |
| **Default rate** | 22.12% (class imbalance: 4.5:1 non-default : default) |

---

## Project Structure

```
credit-risk-scoring-analysis/
│
├── data/
│   ├── raw/                           # Original UCI dataset (gitignored — download separately)
│   └── processed/                     # Cleaned CSV + model predictions (gitignored)
│
├── notebooks/
│   ├── 01_data_loading.ipynb          # Column renaming, categorical remapping, save cleaned CSV
│   ├── 02_eda_and_data_cleaning.ipynb # Class distribution, default by PAY_0, credit limit EDA
│   ├── 03_sql_risk_analysis.ipynb     # SQLite queries: GROUP BY, CASE WHEN, CTEs, NTILE
│   ├── 04_pd_modeling.ipynb           # Logistic Regression + Random Forest, evaluation, feature importance
│   └── 05_expected_loss_analysis.ipynb# EL scenario analysis, risk tiers, LGD sensitivity
│
├── figures/                           # Saved charts referenced in notebooks
├── .gitignore
├── README.md
└── requirements.txt
```

> **To run**: Download the dataset from the UCI link above, place `default_of_credit_card_clients.csv` in `data/raw/`, then execute the notebooks in order (01 → 05).

---

## Technologies

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data | pandas, numpy |
| Visualisation | matplotlib |
| Machine Learning | scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler) |
| SQL | SQLite (in-memory, via Python `sqlite3`) |
| Notebooks | Jupyter / nbconvert |

---

## Methodology

### 1. Data Preparation (Notebook 01)

The raw UCI dataset uses coded column names (`X1`–`X23`, `Y`). These are renamed to meaningful field names. Two categorical features contain undocumented codes that are remapped following standard practice:

- **EDUCATION**: Codes 0, 5, 6 (undocumented) → 4 (Other)
- **MARRIAGE**: Code 0 (undocumented) → 3 (Other)

The cleaned dataset is saved to `data/processed/` and used by all subsequent notebooks.

### 2. Exploratory Data Analysis (Notebook 02)

Key findings from EDA:

- **Class imbalance**: 77.88% non-default vs. 22.12% default — straightforward accuracy is unreliable
- **Dominant predictor**: `PAY_0` (most recent repayment status) — default rate increases sharply at 2+ month delays
- **Credit limit**: Defaulters have lower average credit limits (~NTD 130K vs. ~NTD 178K for non-defaulters), but credit limit alone is a weak predictor

### 3. SQL Risk Analysis (Notebook 03)

The cleaned dataset is loaded into an in-memory SQLite database to simulate a relational banking environment. Four queries are executed:

| Query | SQL Techniques | Finding |
|---|---|---|
| Default by education level | `GROUP BY`, `CASE WHEN` | Graduate-school clients have lowest default rates |
| Default by repayment status (PAY_0) | `GROUP BY` | Sharp default spike at PAY_0 ≥ 2 |
| Bill-amount quartile analysis | CTE + `NTILE(4)` window function | Non-linear relationship between bill amount and default |
| Portfolio exposure concentration | `CASE WHEN` bands + `SUM` | Majority of exposure in NTD 100K–300K credit limit band |

### 4. PD Modelling (Notebook 04)

Two models are trained on an 80/20 stratified train/test split (`random_state=42`).

#### Preprocessing

- **Logistic Regression**: `StandardScaler` applied (required for gradient-based optimisation)
- **Random Forest**: No scaling applied (tree-based models are scale-invariant; applying scaling would be unnecessary and technically misleading)

Both models use `class_weight='balanced'` to account for class imbalance.

#### Model A: Logistic Regression (Baseline)

Logistic Regression is the foundation of credit scorecard methodology. It produces interpretable log-odds coefficients and is the standard baseline before introducing non-linear methods.

| Metric | Value |
|---|---|
| Accuracy | 67.95% |
| Precision (Default) | 0.3671 |
| Recall (Default) | 0.6202 |
| F1 (Default) | 0.4612 |
| ROC-AUC | 0.7084 |
| True Positives | 823 |
| False Negatives | 504 |

#### Model B: Random Forest

Random Forest captures non-linear feature interactions (e.g., combined effect of payment delay and credit limit) and provides feature importance rankings.

| Metric | Value |
|---|---|
| Accuracy | 77.62% |
| Precision (Default) | 0.4949 |
| Recall (Default) | 0.5878 |
| F1 (Default) | 0.5374 |
| ROC-AUC | 0.7750 |
| True Positives | 780 |
| False Negatives | 547 |

#### Why accuracy is not the right metric

A model predicting "Non-Default" for every client would achieve ~78% accuracy while providing zero risk management value. In credit risk:

- **Recall** measures the fraction of actual defaulters detected. Missed defaults (False Negatives) translate directly to unexpected credit losses.
- **ROC-AUC** measures rank-ordering ability across all decision thresholds — the standard discriminatory metric in retail credit model validation.

Random Forest achieves superior rank-ordering (AUC 0.775 vs. 0.708) while Logistic Regression detects more defaulters at the default 0.5 threshold (Recall 62% vs. 59%). The optimal choice depends on the institution's cost structure for False Negatives vs. False Positives.

#### Feature Importance (Random Forest)

The five most important features:

| Feature | Importance |
|---|---|
| PAY_0 (most recent repayment status) | 0.251 |
| PAY_2 | 0.102 |
| PAY_4 | 0.054 |
| PAY_3 | 0.054 |
| LIMIT_BAL | 0.051 |

Consistent with EDA findings: recent repayment behaviour dominates predictive signal.

### 5. Expected Loss Scenario Analysis (Notebook 05)

The Basel-standard Expected Loss formula is applied:

$$EL = PD \times LGD \times Exposure\ Proxy$$

#### Methodology disclosures

> **This is a simplified illustrative scenario, not a production-grade Basel IRB calculation.**

| Component | What is used | Limitation |
|---|---|---|
| **PD** | Model `predict_proba()` output | Not formally calibrated; `class_weight='balanced'` inflates predicted PDs above the true base rate |
| **LGD** | Fixed 45% assumption | Not estimated from data — no recovery information in dataset; consistent with Basel unsecured retail benchmarks |
| **Exposure** | `LIMIT_BAL` (approved credit limit) | **Not true EAD** — LIMIT_BAL is an upper-bound proxy; actual outstanding balance at time of default is not available in this dataset |

#### Results (test set — 6,000 clients)

| Component | Value |
|---|---|
| Total exposure proxy (sum of LIMIT_BAL) | NTD 1,007,777,680 |
| Expected Loss — Random Forest (LGD=45%) | NTD 160,805,361 (15.96%) |
| Expected Loss — Logistic Regression (LGD=45%) | NTD 172,988,073 (17.17%) |

#### Risk Tier Segmentation

Clients are classified into illustrative risk tiers based on predicted PD from the Random Forest model. Thresholds are project-defined, not regulatory capital bands.

| Tier | PD Range | Clients | Avg Predicted PD | Observed Default Rate | Avg Exposure Proxy |
|---|---|---|---|---|---|
| Low | PD < 20% | 511 (8.5%) | 17.1% | 4.3% | NTD 321,018 |
| Medium | 20% ≤ PD < 40% | 3,068 (51.1%) | 30.0% | 11.8% | NTD 183,158 |
| High | PD ≥ 40% | 2,421 (40.4%) | 61.8% | 39.0% | NTD 116,401 |

The observed default rate increases monotonically across tiers, validating their directional relevance.

#### LGD Sensitivity

| LGD Assumption | EL (RF) — NTD M | EL/Exposure Ratio |
|---|---|---|
| 30% | 107.2M | 10.64% |
| 40% | 142.9M | 14.18% |
| **45% (base)** | **160.8M** | **15.96%** |
| 55% | 196.5M | 19.50% |
| 65% | 232.3M | 23.05% |
| 75% | 268.0M | 26.59% |

---

## Limitations

1. **EAD proxy**: `LIMIT_BAL` (approved credit limit) overstates actual outstanding exposure for under-utilised accounts. True EAD modelling requires outstanding balance data at time of default and Credit Conversion Factors (CCF) applied to undrawn commitments.

2. **LGD assumption**: The 45% LGD is an industry benchmark, not estimated from empirical recovery data. Actual recovery rates vary by collateral type, vintage, collection strategy, and macroeconomic environment.

3. **PD calibration**: `class_weight='balanced'` shifts predicted PD distributions above the true base rate. A calibration step (Platt scaling or isotonic regression) would be required before using these probabilities as formal Basel IRB PDs.

4. **Dataset scope**: Taiwan consumer credit, April–September 2005. Behavioural patterns may not generalise to other geographies, products, or economic regimes.

5. **Point-in-time model**: The trained model produces a point-in-time PD. Basel IRB requires through-the-cycle PDs incorporating macroeconomic stress scenarios.

6. **Single train/test split**: Metrics are reported on a single 80/20 split. Cross-validation would provide more robust estimates of out-of-sample performance.

---

## Reproducibility

All reported metrics are generated from the current codebase and are fully reproducible:

```
# Run notebooks in order after placing the raw CSV in data/raw/
01_data_loading.ipynb          → saves data/processed/cleaned_credit_card_data.csv
02_eda_and_data_cleaning.ipynb → reads data/processed/
03_sql_risk_analysis.ipynb     → reads data/processed/
04_pd_modeling.ipynb           → reads data/processed/, saves data/processed/test_predictions.csv
05_expected_loss_analysis.ipynb→ reads data/processed/test_predictions.csv
```

Each notebook is self-contained and loads only from saved files — no shared kernel variables across notebooks.