# SECOM Defect Prediction and SPC Feature Screening

This project uses the **UCI SECOM dataset** to analyze semiconductor manufacturing process data. The dataset contains high-dimensional sensor and process measurements collected from a semiconductor production line, together with pass/fail yield labels.

The project has two main objectives:

1. **Train predictive models** to identify likely failed units from process measurements.
2. **Screen and rank important features** so that selected variables can be considered as candidates for an SPC monitoring list.

The final model results are compared against the **baseline results reported by the UCI SECOM dataset page**, where basic feature selection methods with a simple kernel ridge classifier were evaluated using 10-fold cross validation and Balanced Error Rate (BER).

---

## Dataset

Dataset source:

> McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository.  
> DOI: https://doi.org/10.24432/C54305

The UCI SECOM dataset describes a semiconductor manufacturing process. Each row represents one production entity, and the label indicates whether the unit passed or failed final testing.

Basic dataset characteristics:

- 1,567 samples
- 591 original columns, including process features and timestamp information
- Binary target:
  - `-1`: Pass
  - `1`: Fail
- Highly imbalanced labels:
  - 1,463 pass samples
  - 104 fail samples
- High-dimensional feature space
- Missing values
- Many low-information or low-variance features

In this project, the target label is converted to:

```text
0 = Pass
1 = Fail
```

---

## Project Goals

### Goal 1: Predictive Modeling

The first goal is to build machine learning models that can predict failed semiconductor units from process measurements.

Because the dataset is highly imbalanced, model evaluation focuses on metrics that are more suitable than simple accuracy:

- Average Precision
- F1 Score
- Balanced Accuracy
- Recall
- Specificity
- MCC
- Type I Error
- Type II Error
- Cost-weighted error

The training pipeline compares multiple feature sets and model families, including:

- Logistic Regression
- Random Forest
- XGBoost
- Histogram Gradient Boosting
- Support Vector Machine
- Balanced Random Forest
- Isolation Forest
- Local Outlier Factor
- Score-fusion Voting Ensemble

The pipeline also evaluates different imbalance-handling methods:

- No resampling
- Class weights
- SMOTE
- ROSE-style random oversampling

---

### Goal 2: SPC Candidate Feature Screening

The second goal is to identify a smaller set of important process variables that may be useful for **Statistical Process Control (SPC)**.

The feature screening process is designed to remove noisy, redundant, incomplete, or low-information features before model training and interpretation.

The selected features can be used as an initial SPC candidate list for later process-monitoring analysis.

Feature screening includes:

- Removing duplicated features
- Removing features with high missing rates
- Adding missing-value indicator features
- Removing zero-variance features
- Removing near-constant features
- Standardization
- KNN imputation
- Correlation-based feature reduction
- RF-RFE feature selection
- XGB-RFE feature selection
- ANOVA SelectKBest feature selection
- SHAP-based feature attribution

The final SPC feature list should be selected by considering both:

1. **Predictive importance** from the machine learning models
2. **Engineering interpretability** for manufacturing process monitoring

---

## Baseline Comparison

The UCI SECOM dataset page reports baseline results using basic feature selection techniques with a simple kernel ridge classifier and 10-fold cross validation. The reported metric is **Balanced Error Rate (BER)**.

The UCI baseline methods include:

| Baseline Feature Selection Method | BER |
|---|---:|
| Signal-to-Noise | 34.5 ± 2.6 |
| T-test | 33.7 ± 2.1 |
| Relief | 40.1 ± 2.8 |
| Pearson | 34.1 ± 2.0 |
| F-test | 33.5 ± 2.2 |
| Gram-Schmidt | 35.6 ± 2.4 |

This project compares its final model performance against these baseline results.

Since this project reports additional metrics such as F1 Score, Average Precision, Recall, and Balanced Accuracy, BER can be computed for direct comparison:

```text
BER = 1 - Balanced Accuracy
```

or equivalently:

```text
BER = (False Positive Rate + False Negative Rate) / 2
```

---

## Project Structure

```text
secom-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── uci-secom.csv
│   └── processed/
│
├── models/
│
├── outputs/
│   ├── eda/
│   ├── preprocessing/
│   ├── training/
│   ├── shap/
│   ├── spc_values/
│   └── spc_plots/
│
└── scripts/
    ├── 01_eda.py
    ├── 02_preprocess.py
    ├── 03_train_models.py
    ├── 04_shap_analysis.py
    ├── 05_spc_values.py
    └── 06_plot_spc.py
```

---

## Workflow

Run the scripts in order.

### 1. Exploratory Data Analysis

```bash
python scripts/01_eda.py
```

Outputs:

```text
outputs/eda/
```

The EDA script highlights:

- Dataset size
- High-dimensional structure
- Class imbalance
- Missing-value patterns
- Low-variance features

---

### 2. Data Preprocessing and Feature Selection

```bash
python scripts/02_preprocess.py
```

Outputs:

```text
data/processed/
outputs/preprocessing/
models/
```

Main processed datasets:

```text
data/processed/X_train_base.csv
data/processed/X_test_base.csv
data/processed/X_train_rferf.csv
data/processed/X_test_rferf.csv
data/processed/X_train_rfexgb.csv
data/processed/X_test_rfexgb.csv
data/processed/X_train_anova.csv
data/processed/X_test_anova.csv
data/processed/y_train.csv
data/processed/y_test.csv
```

---

### 3. Model Training and Evaluation

```bash
python scripts/03_train_models.py
```

Outputs:

```text
outputs/training/all_model_evaluation_summary.csv
outputs/training/all_model_evaluation_detail.csv
outputs/training/best_model_summary.json
models/all_best_models_and_thresholds.joblib
```

The training script ranks models by:

1. F1 Score
2. Average Precision
3. Balanced Accuracy
4. MCC

---

### 4. SHAP Feature Attribution

```bash
python scripts/04_shap_analysis.py
```

Outputs:

```text
outputs/shap/
```

This step identifies important features from selected high-performing models and supports root-cause analysis.

---

### 5. SPC Value Calculation

```bash
python scripts/05_spc_values.py
```

Outputs:

```text
outputs/spc_values/
```

This step calculates SPC-related values for the selected candidate features.

---

### 6. SPC Plotting

```bash
python scripts/06_plot_spc.py
```

Outputs:

```text
outputs/spc_plots/
```

This step generates SPC charts for process-monitoring interpretation.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

A minimal `requirements.txt` should include:

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
imbalanced-learn
feature-engine
shap
joblib
scipy
```

Optional packages:

```text
seaborn
plotly
missingno
```

---

## Notes on Data Files

Place the raw dataset here:

```text
data/raw/uci-secom.csv
```

Generated files should not usually be committed to GitHub:

```text
data/processed/
outputs/
models/
pipeline_cache/
```

These folders can be regenerated by running the scripts.

---

## Expected Final Outputs

The main project outputs are:

```text
outputs/training/all_model_evaluation_summary.csv
outputs/training/best_model_summary.json
outputs/shap/
outputs/spc_values/
outputs/spc_plots/
```

These outputs support the final comparison:

1. Predictive model performance vs. UCI baseline BER
2. Selected feature sets for potential SPC monitoring
3. SHAP-based feature importance and root-cause interpretation
4. SPC visualization for process control review
