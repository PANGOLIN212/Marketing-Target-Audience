# Marketing Target Audience Prediction

A machine learning project that predicts which bank customers are most likely to respond to a term-deposit marketing offer, so campaigns can focus on the customers most likely to say "yes" instead of contacting everyone.

## Overview

The notebook `Marketing_target_audience.ipynb` walks through a complete binary-classification workflow:

1. Load and explore a bank marketing dataset
2. Preprocess numerical and categorical features
3. Train and compare two models (Logistic Regression and Random Forest)
4. Score customers by their probability of responding
5. Export a prioritized list of high-probability customers

## Dataset

- **File:** `bank.csv` (bank marketing campaign data)
- **Size:** 11,162 customers, 17 columns
- **Target:** `deposit` (`yes` = subscribed to a term deposit, `no` = did not)
- **Class balance:** 5,873 `no` / 5,289 `yes`, which is fairly balanced
- **Missing values:** none

| Type | Features |
|---|---|
| Numerical | `age`, `balance`, `day`, `campaign`, `pdays`, `previous` |
| Categorical | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` |
| Dropped | `duration` |

**Why `duration` is dropped:** call duration is only known *after* the call is made, so using it would leak information and make the model unusable for deciding who to contact in advance.

## Methodology

- **Split:** 80/20 train/test, stratified on the target (`random_state=42`)
- **Preprocessing** (via `ColumnTransformer`):
  - `StandardScaler` for numerical features
  - `OneHotEncoder(handle_unknown="ignore")` for categorical features
- **Models** (each wrapped in a scikit-learn `Pipeline`):
  - Logistic Regression (`max_iter=1000`, `class_weight="balanced"`)
  - Random Forest (`n_estimators=300`, `max_depth=12`, `class_weight="balanced"`)
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve

## Results

Evaluated on the 2,233-customer hold-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.693 | 0.712 | 0.593 | 0.647 | 0.758 |
| Random Forest | **0.731** | **0.761** | **0.630** | **0.690** | **0.787** |

Random Forest outperforms Logistic Regression on every metric.

## Marketing List Output

The notebook ranks test-set customers by predicted response probability and exports those with a probability of **0.70 or higher** (458 customers) to `customers_likely_to_respond.csv`. Columns include age, job, marital status, education, balance, housing and loan status, contact method, month, and `Response_Probability`.

## Getting Started

### Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run

1. Place `bank.csv` in the project folder.
2. Update the data path in the second code cell. It currently points to a local Windows path:
   ```python
   df = pd.read_csv("bank.csv")
   ```
3. Launch and run all cells:
   ```bash
   jupyter notebook Marketing_target_audience.ipynb
   ```

## Project Structure

```
.
├── Marketing_target_audience.ipynb    # Analysis and modeling
├── bank.csv                           # Input data (not included)
├── customers_likely_to_respond.csv    # Generated marketing list
└── README.md
```

## Known Issues and Suggested Improvements

- **F1 print bug:** in the Logistic Regression results cell, the "F1 Score" line prints `recall` instead of `f1`. The comparison table uses the correct value (0.647).
- **Model mismatch:** the marketing list is built from Logistic Regression probabilities, although Random Forest performs better. Switching to `rf_probability` would likely improve targeting.
- **Hard-coded path:** replace the absolute file path with a relative one for portability.
- **Next steps:** hyperparameter tuning (e.g. `GridSearchCV`), cross-validation, feature importance analysis, and threshold tuning based on campaign cost vs. conversion value.

Dataset Credit

This appears to be the widely used Bank Marketing dataset (Moro et al.), available on the UCI Machine Learning Repository and Kaggle. Please check the original source for licensing and citation requirements.
