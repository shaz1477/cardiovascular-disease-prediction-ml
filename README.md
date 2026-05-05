# Cardiovascular Disease Prediction using Machine Learning

Predicting cardiovascular disease presence on 70,000 patient records using 
7 ML algorithms with hyperparameter tuning, domain-driven data cleaning, 
and feature importance analysis.

## Model Comparison

| Model | Train Acc | CV Acc | Test Acc |
|---|---|---|---|
| **XGBoost** ✅ | 74.52% | 73.35% | 73.45% |
| Gradient Boost | 73.73% | 73.44% | 73.48% |
| Decision Tree | 73.06% | 72.98% | 73.14% |
| AdaBoost | 72.89% | 72.89% | 72.76% |
| KNN | 72.63% | 72.13% | 72.08% |
| Logistic Regression | 71.49% | 71.43% | 71.38% |
| Random Forest | 97.84% | 70.16% | 70.71% ⚠️ overfit |

**XGBoost selected** — most consistent across train/CV/test with no overfitting gap (F1: 0.73).

## XGBoost Feature Importance

| Feature | Importance |
|---|---|
| ap_hi (Systolic BP) | **61.4%** |
| Cholesterol | 14.7% |
| Age | 7.97% |
| Others | <5% each |

**Finding:** Systolic blood pressure dominates cardiovascular risk prediction — 
consistent with established medical research on hypertension as the leading 
modifiable risk factor.

## Key Technical Decisions

- **Domain-driven outlier removal** over IQR replacement — physiologically 
  impossible BP values (-150, 11500 mmHg) removed; clinically valid 
  hypertension readings (180–200 mmHg) preserved
- **GridSearchCV** used across all 7 models for hyperparameter tuning
- **Random Forest rejected** despite high train accuracy due to 27% 
  train-test gap (overfitting)

## Dataset

Kaggle Cardiovascular Disease Dataset — 70,000 records, 11 features  
(age, gender, height, weight, systolic/diastolic BP, cholesterol, 
glucose, smoking, alcohol, physical activity)

After cleaning: **68,509 records**

## Tech Stack

Python · XGBoost · Scikit-learn · Pandas · NumPy · Matplotlib · GridSearchCV

## Setup

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

Update dataset path in notebook:
```python
df = pd.read_csv("cardio_train.csv", sep=";")
```

Then run: `Cardiovascular Disease Dataset-checkpoint.ipynb`
