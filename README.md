# Loan-Approval-Prediction-project
# Loan Approval Prediction

An end-to-end machine learning classification project predicting loan approval status using Logistic Regression, built with a complete scikit-learn Pipeline.

## Business Problem
Predict whether a loan applicant will be approved or not, based on their personal, financial, and credit history details.

## Dataset
`loan_data.csv` — includes applicant details such as age, gender, education, income, employment experience, home ownership, loan amount, loan intent, interest rate, credit history length, credit score, previous defaults, and the target column `loan_status` (approved/not approved).

## Tools Used
- Python
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, LogisticRegression)

## Approach
- Cleaned outliers (e.g. invalid age values)
- Built a `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for categorical features
- Combined preprocessing + model into a single `Pipeline` with `LogisticRegression(class_weight='balanced')` to handle class imbalance
- Trained and evaluated the model, then serialized the full pipeline using `pickle` for reuse without retraining

## Results
- **Accuracy**: ~88%
- Precision/Recall (approved class): ~0.73 each
- Train vs test accuracy nearly identical — no overfitting

## How to Run
```bash
pip install -r requirements.txt
```
Then open and run `loan_approval_project(sklearn_pipeline).ipynb`.
