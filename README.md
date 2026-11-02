# Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecommunications customer is likely to leave a service using Logistic Regression and an interactive Streamlit application.

## Project Overview

Customer churn is an important business problem for telecommunications companies. Identifying customers who are likely to leave can help businesses understand customer behaviour and improve retention strategies.

This project uses customer demographic, service and account information to predict churn.

## Application Preview

The application allows users to enter customer details and receive:
- A customer churn prediction (Yes/No).
- An estimated churn probability.

## Technologies Used

- **Python** — Programming language
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Machine Learning and preprocessing
- **Logistic Regression** — Classification algorithm
- **Streamlit** — Interactive web application
- **Joblib** — Model serialization
- **Jupyter Notebook** — Data exploration and model development

## Machine Learning Workflow

1. Data loading and exploration
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Train-test split using stratification
5. Numerical feature scaling using StandardScaler
6. Categorical feature encoding using OneHotEncoder
7. Logistic Regression model training
8. Class imbalance handling using `class_weight="balanced"`
9. Model evaluation
10. Saving the complete preprocessing and prediction pipeline using Joblib
11. Building an interactive Streamlit application

## Model Performance

The balanced Logistic Regression model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 74.35% |
| Precision | 51.09% |
| Recall | 79.68% |
| F1-score | 62.26% |

The model prioritizes identifying customers who may churn, resulting in higher recall. This comes with a trade-off in precision and accuracy.

## Project Structure

```text
Customer_Churn/
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── EDA.ipynb
├── app.py
├── churn_pipeline.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/SahilJadhav-007/Customer-Churn-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Customer-Churn-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## Future Improvements

- Experiment with additional classification algorithms.
- Perform hyperparameter tuning.
- Improve probability calibration.
- Enhance the application interface.
- Deploy the application online.

## Disclaimer

The predictions are machine learning estimates and should not be treated as guaranteed customer behaviour or as the sole basis for business decisions.
