# ML-PROJECT-PREMIUM-HEALTH-PREDICTION
An end-to-end machine learning project for predicting annual health insurance premiums from customer demographic, financial, lifestyle, and medical attributes.

## Introduction
Health insurance premiums depend on multiple customer attributes such as age, income, dependant , lifestyle, BMI, employment, insurance plan, and medical history. A single model may not capture the behavior of different age groups equally well.
This project therefore builds separate prediction pipelines for:
Young: Age ≤ 25
Rest: Age > 25
The final system predicts the customer's Annual Premium Amount.

Dataset used is health-insurance customer dataset containing demographic, financial, lifestyle, medical, and insurance information. Original dataset is divided into two age groups.
After missing-value removal and outlier handling, the modeling datasets are further cleaned before training.

## Features
The model uses the following features:
Numerical/Ordinal Features: 
  Age, 
  Number of Dependants, 
  Income in Lakhs,
  Insurance Plan,
  Genetical Risk,
  Normalized Medical Risk Score
  
Categorical Features:
  Gender,
  Region,
  Marital Status,
  BMI Category,
  Smoking Status,
  Employment Status

Target: Annual Premium Amount

## Machine Learning Pipeline:
1. Age Based Segmentation : The original dataset hence producing two separate modeling datasets.
2. Data Cleaning : The notebooks perform several data-quality checks and transformation.
   Missing values are identified and rows containing missing values are removed.
   Invalid negative dependant values are converted to their absolute values.
   Age values above 100 are treated as invalid. Since the segmented datasets already contain valid age ranges, the final datasets contain realistic ages.
   Income distributions are examined using quartiles/IQR statistics. The final filtering uses the 99.9th percentile as the upper threshold. This removes extreme income observations while retaining the majority of the data.
3. EDA is performed to understand Numerical feature distributions, outliers, categorical values, income distribution, age distribution, Relationship between features and premium.
4. Medical Feature Engineering : Medical history is originally represented as text, including combinations such as Diabetes, Heart disease, High blood pressure, Diabetes & Thyroid and Diabetes & Heart disease. The medical history field is split into individual disease components. A domain-based risk mapping is applied. For combined conditions, the individual scores are added. The resulting total risk is normalized using normalized_risk_score = (total_risk_score-min_score)/(max_score-min_score).The raw medical-history and intermediate disease columns are then removed, leaving the engineered normalized risk score.
5. Encoding: Ordinal and One Hot Encoding is done
6. Feature Scaling : Selected numerical/ordinal features are scaled using MinMaxScaler. The resulting scaler and the list of scaled columns are serialized for use during deployment.
7. Multicollinearity Analysis with VIF : Variance Inflation Factor (VIF) is calculated to identify redundant relationships among input features. The analysis showed high VIF for: income_level and income_lakhs. So income_level is removed from the final modeling feature set.
8. Train/Test Split : Each segmented dataset is divided into 70% training and 30% testing. The fixed random state makes the experiment reproducible.
9. Model Development : Two models are evaluated. Linear Regression is used as the final Young-segment model. For Rest segment, Linear Regression was evaluated as a baseline. XGBoost substantially improved the baseline. XGBoost model is tuned using Randomized Search CV. The resulting tuned XGBoost model is selected as the final Rest-segment model.
10. Extreme Error Analysis : The project evaluates prediction errors using percentage error. Prediction is classified as an extreme error when Absolute percentage Error > 10%.
11. Model Serialization : After selecting the final models, they are saved using joblib. The scaler artifact stores both : the fitted MinMaxScaler, the columns that need to be scaled. This allows the deployment code to reproduce the same scaling operation used during model training.
12. Deployment : The project includes a Streamlit Web Application.

## Tech Stack
1. Python
2. Pandas : Data Manipulation
3. NumPy : numerical computation
4. Scikit-learn : preprocessing, Linear Regression, evaluation, hyperparameter search
5. XGBoost : nonlinear regression
6. Statsmodels : VIF analysis
7. Matplotlib / Seaborn : visualization and EDA
8. Joblib : model/scaler serialization
9. Streamlit : deployment and interactive prediction interface

## Installation
Clone the repository: git clone https://github.com/s01kaur/ML-PROJECT-PREMIUM-HEALTH-PREDICTION.git
cd ML-PROJECT-PREMIUM-HEALTH-PREDICTION

Create and activate a virtual environment

Install dependencies: pip install -r requirements.txt

## Run the Application
From the project root: streamlit run/main.py
The application will open in the browser
