# Medical Insurance Cost Prediction - EDA and Machine Learning

This notebook presents a full workflow for predicting individual medical insurance costs based on demographic and health-related features.

## Objective

To model and predict insurance charges using features such as age, BMI, smoking status, region, and number of dependents.

## Workflow

### 1. Data Preprocessing
- Load dataset from CSV
- Encode categorical features (sex, smoker, region)
- Handle missing values if present
- Feature scaling (StandardScaler)

### 2. Exploratory Data Analysis (EDA)
- Histograms and boxplots to explore distributions
- Correlation heatmaps
- Analysis of categorical impacts (e.g., smoker vs. non-smoker)

### 3. Model Building
- Regression models used:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
- Model evaluation:
  - R² Score
  - MAE, RMSE

### 4. Model Comparison
- Tabulated metrics across all models

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

## Instructions

1. Clone this repository
2. Place the dataset file in the same directory
3. Open `4_Insurance_EDA_ML.ipynb`
4. Execute all cells in order

## Output

- Cleaned dataset and visual EDA
- Trained regression models
- Evaluation of model performance

