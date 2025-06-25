# Diabetes Dataset - EDA and Machine Learning

This notebook provides a complete pipeline for analyzing and modeling the Pima Indians Diabetes dataset using various machine learning algorithms.

## Objective

To predict the onset of diabetes based on diagnostic measurements such as glucose level, BMI, insulin, age, and other medical attributes.

## Workflow

### 1. Data Preprocessing
- Load dataset from CSV
- Replace zero values in features like BMI, Insulin, etc., with NaNs
- Handle missing data using mean imputation
- Feature scaling using StandardScaler

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for key features
- Correlation matrix using a heatmap
- Boxplots and pair plots for multivariate insights

### 3. Model Building
- Classification models trained:
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - XGBoost
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

### 4. Model Comparison
- Tabulated performance metrics
- ROC-AUC curves for classifier evaluation

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

## Instructions

1. Clone this repository
2. Ensure the diabetes CSV file is available in the notebook directory
3. Open and run `2_Diabetes_EDA_ML.ipynb` in Jupyter Notebook or similar environment

## Outputs

- Cleaned dataset ready for modeling
- Visualizations for insights
- Classification model performance metrics


