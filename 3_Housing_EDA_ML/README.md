# Housing Price Prediction - EDA and Machine Learning

This repository contains a Jupyter Notebook for performing data cleaning, exploration, and predictive modeling on a housing dataset.

## Objective

To build regression models that predict housing prices based on key features such as crime rate, number of rooms, tax rate, and property characteristics.

## Workflow

### 1. Data Preprocessing
- Load housing dataset (e.g., Boston housing)
- Check and handle missing values
- Standardize column names
- Feature scaling using StandardScaler

### 2. Exploratory Data Analysis (EDA)
- Summary statistics and distribution plots
- Heatmap to visualize feature correlations
- Boxplots and histograms for feature patterns

### 3. Model Building
- Regression models used:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
- Train-test split
- Model evaluation using:
  - R² Score
  - MAE, MSE, RMSE

### 4. Model Comparison
- Performance metrics summarized in tabular form

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

## Instructions

1. Clone this repository
2. Place the housing dataset CSV file in the same directory as the notebook
3. Open `3_Housing_EDA_ML.ipynb` in Jupyter or VS Code
4. Run all cells sequentially

## Output

- Cleaned and transformed dataset
- Visualizations and correlation insights
- Trained regression models with performance metrics

