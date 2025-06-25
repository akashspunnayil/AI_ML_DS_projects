# Automobile Dataset - EDA and Machine Learning

This repository contains a Jupyter Notebook that performs end-to-end Exploratory Data Analysis (EDA) and builds Machine Learning models on an automobile dataset.

## Overview

The notebook walks through the complete data science pipeline from data loading and cleaning to model training and evaluation. It aims to predict car prices based on various features such as horsepower, engine size, fuel type, body style, etc.

## Key Components

### 1. Data Preprocessing
- Loading CSV dataset
- Renaming columns for clarity
- Handling missing values
- Feature encoding (LabelEncoder, OneHotEncoder)
- Scaling numerical features (StandardScaler)

### 2. Exploratory Data Analysis (EDA)
- Distribution and correlation plots
- Boxplots to detect outliers
- Heatmaps to visualize correlations

### 3. Feature Engineering
- Creating derived variables (e.g., price per horsepower)
- Dropping irrelevant or redundant features

### 4. Model Building
- Train-test split
- Model training using:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
- Model evaluation using R², MAE, RMSE

### 5. Model Comparison
- Performance comparison table across all models

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

## How to Run

1. Clone the repository.
2. Place the CSV dataset in the notebook directory.
3. Open `1_Automobiles_EDA_ML.ipynb` in Jupyter Notebook or VS Code.
4. Run all cells sequentially.

## Output

- Cleaned and transformed dataset
- Data visualizations for insights
- Trained machine learning models with performance metrics


