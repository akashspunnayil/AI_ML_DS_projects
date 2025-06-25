# MNIST Handwritten Digit Classification - ANN with Hyperparameter Tuning

This notebook demonstrates image classification on the MNIST dataset using an Artificial Neural Network (ANN) built with TensorFlow/Keras, including hyperparameter tuning.

## Objective

To classify handwritten digits (0–9) using a feed-forward neural network, and evaluate performance under different hyperparameters.

## Workflow

### 1. Dataset Preparation
- Load MNIST from Keras datasets
- Normalize pixel values
- One-hot encode target labels

### 2. ANN Architecture
- Input layer for 784-pixel images
- Hidden layers (configurable neurons and activation)
- Output layer with softmax

### 3. Hyperparameter Tuning
- Number of hidden layers and units
- Batch size
- Epochs
- Optimizer selection

### 4. Model Evaluation
- Accuracy per configuration
- Confusion Matrix
- Training vs. validation loss plots

## Dependencies

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `seaborn`

## Instructions

1. Clone the repository
2. Install required libraries
3. Open and run `5_MNIST_hyperparameter_ANN.ipynb`

## Output

- Accuracy scores for various hyperparameter combinations
- Confusion matrix and prediction reports
- Training history plots

