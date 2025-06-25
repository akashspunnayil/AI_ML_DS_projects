# CIFAR-10 Image Classification using Artificial Neural Network

This notebook implements a basic Artificial Neural Network (ANN) for image classification on the CIFAR-10 dataset using Keras.

## Objective

To classify 32x32 RGB images into one of 10 object categories such as airplane, car, bird, etc., using a fully connected neural network.

## Workflow

### 1. Data Preparation
- Load CIFAR-10 dataset from Keras
- Normalize RGB pixel values to [0, 1]
- One-hot encode the class labels

### 2. ANN Model
- Flatten 32x32x3 input to 1D
- Hidden dense layers with ReLU activation
- Output layer with 10 units and softmax

### 3. Training and Evaluation
- Train on training set
- Evaluate accuracy and loss on test set
- Plot training and validation metrics

### 4. Visualization
- Plot random predictions on test samples
- Display confusion matrix

## Dependencies

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `seaborn`

## Instructions

1. Clone this repository
2. Run `7_CIFAR_ANN.ipynb` in Jupyter Notebook
3. Follow the cell sequence for training and evaluation

## Output

- Test set accuracy
- Training history visualization
- Predicted sample outputs and confusion matrix

