# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

This notebook demonstrates CNN-based classification of CIFAR-10 dataset images using TensorFlow/Keras.

## Objective

To classify 32x32 RGB images from 10 object categories (e.g., airplane, automobile, bird) using convolutional neural networks.

## Workflow

### 1. Data Preparation
- Load CIFAR-10 dataset from Keras
- Normalize RGB pixel values
- One-hot encode class labels

### 2. CNN Architecture
- Stacked Conv2D and MaxPooling layers
- BatchNormalization and Dropout for generalization
- Dense layers for final classification
- Softmax output for 10 classes

### 3. Training and Evaluation
- Model compilation with categorical crossentropy
- Training with validation split
- Evaluation on test set
- Accuracy/loss curves plotted

### 4. Visualization
- Plot random test predictions
- Display confusion matrix

## Dependencies

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `seaborn`

## Instructions

1. Clone this repository
2. Open `9_CIFAR_CNN.ipynb`
3. Run all cells to train and evaluate the CNN

## Output

- Trained CNN model
- Evaluation metrics
- Confusion matrix and prediction plots

