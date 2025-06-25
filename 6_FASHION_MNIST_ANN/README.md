# Fashion MNIST Classification using Artificial Neural Network

This notebook implements an Artificial Neural Network (ANN) for classifying clothing items from the Fashion MNIST dataset using TensorFlow and Keras.

## Objective

To accurately classify grayscale images of fashion items into 10 predefined categories such as shirts, shoes, bags, etc., using a simple feedforward neural network.

## Workflow

### 1. Data Preparation
- Load Fashion MNIST dataset from Keras
- Normalize pixel values (0–255 to 0–1)
- One-hot encode the target labels

### 2. ANN Model
- Input layer: 784 neurons (flattened 28x28 images)
- Hidden layers: Dense layers with ReLU
- Output layer: 10 neurons with softmax

### 3. Training and Evaluation
- Compile with Adam optimizer and categorical crossentropy loss
- Train on training set with validation split
- Evaluate on test set
- Visualize accuracy and loss curves

### 4. Prediction
- Display random predictions on test images
- Confusion matrix visualization

## Dependencies

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `seaborn`

## Instructions

1. Clone this repository
2. Open `6_FASHION_MNIST_ANN.ipynb` in Jupyter Notebook
3. Run all cells in order

## Output

- Classification accuracy
- Training history plots
- Visualized predictions and confusion matrix

