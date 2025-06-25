# Fashion MNIST Classification using Convolutional Neural Networks (CNN)

This notebook implements a Convolutional Neural Network (CNN) to classify fashion apparel images from the Fashion MNIST dataset using TensorFlow and Keras.

## Objective

To improve image classification performance compared to simple ANN by leveraging spatial features using convolutional layers.

## Workflow

### 1. Data Preparation
- Load Fashion MNIST dataset from Keras
- Normalize image pixel values
- Reshape input data to include channel dimension
- One-hot encode class labels

### 2. CNN Architecture
- Convolutional + MaxPooling layers
- Dropout for regularization
- Fully connected dense layers
- Output layer with 10 units and softmax

### 3. Training and Evaluation
- Compile with categorical crossentropy and Adam optimizer
- Train on training set with validation
- Evaluate on test set
- Plot accuracy and loss curves

### 4. Visualization
- Display test images with predicted labels
- Confusion matrix for classification performance

## Dependencies

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `seaborn`

## Instructions

1. Clone the repository
2. Open and run `8_FASHION_MNIST_CNN.ipynb` in Jupyter
3. Follow the training and evaluation steps

## Output

- Classification metrics and confusion matrix
- Sample prediction visualizations
- Improved accuracy vs ANN baseline

