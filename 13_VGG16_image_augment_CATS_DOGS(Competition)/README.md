# Cats vs Dogs Classification using VGG16 with Image Augmentation

This notebook implements a transfer learning approach using the VGG16 model for the Cats vs Dogs classification task, including augmentation and fine-tuning.

## Objective

To classify images of cats and dogs using pre-trained deep learning (VGG16) and improve performance with image augmentation.

## Workflow

### 1. Dataset Preparation
- Load image data from directories
- Apply data augmentation using `ImageDataGenerator`
- Split data into training and validation sets

### 2. VGG16 Transfer Learning
- Load VGG16 model without top layers
- Freeze base layers
- Add custom dense layers
- Compile with categorical crossentropy

### 3. Training and Evaluation
- Fit model on augmented data
- Plot accuracy/loss curves
- Evaluate on validation set

### 4. Prediction
- Predict and visualize random samples
- Confusion matrix for model performance

## Dependencies

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `sklearn`

## Instructions

1. Clone the repository
2. Prepare `cats` and `dogs` images under appropriate directories
3. Run `13_VGG16_image_augment_CATS_DOGS(Competition).ipynb`

## Output

- Fine-tuned VGG16 model
- Validation accuracy and metrics
- Visual predictions and evaluation plots

