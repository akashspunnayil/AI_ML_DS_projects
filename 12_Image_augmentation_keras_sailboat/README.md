# Image Augmentation using Keras - Sailboat Dataset

This notebook applies a variety of image augmentation techniques to a sailboat image using Keras’ `ImageDataGenerator`.

## Objective

To demonstrate how real-time data augmentation can diversify limited datasets and improve generalization in deep learning models.

## Workflow

### 1. Image Loading
- Load a sample sailboat image
- Convert to array and reshape for augmentation

### 2. Augmentation Configuration
- Setup transformations with `ImageDataGenerator`:
  - Rotation
  - Zoom
  - Flip
  - Shear
  - Brightness adjustment

### 3. Visualization
- Generate multiple augmented versions of the image
- Plot augmented samples using `matplotlib`

## Dependencies

- `tensorflow.keras`
- `numpy`
- `matplotlib`

## Instructions

1. Clone the repository
2. Place the image under `dataset/sailboat/`
3. Run the notebook `12_Image_augmentation_keras_sailboat.ipynb`

## Output

- Grid of augmented sailboat images
- Visual inspection of augmentation range

