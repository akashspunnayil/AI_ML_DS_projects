# Image Augmentation using Keras ImageDataGenerator - Horse Dataset

This notebook demonstrates image augmentation techniques on a horse image dataset using Keras' `ImageDataGenerator`.

## Objective

To apply real-time data augmentation techniques to enhance dataset diversity and reduce overfitting when training deep learning models.

## Workflow

### 1. Setup
- Import necessary libraries (TensorFlow, Matplotlib)
- Load a sample horse image from directory

### 2. ImageDataGenerator Usage
- Define transformations:
  - Rotation
  - Zoom
  - Shear
  - Horizontal/vertical flip
  - Brightness range
- Generate augmented image batches
- Visualize augmented samples

### 3. Visualization
- Plot grid of transformed images
- Demonstrate variation from a single input

## Dependencies

- `tensorflow`
- `matplotlib`
- `numpy`
- `PIL` or similar image loaders

## Instructions

1. Clone the repository
2. Place a horse image under `dataset/horse/` folder
3. Run `10_Image_augmentation_imagedatagen_horse.ipynb` to view augmentations

## Output

- Visualized augmented versions of a single horse image
- Demonstration of the range of transformation effects

