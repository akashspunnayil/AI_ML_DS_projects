# Repo for IPCS ML/DS projects 

# 🔍 Machine Learning, Computer Vision, and Deep Learning Projects

This repository contains a collection of applied notebooks and Streamlit apps focusing on **machine learning**, **image classification**, **data augmentation**, and **object detection** using popular libraries such as `scikit-learn`, `Keras`, `TensorFlow`, `YOLOv8`, and `OpenCV`.

Each project is structured as a Jupyter notebook or Python app, and addresses a specific real-world or academic use case.

---

## 📁 Project List & Descriptions

### 1. 🧠 Machine Learning (ML) on Tabular Data

| Notebook | Description |
|----------|-------------|
| `1_Automobiles_EDA_ML` | End-to-end EDA and ML modeling (Linear Regression, Decision Trees, XGBoost) for automobile price prediction. |
| `2_Diabetes_EDA_ML` | Classification modeling (SVM, KNN, Random Forest, XGBoost) on PIMA Diabetes dataset. |
| `3_Housing_EDA_ML` | Regression models for Boston housing price prediction using scikit-learn and XGBoost. |
| `4_Insurance_EDA_ML` | ML modeling on insurance cost prediction using both regression and feature importance analysis. |

---

### 2. 🖼️ Image Classification with ANN & CNN

| Notebook | Description |
|----------|-------------|
| `5_MNIST_hyperparameter_ANN` | ANN-based handwritten digit recognition with hyperparameter tuning. |
| `6_FASHION_MNIST_ANN` | ANN classification of Fashion MNIST apparel images. |
| `7_CIFAR_ANN` | Multi-class ANN classifier on CIFAR-10 RGB image dataset. |
| `8_FASHION_MNIST_CNN` | CNN model for Fashion MNIST with convolutional and pooling layers. |
| `9_CIFAR_CNN` | CNN for CIFAR-10 image classification using Conv2D, Dropout, BatchNorm. |

---

### 3. 🧪 Image Augmentation Projects

| Notebook | Description |
|----------|-------------|
| `10_Image_augmentation_imagedatagen_horse` | Data augmentation on horse images using Keras `ImageDataGenerator`. |
| `11_Image_augmentation_opencv_helicopter` | Manual augmentation (rotation, scaling, affine) using OpenCV. |
| `12_Image_augmentation_keras_sailboat` | Sailboat image transformations with Keras. |
| `13_VGG16_image_augment_CATS_DOGS(Competition)` | Transfer learning with VGG16 on Cats vs Dogs dataset with data augmentation. |

---

### 4. 🛰️ Object Detection with YOLOv8

| Notebook | Description |
|----------|-------------|
| `14_Object_detection_YOLO_ship` | Ship detection using Ultralytics YOLOv8 on static images. |
| `15_Object_detection_YOLO_classroom` | Classroom object detection (e.g., people, bags, desks) using YOLO. |
| `16_Object_detection_YOLO_traffic2` | Object detection on traffic videos with frame annotation. |
| `17_Real-time-face-detection-YOLO` | Live face detection using webcam and YOLOv8. |
| `18_Object_detection_input_traffic_video` | Object detection on custom traffic video input with export. |
| `19_Face_detection_and_count_haarcascade_yolo` | Compare face detection using Haarcascade vs YOLO and count faces. |
---

### 5. 🖥️ Streamlit Apps

| File | Description |
|------|-------------|
| `Face_Detection` | Traditional face/eye/smile detection using Haarcascade via Streamlit UI. |see app: https://a-face-detection-app.streamlit.app/?app=face-detection |
| `House_Intrusion_Detection` | Webcam-based intruder alert system using YOLO and optional email alerts. |see app: https://a-house-intrusion-detection-app.streamlit.app |
| `Face_Mask_Detection` | YOLO-based face mask detection app with image, video, webcam input. |see app: https://facemask-app.streamlit.app/ |


---

## 📦 Core Tools & Libraries

- **Machine Learning**: `scikit-learn`, `xgboost`
- **Deep Learning**: `tensorflow`, `keras`
- **Computer Vision**: `opencv-python`, `PIL`
- **Object Detection**: `ultralytics` (YOLOv8)
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Apps**: `streamlit`

---

