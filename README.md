# Repo for IPCS ML/DS projects 

# 🔍 Machine Learning, Computer Vision, and Deep Learning Projects

This repository contains a collection of applied notebooks and Streamlit apps focusing on **machine learning**, **image classification**, **data augmentation**, and **object detection** using popular libraries such as `scikit-learn`, `Keras`, `TensorFlow`, `YOLOv8`, and `OpenCV`.

Each project is structured as a Jupyter notebook or Python app, and addresses a specific real-world or academic use case.

---

## 📁 Project List & Descriptions

### 1. 🧠 Machine Learning (ML) on Tabular Data

| Notebook | Description |
|----------|-------------|
| `1_Automobiles_EDA_ML.ipynb` | End-to-end EDA and ML modeling (Linear Regression, Decision Trees, XGBoost) for automobile price prediction. |
| `2_Diabetes_EDA_ML.ipynb` | Classification modeling (SVM, KNN, Random Forest, XGBoost) on PIMA Diabetes dataset. |
| `3_Housing_EDA_ML.ipynb` | Regression models for Boston housing price prediction using scikit-learn and XGBoost. |
| `4_Insurance_EDA_ML.ipynb` | ML modeling on insurance cost prediction using both regression and feature importance analysis. |

---

### 2. 🖼️ Image Classification with ANN & CNN

| Notebook | Description |
|----------|-------------|
| `5_MNIST_hyperparameter_ANN.ipynb` | ANN-based handwritten digit recognition with hyperparameter tuning. |
| `6_FASHION_MNIST_ANN.ipynb` | ANN classification of Fashion MNIST apparel images. |
| `7_CIFAR_ANN.ipynb` | Multi-class ANN classifier on CIFAR-10 RGB image dataset. |
| `8_FASHION_MNIST_CNN.ipynb` | CNN model for Fashion MNIST with convolutional and pooling layers. |
| `9_CIFAR_CNN.ipynb` | CNN for CIFAR-10 image classification using Conv2D, Dropout, BatchNorm. |

---

### 3. 🧪 Image Augmentation Projects

| Notebook | Description |
|----------|-------------|
| `10_Image_augmentation_imagedatagen_horse.ipynb` | Data augmentation on horse images using Keras `ImageDataGenerator`. |
| `11_Image_augmentation_opencv_helicopter.ipynb` | Manual augmentation (rotation, scaling, affine) using OpenCV. |
| `12_Image_augmentation_keras_sailboat.ipynb` | Sailboat image transformations with Keras. |
| `13_VGG16_image_augment_CATS_DOGS(Competition).ipynb` | Transfer learning with VGG16 on Cats vs Dogs dataset with data augmentation. |

---

### 4. 🛰️ Object Detection with YOLOv8

| Notebook | Description |
|----------|-------------|
| `14_Object_detection_YOLO_ship.ipynb` | Ship detection using Ultralytics YOLOv8 on static images. |
| `15_Object_detection_YOLO_classroom.ipynb` | Classroom object detection (e.g., people, bags, desks) using YOLO. |
| `16_Object_detection_YOLO_traffic2.ipynb` | Object detection on traffic videos with frame annotation. |
| `17_Real-time-face-detection-YOLO.ipynb` | Live face detection using webcam and YOLOv8. |
| `18_Object_detection_input_traffic_video.ipynb` | Object detection on custom traffic video input with export. |

---

### 5. 😊 Face Detection & Streamlit Apps

| File | Description |
|------|-------------|
| `19_Face_detection_and_count_haarcascade_yolo.ipynb` | Compare face detection using Haarcascade vs YOLO and count faces. |
| `detect.py` | Webcam-based intruder alert system using YOLO and optional email alerts. |
| `facemask_streamlit_app.py` | YOLO-based face mask detection app with image, video, webcam input. |
| `face_detection_app.py` | Traditional face/eye/smile detection using Haarcascade via Streamlit UI. |

---

## 📦 Core Tools & Libraries

- **Machine Learning**: `scikit-learn`, `xgboost`
- **Deep Learning**: `tensorflow`, `keras`
- **Computer Vision**: `opencv-python`, `PIL`
- **Object Detection**: `ultralytics` (YOLOv8)
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Apps**: `streamlit`

---

