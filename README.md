# Repo for IPCS ML/DS projects 

# 🔍 Machine Learning, Computer Vision, and Deep Learning Projects

This repository contains a collection of applied notebooks and Streamlit apps focusing on **machine learning**, **image classification**, **data augmentation**, and **object detection**.

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

| App Name | Description | Live Demo |
|----------|-------------|-----------|
| `Face_Detection` | Traditional face/eye/smile detection using Haarcascade via Streamlit UI. | [View App](https://a-face-detection-app.streamlit.app/) |
| `House_Intrusion_Detection` | Webcam-based intruder alert system using YOLO and optional email alerts. | [View App](https://a-house-intrusion-detection-app.streamlit.app) |
| `Face_Mask_Detection` | YOLO-based face mask detection app with image, video, webcam input. | [View App](https://facemask-app.streamlit.app/) |



---

## 📦 Core Tools & Libraries

### 🧠 Machine Learning & Statistical Modeling
- `scikit-learn`: Core ML models, preprocessing, pipelines, evaluation
  - **Regression Models**: `LinearRegression`, `Ridge`, `Lasso`, `PolynomialFeatures`
  - **Classification Models**: `LogisticRegression`, `KNeighborsClassifier`, `SVC`
  - **Tree-based Models**: `DecisionTreeRegressor`, `DecisionTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesClassifier`
  - **Ensemble Methods**: `GradientBoostingClassifier`, `VotingClassifier`, `StackingClassifier`
  - **Metrics & Tools**: `classification_report`, `confusion_matrix`, `roc_auc_score`, `train_test_split`, `cross_val_score`
  - **Hyperparameter Tuning**: `GridSearchCV`, `RandomizedSearchCV`, `Pipeline`, `StandardScaler`, `MinMaxScaler`

- `xgboost`: Efficient boosting algorithm used for both classification and regression with hyperparameter tuning via `GridSearchCV`.

- `statsmodels` *(optional)*: Advanced statistical modeling and regression diagnostics (can be extended for inference).

---

### 🧪 Deep Learning & Image Classification
- `tensorflow`, `keras`: ANN & CNN modeling on MNIST, Fashion-MNIST, CIFAR-10
  - Dense Networks (MLPs)
  - CNNs with `Conv2D`, `MaxPooling2D`, `Dropout`, `Flatten`
  - Transfer Learning: `VGG16` from `keras.applications`
- `keras.preprocessing.image`: Image loading, flow from directory, augmentation
- `keras.utils`: Label encoding (`to_categorical`), utility conversions
- `keras.models`: Model saving, loading, serialization
- `keras.callbacks`: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

---

### 🎨 Data Visualization
- `matplotlib`: Static plots, subplots, heatmaps, animation
- `seaborn`: Statistical graphics, pair plots, violin plots, heatmaps
- `plotly`: Interactive plots (e.g., in Streamlit)
- `PIL` (Pillow): Image processing and display
- `gridspec`, `mpl_toolkits.axes_grid1`: Layout design and axis management

---

### 📸 Computer Vision & Image Processing
- `opencv-python`: Real-time video frame processing, image annotation, webcam integration
- `cv2.dnn`: Deep Neural Network module (used with Haarcascade in legacy detection)
- `albumentations` *(optional)*: High-performance image augmentation
- `imageio`, `moviepy.editor`: Creating and exporting GIFs and annotated video

---

### 🛰️ Object Detection & Image Annotation
- `ultralytics`: YOLOv8 model inference and visualization
- `torch`: Backend engine used by YOLO models (in `ultralytics`)
- `cvzone` *(optional)*: Overlay utilities for real-time annotation

---

### 📁 Data Handling & Utilities
- `numpy`: Core numerical computing
- `pandas`: DataFrame operations, grouping, filtering, transformations
- `os`, `glob`: File path handling and batch operations
- `tempfile`: Used for temporary storage of uploaded files in Streamlit
- `io`, `base64`: Encoding media (e.g., for image/video preview in Streamlit)

---

### 🌐 App Development
- `streamlit`: UI framework for building ML and CV dashboards
- `streamlit.components.v1`: Embed HTML or external elements
- `streamlit_webrtc` *(optional)*: Real-time webcam streaming in apps

---

### 📧 Notification & System Integration (App Specific)
- `smtplib`, `email.message`: For sending email alerts (intrusion detection)
- `dotenv`: Manage credentials via `.env` (email username, password)
- `time`, `datetime`: Timestamping frames, logs, and real-time triggers
- `re`: Regular expressions (e.g., parsing filenames, labels)

---


