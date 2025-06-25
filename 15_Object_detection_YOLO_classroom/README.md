# Object Detection in Classroom using YOLO

This notebook performs object detection on classroom scenes using the YOLOv8 model.

## Objective

To detect multiple objects in classroom images or videos (e.g., students, chairs, bags) using YOLO.

## Workflow

### 1. Setup
- Load pre-trained YOLOv8 model from Ultralytics
- Load classroom image or video

### 2. Detection
- Perform object detection using `model.predict()`
- Annotate frames with bounding boxes and class labels

### 3. Display
- Use OpenCV and Matplotlib to display results

## Dependencies

- `ultralytics`
- `opencv-python`
- `matplotlib`
- `PIL`

## Instructions

1. Install dependencies: `pip install ultralytics opencv-python`
2. Run `15_Object_detection_YOLO_classroom.ipynb`
3. Provide path to classroom image or video

## Output

- Annotated frames showing detected classroom objects

