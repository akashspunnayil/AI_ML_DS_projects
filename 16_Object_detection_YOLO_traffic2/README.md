# YOLO Object Detection on Traffic Video

This notebook demonstrates object detection on traffic footage using YOLOv8, including detection of cars, bikes, and pedestrians.

## Objective

To detect and label traffic participants in video frames using the YOLOv8 object detection model.

## Workflow

### 1. Load Video
- Read traffic footage using OpenCV

### 2. Load Model
- Use Ultralytics YOLOv8 for detection

### 3. Frame-by-Frame Inference
- Detect objects in each frame
- Annotate results with bounding boxes

### 4. Output
- Save the annotated video
- Display selected frames

## Dependencies

- `ultralytics`
- `opencv-python`
- `matplotlib`

## Instructions

1. Place traffic video file in working directory
2. Run `16_Object_detection_YOLO_traffic2.ipynb`
3. Output video will be saved with detections

## Output

- Real-time object-labeled traffic video

