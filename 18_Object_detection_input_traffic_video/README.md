# YOLOv8 Object Detection on Traffic Video Input

This notebook performs object detection using a pre-recorded traffic video as input, using the YOLOv8 model.

## Objective

To detect and annotate traffic entities (vehicles, people, etc.) in a video using YOLO.

## Workflow

### 1. Video Input
- Load video using OpenCV

### 2. YOLO Inference
- Frame-by-frame prediction with YOLOv8
- Annotate detections

### 3. Save Output
- Save annotated video using OpenCV VideoWriter

## Dependencies

- `ultralytics`
- `opencv-python`
- `matplotlib`

## Instructions

1. Provide video file as input
2. Run `18_Object_detection_input_traffic_video.ipynb`
3. Annotated video will be saved with YOLO predictions

## Output

- Video file with object detections rendered

