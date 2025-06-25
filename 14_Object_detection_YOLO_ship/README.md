# Ship Object Detection using YOLO

This notebook demonstrates object detection on ship images using the YOLO (You Only Look Once) architecture from the Ultralytics `yolo` module.

## Objective

To detect ships in images using a pre-trained YOLO model and visualize the detected bounding boxes.

## Workflow

### 1. Setup
- Load YOLOv8 model from Ultralytics
- Configure image/video input path

### 2. Inference
- Perform object detection using `model.predict()`
- Annotate images with bounding boxes and labels

### 3. Visualization
- Plot original vs detected images
- Save outputs if needed

## Dependencies

- `ultralytics`
- `opencv-python`
- `matplotlib`
- `PIL`

## Instructions

1. Install YOLOv8: `pip install ultralytics`
2. Run `14_Object_detection_YOLO_ship.ipynb`
3. Ensure ship image/video is available in the path provided

## Output

- Annotated images with ship detections
- Visual confirmation of YOLO’s performance on test input

