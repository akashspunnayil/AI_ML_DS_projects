# Face Detection and Counting using Haarcascade & YOLO

This notebook demonstrates a hybrid face detection pipeline using both Haarcascade and YOLO. It compares both approaches and counts the number of detected faces.

## Objective

To detect and count faces in input images or webcam snapshots using both Haarcascade and YOLOv8 for evaluation and comparison.

## Workflow

### 1. Input
- Load image from file or camera

### 2. Detection
- Use OpenCV Haarcascade to detect faces
- Use YOLOv8 to detect and label faces
- Count number of faces detected by both models

### 3. Output
- Annotated images for both models
- Count summaries and visual comparison

## Dependencies

- `opencv-python`
- `ultralytics`
- `matplotlib`
- `streamlit` (for optional UI)

## Instructions

1. Run the notebook step by step
2. Compare results from both detection methods
3. Inspect accuracy and count differences

## Output

- Side-by-side output of Haar vs YOLO
- Face count summary

