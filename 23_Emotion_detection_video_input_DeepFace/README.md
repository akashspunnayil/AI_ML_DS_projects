# 🎥 Emotion Detection in Video using DeepFace (Colab / CUDA Version Only)

This project performs **frame-wise facial emotion recognition** in videos using the DeepFace library. It focuses on two iconic South Indian actors — **Jagathi Sreekumar** and **Ajith Kumar** — known for their expressive performances, analyzing how well DeepFace detects and annotates their emotional transitions in selected film scenes.

---

## 🎯 Project Goals

- Detect facial emotions **frame-by-frame** in real video clips.
- Overlay dominant emotions on the face region using bounding boxes.
- Apply **temporal smoothing** to avoid erratic emotion shifts.
- Export annotated videos with both **visual overlays and original audio**.

---

## 📂 File Structure

```
video_emotion_detection/
├── video_emotion_detection_deepface_Jagathi.ipynb    # Notebook for Jagathi video
├── video_emotion_detection_deepface_Ajith.ipynb      # Notebook for Ajith video
├── annotated_with_audio.mp4                          # Final annotated video for Jagathi
├── annotated_with_audio_2.mp4                        # Final annotated video for Ajith
└── README.md                                         # This file
```

---

## 🎞️ Source Videos

The input videos are:

* **Jagathi**: A scene from *Udayananu Tharam* (Malayalam), showcasing a range of emotional expressions.
* **Ajith**: A scene from *Vedalam* (Tamil), where the actor demonstrates rapid emotional transformation.

> ⚠️ These input videos are not included in the repo due to copyright. Only processed outputs are provided.

---

## 🧠 Methodology

1. **Read video frames** using OpenCV.

2. **Detect faces and analyze emotions** for each frame using:

   ```python
   analysis = DeepFace.analyze(img_path=frame, actions=["emotion"], enforce_detection=False)[0]
   ```

3. **Apply optional confidence filtering**:

   * If the confidence of dominant emotion is below 60%, it's marked as `"Uncertain"`.

4. **Temporal smoothing** using a rolling window (`deque`) over the last 5 frames:

   * Reduces flickering between emotions.
   * Emotion with majority count is overlaid.

5. **Annotate video** with bounding boxes and emotion labels.

6. **Merge original audio** with the annotated video using `ffmpeg`.

---

## 🛠️ Requirements

Run the following in Colab:

```bash
pip install deepface opencv-python
apt-get -y install ffmpeg
```

---

## 🧪 Execution Notebooks

### 1️⃣ Jagathi Emotion Detection

* **File**: `video_emotion_detection_deepface_Jagathi.ipynb`
* **Output**: `annotated_with_audio.mp4`
* **Features**:

  * Real-time emotion annotation of a highly expressive scene.
  * Good for testing DeepFace’s sensitivity to theatrical expression.

### 2️⃣ Ajith Emotion Detection

* **File**: `video_emotion_detection_deepface_Ajith.ipynb`
* **Output**: `annotated_with_audio_2.mp4`
* **Features**:

  * Captures abrupt and exaggerated shifts in emotion.
  * Highlights DeepFace’s robustness in dynamic, cinematic scenarios.

---

## 💡 Sample Output Annotation (Illustrative)

```
[Frame 203]
Bounding Box: (x=145, y=80, w=180, h=180)
Smoothed Emotion: angry
Confidence: 94.3%
```

→ Rendered on frame as a green box with "angry" above it.

---

## 🔊 Audio Merge with FFmpeg

After video annotation, the original audio is restored:

```bash
ffmpeg -y -i annotated_video.mp4 -i original_video.mp4 \
-map 0:v:0 -map 1:a:0 -c:v libx264 -c:a aac -strict experimental \
-pix_fmt yuv420p -movflags +faststart -shortest output_with_audio.mp4
```

---

## 📌 Notes

* The `enforce_detection=False` option allows processing frames even when a face isn't confidently detected (fallback handling).
* Emotion confidence threshold is adjustable. Default is 60%.
* This implementation is optimized for **demonstration**, not real-time deployment.

---

## 📄 License

This project is released under the MIT License. Input videos belong to their respective copyright holders.

---

