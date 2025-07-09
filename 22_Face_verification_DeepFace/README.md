# Face Verification using DeepFace (CUDA and Non-CUDA Versions)

This project demonstrates **face verification** using the `DeepFace` library. It determines whether two given images belong to the same person by comparing their facial features.

It includes:
- A **CUDA/Colab version** that performs full verification using GPU.
- A **non-CUDA version** that loads verification results from a precomputed `.pkl` file for use on CPU-only machines.

---

## 📌 Project Highlights

- Uses `DeepFace.verify()` to compare two face images.
- Saves verification result (including distance, threshold, and match status) as a `.pkl` file.
- Visualizes both face images with verification status and distance.

---

## 🗂️ File Structure

```
face_verification/
├── verify_face_cuda_version.ipynb       # Full DeepFace verification with GPU (Colab)
├── verify_face_non_cuda.ipynb           # Lightweight CPU version using cached result
├── verify_face.pkl                      # Pickled output containing DeepFace verification result
├── verify_1.jpg                         # First face image
├── verify_2.jpg                         # Second face image
└── README.md                            # This file
```

---

## ⚙️ Installation

Install the required dependencies:

```bash
pip install deepface opencv-python matplotlib
```

---

## 🚀 Version 1: CUDA / Colab-Based Execution

This version performs face verification using DeepFace in a GPU-enabled environment (e.g., Google Colab).

### Steps:

1. Set `img1_path` and `img2_path` to two face images.
2. Run the DeepFace verification:

```python
result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path)
```

3. Save the result as a `.pkl` file:

```python
with open("verify_face.pkl", "wb") as f:
    pickle.dump(result, f)
```

4. Plot the two images side by side.
5. Show match status, cosine distance, and threshold as a title.

### Output Example:

```
✅ Match — Distance: 0.422 | Threshold: 0.55
```

---

## ❄️ Version 2: Non-CUDA / CPU-Only Execution

This version uses a `.pkl` file generated from the CUDA version for local lightweight verification display.

### Steps:

1. Place the following files in your working directory:

   * `verify_1.jpg`
   * `verify_2.jpg`
   * `verify_face.pkl`

2. Load the result:

```python
with open("verify_face.pkl", "rb") as f:
    result = pickle.load(f)
```

3. Display the images side by side using matplotlib.
4. Show whether the faces match, along with the computed distance and threshold.

---

## 🧠 Why Two Versions?

| Feature         | CUDA / Colab Version    | Non-CUDA Version           |
| --------------- | ----------------------- | -------------------------- |
| Verification    | Full DeepFace inference | Uses cached `.pkl` result  |
| GPU Dependency  | Yes                     | No                         |
| Execution Speed | Fast (with GPU)         | Instant (on CPU)           |
| Ideal For       | First-time verification | Reuse, demo, or deployment |

---

## 🎯 Example Output

```text
Match — Distance: 0.422 | Threshold: 0.55
```

And a side-by-side plot:

```
+-----------+   +-----------+
|  Image 1  |   |  Image 2  |
| (person)  |   | (person)  |
+-----------+   +-----------+
     ✅ Match
```

---

## ⚠️ Notes

* DeepFace uses cosine distance to compare face embeddings.
* A match is declared if the computed distance is **less than the model-specific threshold**.
* The `.pkl` file contains keys such as:

  * `verified` (bool)
  * `distance` (float)
  * `threshold` (float)
  * `model` (str)

---

## 📄 License

This project is released under the MIT License.

---

