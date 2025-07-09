# Face Recognition with DeepFace (CUDA & Non-CUDA Versions)

This project demonstrates a **simple face recognition system** using the `DeepFace` library. It allows identifying a person in a test image by comparing its facial embedding with a set of known faces. 

The project is structured into **two versions**:

- **Colab/CUDA version**: Computes facial embeddings on-the-fly using GPU acceleration.
- **Non-CUDA version**: Uses pre-generated `.pkl` files for fast inference on CPU-only systems.

---

## 🚀 Project Overview

This project performs:
- Face embedding extraction using `DeepFace` and the `Facenet` model.
- Similarity comparison using **cosine similarity**.
- Face recognition (i.e., identity matching).
- Optional display of the test image with the predicted identity.

---

## 📁 File Structure

face_recognition/
├── project/
│ ├── known_faces/ # Folder with known face images (e.g., "person1.jpg", "person2.jpg")
│ └── mohanlal_test_1.jpeg # Test image to identify
├── known_embeddings.pkl # Pickled known face embeddings (created by Colab version)
├── test_embedding.pkl # Pickled test image embedding (created by Colab version)
├── face_recognition_deepface_cuda_version.ipynb # GPU-enabled Colab version
├── face_recognition_deepface_non_cuda_version.ipynb # CPU version using saved pkl files
└── README.md # This file




## ⚙️ Installation

Install the required packages:

```bash
pip install deepface opencv-python scikit-learn matplotlib
```

---

## 🚀 Version 1: CUDA / Colab-Based Execution

**Steps:**

1. Add known face images to:
   `/content/drive/MyDrive/face_recognition/project/known_faces/`

2. The script:

   * Extracts embeddings for all known images.
   * Extracts embedding for the test image.
   * Saves them as `known_embeddings.pkl` and `test_embedding.pkl`.
   * Compares the test embedding with all known embeddings using cosine similarity.

3. If the similarity exceeds `0.5`, it declares a match.

**Outputs:**

* Terminal output:
  `✅ Recognized: [name]`

* Image plot with predicted identity as title.

---

## ❄️ Version 2: Non-CUDA CPU-Only Execution

**Steps:**

1. Ensure `known_embeddings.pkl` and `test_embedding.pkl` are present in `./data/`.

2. The script:

   * Loads the `.pkl` files.
   * Computes cosine similarity between test embedding and known embeddings.
   * Identifies the best match if similarity > 0.5.

3. Displays test image with predicted label.

**Outputs:**

* Terminal output:
  `✅ Recognized: [name] (Similarity: 0.78)`

---

## 🧠 Why Two Versions?

| Feature           | CUDA / Colab Version          | Non-CUDA Version                |
| ----------------- | ----------------------------- | ------------------------------- |
| Execution Speed   | Fast (GPU)                    | Fast (no embedding computation) |
| DeepFace Required | Yes                           | No (only for preprocessing)     |
| Use Case          | Initial embedding generation  | Lightweight local inference     |
| Ideal For         | Google Colab / GPU-enabled PC | Any local system                |

---

## 🎯 Example Output

```
✅ Recognized: mohanlal (Similarity: 0.782)
```

And the test image is displayed with this label using matplotlib.

---

## ⚠️ Notes

* Face matching threshold (0.5) is empirical; adjust for better accuracy.
* Known images should be clear, front-facing, and consistent in lighting.
* Only identity with the highest similarity above threshold is considered a match.

---

## 📄 License

This project is released under the MIT License.

---

