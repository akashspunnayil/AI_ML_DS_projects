# 📧 RNN for Spam–Ham Classification

## 🔍 Overview

This notebook builds a **Recurrent Neural Network (RNN)** to classify text messages as **Spam** or **Ham (not spam)**.

Key steps:

1. Data cleaning and preprocessing
2. Tokenization and padding
3. RNN model building (Keras/TensorFlow)
4. Training and evaluation
5. Predictions on new messages

---

## 1️⃣ Data Preprocessing

### 🔹 What & Why

Spam/ham datasets often contain short, noisy messages (SMS or emails). Preprocessing ensures models learn meaningful patterns.

### 🔹 Steps

* Regex cleanup (remove non-alphabetic characters).
* Lowercasing.
* Tokenization into words → integer sequences.
* Padding to fixed sequence length.

### 🔹 Example Code

```python
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    return text.lower().strip()

corpus = [clean_text(t) for t in df['message']]

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
padded = pad_sequences(sequences, maxlen=100, padding='post')
```

---

## 2️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42, stratify=labels
)
```

---

## 3️⃣ Building the RNN

### 🔹 Model Architecture

* **Embedding Layer** → turns words into dense vectors.
* **SimpleRNN Layer** → sequential processing with hidden state.
* **Dense Output Layer** → binary classification (sigmoid).

### 🔹 Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    SimpleRNN(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

---

## 4️⃣ Training & Evaluation

```python
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
```

* Monitor both train and validation accuracy.
* Use early stopping if accuracy plateaus.

---

## 5️⃣ Predictions on New Messages

```python
def predict_message(text):
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    pad = pad_sequences(seq, maxlen=100, padding='post')
    return "Spam" if model.predict(pad)[0] > 0.5 else "Ham"

predict_message("You have won $1000 cash prize!!!")
predict_message("Let's meet at 5 pm for coffee")
```

---

## 📌 Applications

* **Email spam filters**
* **SMS spam detection**
* **Content moderation**

---

## 💡 Tips & Best Practices

* Keep negations (*not happy*) intact for context.
* Adjust `maxlen` in `pad_sequences` depending on dataset.
* RNNs are prone to vanishing gradients → for better performance, try **LSTM or GRU**.
* Regularize with dropout and early stopping.
* For production, prefer embeddings like **Word2Vec, GloVe, or BERT** for better semantics.

---

## ❗ Clarification

⚠️ **Spam–Ham ≠ Sentiment Analysis**

* Spam–Ham → detect unwanted vs valid messages (objective task).
* Sentiment → detect positive/negative opinion (subjective task).

---

## 📖 References

* [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)
* [Keras SimpleRNN Docs](https://keras.io/api/layers/recurrent_layers/simple_rnn/)
* Research: *SMS Spam Collection Dataset*

---

