# 🧠 Simple RNN for Sentiment Analysis

## 🔍 Overview

This notebook demonstrates how to build a **Recurrent Neural Network (RNN)** for sentiment classification using TensorFlow/Keras.
RNNs are suited for sequential data like text because they maintain a **hidden state (memory)** that captures dependencies across time steps.

Workflow:

1. Data preprocessing (cleaning, tokenization, padding).
2. Prepare train/test splits.
3. Build a simple RNN model in Keras.
4. Train and evaluate.
5. Predict sentiment for new text.

---

## 1️⃣ Data Preprocessing

### 🔹 Steps

* Clean text with regex (remove punctuation, numbers, special chars).
* Tokenize sentences into sequences of integers.
* Pad sequences to a fixed length for batching.

### 🔹 Example Code

```python
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

corpus = [clean_text(t) for t in raw_texts]

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)
padded = pad_sequences(sequences, maxlen=100, padding='post')
```

---

## 2️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)
```

---

## 3️⃣ Building the RNN

### 🔹 Model Structure

* **Embedding Layer**: maps tokens → dense vectors.
* **SimpleRNN Layer**: processes sequences step by step, maintaining hidden state.
* **Dense Layer**: final classification output (sigmoid for binary).

### 🔹 Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    SimpleRNN(64, activation='tanh'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

---

## 4️⃣ Training

```python
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
```

* Monitor training/validation accuracy.
* Use early stopping to prevent overfitting.

---

## 5️⃣ Evaluation

```python
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
```

* Metrics: Accuracy, Precision, Recall, F1-score.
* Can extend with confusion matrix & ROC-AUC.

---

## 6️⃣ Predictions on New Text

```python
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    pad = pad_sequences(seq, maxlen=100, padding='post')
    return "Positive" if model.predict(pad)[0] > 0.5 else "Negative"

predict_sentiment("The food was excellent and service was great")
```

---

## 📌 Applications

* Customer review analysis (restaurants, e-commerce).
* Social media sentiment tracking.
* Opinion mining for surveys and feedback.

---

## 💡 Tips & Best Practices

* **Preprocessing**: Keep negations like "not" to preserve sentiment cues.
* **Tokenization**: Set `oov_token` to handle unseen words.
* **Padding**: Use consistent length across dataset.
* **RNN Limitations**: struggles with long-term dependencies → consider LSTM/GRU.
* **Regularization**: Dropout in RNN layers can reduce overfitting.

---

## 📖 References

* [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)
* [Keras SimpleRNN Documentation](https://keras.io/api/layers/recurrent_layers/simple_rnn/)
* Jurafsky & Martin — *Speech and Language Processing*

