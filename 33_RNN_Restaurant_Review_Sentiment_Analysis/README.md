# RNN — Restaurant Review Sentiment Analysis

## Overview

Binary sentiment classification on restaurant reviews using a Recurrent Neural Network (TensorFlow/Keras).

Pipeline:

1. Clean and normalize raw text
2. Tokenize to integer sequences and pad to fixed length
3. Build an RNN model with an embedding layer
4. Train, validate, and evaluate
5. Predict sentiment for new reviews

Use this as a baseline deep learning approach for short review texts.

---

## 1) Data and Preprocessing

### What and why

Short user reviews are noisy. Preprocessing standardizes inputs and preserves key sentiment cues.

### Steps

* Regex cleanup (letters only)
* Lowercasing
* Tokenization to integer sequences (with OOV handling)
* Padding/truncation to consistent sequence length

### Code

```python
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    return text.lower().strip()

# corpus: your raw review texts
corpus = [clean_text(t) for t in df['review']]

MAX_VOCAB = 10000
MAX_LEN = 100  # adjust after inspecting length distribution

tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tok.fit_on_texts(corpus)

seqs = tok.texts_to_sequences(corpus)
X = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
y = df['label'].values   # 0/1 or similar
```

### Tips

* Keep negations (not, never) intact; they flip sentiment.
* Inspect sequence length distribution; set `MAX_LEN` to cover \~95% without excessive padding.
* If text is very short, bigram TF-IDF with linear models can be a strong baseline; RNNs shine as sequences grow.

---

## 2) Train–Test Split

```python
from sklearn.model_selection import train_test_split

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Use `stratify=y` to preserve class proportions.

---

## 3) Model Architecture (Simple RNN)

### Structure

* Embedding: token indices → dense vectors
* SimpleRNN: sequential modeling with hidden state
* Dense + Sigmoid: binary probability

### Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

EMB_DIM = 64

model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=EMB_DIM, input_length=MAX_LEN),
    SimpleRNN(64, activation="tanh", dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
```

### Notes

* SimpleRNN is a good didactic baseline but struggles with long-range dependencies. For better retention, prefer GRU/LSTM (see Extensions).

---

## 4) Training

```python
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(
    Xtr, ytr,
    validation_data=(Xte, yte),
    epochs=10,
    batch_size=32,
    callbacks=[es],
    verbose=1
)
```

Tips:

* Monitor both `val_loss` and `val_accuracy`.
* If overfitting, increase dropout, reduce hidden units, or add L2.

---

## 5) Evaluation

```python
loss, acc = model.evaluate(Xte, yte, verbose=0)
print(f"Test Accuracy: {acc:.4f}")
```

Extended checks:

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

probs = model.predict(Xte).ravel()
pred = (probs >= 0.5).astype(int)

print(classification_report(yte, pred, digits=3))
print(confusion_matrix(yte, pred))
```

What to watch:

* Macro F1 if classes are imbalanced.
* Confusion matrix to inspect false positives vs false negatives.
* Calibrate threshold (>0.5 may not be optimal).

---

## 6) Inference (Single Review)

```python
def predict_review(text, threshold=0.5):
    seq = tok.texts_to_sequences([clean_text(text)])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    p = float(model.predict(pad)[0])
    return ("positive", p) if p >= threshold else ("negative", p)

predict_review("Food was amazing but service was slow")
```

---

## 7) Applications

* Real-time review monitoring and alerting
* Feedback triage and escalation
* Trend analysis for menu/service changes
* Input to dashboards for operations and CRM

---

## 8) Tips, Pitfalls, and Extensions

### Tips

* Use an `oov_token` to reduce “unknown” word failures.
* Shuffle data before split; stratify to preserve label balance.
* Track token coverage: how many tokens fall outside `num_words`.

### Pitfalls

* Excessive cleaning can remove sentiment cues (exclamation, emojis).
* Too small `MAX_VOCAB` hurts coverage; too large increases overfitting and memory.
* Fixed `MAX_LEN` that is too short truncates sentiment-bearing words.

### Extensions

* Swap `SimpleRNN` with **GRU** or **LSTM**:

  ```python
  from tensorflow.keras.layers import GRU  # or LSTM
  model = Sequential([
      Embedding(MAX_VOCAB, EMB_DIM, input_length=MAX_LEN),
      GRU(64, dropout=0.2, recurrent_dropout=0.2),
      Dense(1, activation="sigmoid")
  ])
  ```
* Add **Bidirectional** wrappers for better context:

  ```python
  from tensorflow.keras.layers import Bidirectional, LSTM
  model = Sequential([
      Embedding(MAX_VOCAB, EMB_DIM, input_length=MAX_LEN),
      Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
      Dense(1, activation="sigmoid")
  ])
  ```
* Pretrained embeddings (GloVe/fastText) to improve generalization.
* Regularization: dropout, L2, early stopping, learning-rate schedules.
* Threshold tuning via ROC curve; optimize F1 or business-specific costs.

---

## 9) Reproducibility and Requirements

Set seeds:

```python
import numpy as np, random, tensorflow as tf
np.random.seed(42); random.seed(42); tf.random.set_seed(42)
```

Minimal `requirements.txt`:

```
numpy
pandas
scikit-learn
tensorflow
```

Optional:

```
matplotlib
seaborn
```

---

## Quick Model Guide

| Architecture | Pros                                    | Cons                            | When to use                          |
| ------------ | --------------------------------------- | ------------------------------- | ------------------------------------ |
| SimpleRNN    | Simple, fast on short sequences         | Weak long-term memory           | Didactic baseline, very short texts  |
| GRU          | Better memory than SimpleRNN, efficient | Slightly heavier than SimpleRNN | Strong default for short–medium text |
| LSTM         | Best memory among RNNs                  | Heavier, slower                 | Longer sequences, nuanced context    |

---

