# 🍽️ Restaurant Reviews Sentiment — Logistic Regression, Naive Bayes, SVC

## Overview

Binary sentiment classification of restaurant reviews using:

* Text preprocessing with NLTK
* TF-IDF vectorisation
* Models: Logistic Regression, Multinomial Naive Bayes, Support Vector Classifier
* Train–test split evaluation
* Quick prediction helper for single inputs

Use this as a baseline pipeline for review analytics and feedback mining.

---

## 1) Data and Preprocessing

### Goal

Convert raw reviews to clean, informative text for ML models.

### Steps

* Keep alphabetic chars via regex
* Lowercase
* Tokenize
* Remove stopwords
* Join back to a clean string

### Code

```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

df['clean_Review'] = df['Review'].apply(preprocess)
y = df['Liked']          # or your sentiment/label column
```

### Tips

* Preserve negation if sentiment is subtle. Consider keeping “not”, “no”, “never”.
* Optional: lemmatize to reduce sparsity.
* Drop rows that become empty after cleaning.

---

## 2) TF-IDF Vectorisation

### Why

Transforms cleaned text to numeric features that weight informative words higher and common words lower.

### Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),     # start with (1,1); add bigrams for phrases like "not good"
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
X = vectorizer.fit_transform(df['clean_Review'])
```

### Tips

* If data is small, keep vocabulary modest to avoid overfitting.
* Check feature count to ensure it’s not exploding.

---

## 3) Train–Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

Use `stratify=y` to preserve class ratios.

---

## 4) Models

### Logistic Regression

* Strong baseline on TF-IDF. Probabilistic outputs.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=300)
lr.fit(X_train, y_train)
```

**Tune**

* `C` ∈ {0.1, 1, 3, 10}
* `solver='liblinear'` for small/medium datasets

### Multinomial Naive Bayes

* Fast, low-memory, works well on sparse counts.

```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)
```

**Tune**

* `alpha` ∈ {0.1, 0.5, 1.0}

### Support Vector Classifier

* Linear margin maximization. Often top accuracy with TF-IDF.

```python
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
```

**Tune**

* `C` ∈ {0.1, 1, 3, 10}

---

## 5) Evaluation

### Quick metrics and report

```python
from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, X_te, y_te, name):
    y_pred = model.predict(X_te)
    print(f"{name} accuracy: {accuracy_score(y_te, y_pred):.4f}")
    print(classification_report(y_te, y_pred, digits=3))

evaluate(lr,  X_test, y_test, "Logistic Regression")
evaluate(nb,  X_test, y_test, "Naive Bayes")
evaluate(svc, X_test, y_test, "Linear SVC")
```

### What to watch

* Macro F1 for balance across classes
* Confusion matrix to see false positives vs false negatives
* If classes are imbalanced, consider class weights or resampling

---

## 6) Inference on New Reviews

```python
def predict_sentiment(text, model):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

predict_sentiment("I am not happy with the service", lr)
```

---

## 7) Applications

* Auto-tag positive/negative customer feedback
* Alerting for negative experience detection
* Prioritization of complaints in CRM workflows
* Dashboarding for restaurant or food-delivery platforms

---

## 8) Tips, Pitfalls, Extensions

**Tips**

* Use `ngram_range=(1,2)` to capture “not good”, “very tasty”.
* Sublinear TF can help LR/SVC.
* Keep a validation set or use cross-validation for robust selection.

**Pitfalls**

* Over-cleaning may remove sentiment cues like “!” or emojis that carry tone.
* Imbalance can inflate accuracy while F1 suffers.
* Data leakage if vectorizer is fit on full data before splitting.

**Extensions**

* Add lemmatization + POS-aware preprocessing.
* Probabilities: use `LogisticRegression` and calibrate if needed.
* Pipeline + GridSearchCV for end-to-end tuning.
* Compare TF-IDF with embeddings (Word2Vec, fastText, sentence transformers) for nuanced sentiment.

---

## 9) Reproducibility

```python
import numpy as np, random
random.seed(42); np.random.seed(42)
# scikit-learn splits use random_state=42 above
```

Pin versions in `requirements.txt`:

```
numpy
pandas
scikit-learn
nltk
```

---

## Quick Model Guide

| Model               | Pros                          | Cons                        | When to use                  |
| ------------------- | ----------------------------- | --------------------------- | ---------------------------- |
| Logistic Regression | Probabilities, interpretable  | Needs tuning of C           | Baseline, thresholding tasks |
| Naive Bayes         | Very fast, simple             | Independence assumption     | Small data, quick baseline   |
| Linear SVC          | Often best accuracy on TF-IDF | No probabilities by default | Max accuracy, large vocab    |

---

