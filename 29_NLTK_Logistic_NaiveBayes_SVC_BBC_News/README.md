# 📰 BBC News Text Classification — Logistic Regression, Naive Bayes, SVC

## Overview

Goal: classify BBC news articles into categories using **TF-IDF features** and three models:

* Logistic Regression
* Multinomial Naive Bayes
* Support Vector Classifier

Pipeline:

1. Clean and normalize raw text
2. Tokenize and remove stopwords
3. TF-IDF vectorisation
4. Train LR, NB, SVC
5. Evaluate with accuracy and per-class metrics
6. Quick prediction helper for new text

---

## 1) Data and Preprocessing

### What and why

Preprocessing standardizes text so models focus on signal, not noise.
Steps used in the notebook:

* Keep letters only with regex
* Lowercase
* Tokenize
* Remove stopwords
* Join back into a clean string

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

df['clean_data'] = df['data'].apply(preprocess)   # 'data' column holds article text
y = df['target']                                  # label column
```

### Tips

* Preserve negation if doing sentiment; less critical here.
* Consider lemmatization for clarity of vocabulary.
* Drop empty rows after cleaning to avoid zero-length documents.

---

## 2) Vectorisation with TF-IDF

### What and why

Converts text to numeric features weighted by importance across the corpus.
TF captures frequency in a document. IDF downweights terms common across documents.

### Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),      # try (1,1) first, add bigrams if helpful
    min_df=2,               # drop very rare tokens
    max_df=0.9,             # drop overly common tokens
    sublinear_tf=True       # log-scaling often helps LR/SVM
)
X = vectorizer.fit_transform(df['clean_data'])
```

### Tips

* Start with unigrams; add bigrams if classes are phrase-driven.
* Limit vocabulary with `min_df` and `max_df` for speed and robustness.
* Keep TF-IDF for linear models; embeddings are a different route.

---

## 3) Train-test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

Use `stratify=y` to maintain class proportions.

---

## 4) Models

### Logistic Regression

* Linear classifier with probabilistic outputs.
* Strong baseline for TF-IDF features.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=200, n_jobs=-1)  # n_jobs available in some dists; if not, omit
lr.fit(X_train, y_train)
```

**Tuning checklist**

* `C` controls regularization strength (smaller is stronger). Try `C` in \[0.1, 1, 3, 10].
* Use `solver='liblinear'` or `saga` if needed.

### Multinomial Naive Bayes

* Assumes feature independence; often competitive for text.
* Very fast, low memory.

```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)
```

**Tuning checklist**

* `alpha` smoothing in \[0.1, 0.5, 1.0].

### Support Vector Classifier

* Maximizes margin. Good with high-dimensional sparse features.
* May be slower; linear SVC is often enough.

```python
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
```

**Tuning checklist**

* `C` in \[0.1, 1, 3, 10].
* Prefer `LinearSVC` for large vocab TF-IDF. Kernel SVC is usually too slow here.

---

## 5) Evaluation

### Quick metrics

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model, X_tr, y_tr, X_te, y_te, name):
    y_pred = model.predict(X_te)
    print(f"{name} accuracy: {accuracy_score(y_te, y_pred):.4f}\n")
    print(classification_report(y_te, y_pred, digits=3))
    return confusion_matrix(y_te, y_pred)

cm_lr  = evaluate(lr,  X_train, y_train, X_test, y_test, "Logistic Regression")
cm_nb  = evaluate(nb,  X_train, y_train, X_test, y_test, "Naive Bayes")
cm_svc = evaluate(svc, X_train, y_train, X_test, y_test, "Linear SVC")
```

### What to watch

* Report macro-averaged F1 for class balance sensitivity.
* Inspect confusion matrices to see which categories are confused.
* If imbalance exists, use class weights or more data.

---

## 6) Inference on New Text

```python
def predict_news_category(text, model):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

predict_news_category("Apple unveils new iPhone with camera upgrade", lr)
```

---

## 7) Typical Results and Model Selection

* **Linear SVC** often gives the best accuracy on TF-IDF with news topics.
* **Logistic Regression** close second and provides probabilities via `predict_proba` if you use `LogisticRegression` (LinearSVC does not).
* **MultinomialNB** is fastest and strong for sparse counts, but can underperform when classes hinge on nuanced phrases.

Pick based on constraints:

* Need calibrated probabilities or thresholding → Logistic Regression
* Need top accuracy with simple features → Linear SVC
* Need speed and simplicity → MultinomialNB

---

## 8) Applications

* News topic routing and auto-tagging
* Media monitoring and trend tracking
* Content recommendation pipelines
* Archival search and semantic filters when combined with rules

---

## 9) Tips, Pitfalls, and Extensions

**Tips**

* Always stratify splits.
* Use `TfidfVectorizer(sublinear_tf=True)` for LR and SVC.
* Try `ngram_range=(1,2)` when phrases matter.

**Pitfalls**

* Leaking label info via preprocessing. Keep preprocessing independent of labels.
* Overfitting with huge vocabularies. Control with `min_df`, `max_df`, and `C`.
* Ignoring per-class metrics. High accuracy can mask poor minority class performance.

**Extensions**

* Add lemmatization to reduce sparsity.
* Try class weights in LR or SVC if classes are imbalanced.
* Use `Pipeline` with `GridSearchCV` for end-to-end tuning.
* Compare TF-IDF to averaged word embeddings or sentence transformers when semantics matter.

---

## 10) Reproducibility

```python
import numpy as np
import random
random.seed(42)
np.random.seed(42)

# scikit-learn split already uses random_state=42
```

Pin versions in `requirements.txt`:

```
numpy
pandas
scikit-learn
nltk
```

---

## References

* [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* [Scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)

---

