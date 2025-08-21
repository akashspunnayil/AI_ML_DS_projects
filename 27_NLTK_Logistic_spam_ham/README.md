# 📝 Spam vs Ham Classification with Logistic Regression (NLTK + Scikit-learn)

## 🔍 Introduction

This notebook demonstrates a **binary text classification task**: classifying messages as **Spam (unwanted/advertising) or Ham (legitimate/normal)**.

Workflow:

1. **Text Preprocessing** (cleaning, tokenization, stopwords removal).
2. **Vectorisation** (TF-IDF).
3. **Model Training** (Logistic Regression).
4. **Evaluation** (accuracy and predictions).

---

## 1️⃣ Text Preprocessing

### 🔹 What is it?

Raw text must be cleaned and normalized before feeding into ML models. Steps include:

* Removing special characters, numbers, punctuation (Regex).
* Tokenizing text into words.
* Removing stopwords.
* Lowercasing.

### 🔹 Example Preprocessing Function

```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # keep letters only
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)
```

### 🔹 Applications

* Improves **signal-to-noise ratio** for ML.
* Removes irrelevant tokens that hurt classification accuracy.

### 🔹 Tips

* Preserve negations (e.g., *not good*).
* Lemmatization can improve model robustness.

---

## 2️⃣ TF-IDF Vectorisation

### 🔹 Why TF-IDF?

* Converts text into **numerical feature vectors**.
* Weighs rare but informative words higher.
* Reduces weight for common words.

### 🔹 Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_Text'])
y = df['Label']
```

---

## 3️⃣ Logistic Regression for Classification

### 🔹 Why Logistic Regression?

* Simple and efficient baseline for binary classification.
* Outputs probabilities → interpretable.
* Works well with high-dimensional sparse data like TF-IDF.

### 🔹 Training Code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 4️⃣ Evaluation

### 🔹 Metrics

* **Accuracy**: % of correct predictions.
* Can extend to Precision, Recall, F1-score, Confusion Matrix.

### 🔹 Example Code

```python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 📌 Applications in Real World

* **Email spam filters**.
* **SMS spam detection**.
* **Content moderation in forums/social media**.
* **Customer feedback categorization**.

---

## 💡 Tips & Best Practices

* Use **stratified splits** to balance spam/ham ratio.
* Try **n-grams (bi/tri-grams)** in TF-IDF for better context.
* Regularize Logistic Regression (`C` parameter) to prevent overfitting.
* For large-scale spam detection, use advanced models (Naive Bayes, SVM, or Transformers).

---

## 📖 References

* [NLTK Documentation](https://www.nltk.org/)
* [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* Research: *A Comparison of Classifiers for Spam Detection*

---


