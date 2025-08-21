# 🎬 Movie Review Sentiment Analysis (Logistic Regression, Naive Bayes, SVC)

## 🔍 Introduction

This notebook demonstrates **sentiment classification of movie reviews** into **positive** or **negative** classes using different machine learning models.

Workflow:

1. Text cleaning and preprocessing.
2. Vectorisation (TF-IDF).
3. Model training (Logistic Regression, Naive Bayes, Support Vector Classifier).
4. Model evaluation and comparison.
5. Sample predictions on unseen reviews.

---

## 1️⃣ Text Preprocessing

### 🔹 Steps

* **Regex Cleaning**: remove numbers, punctuation, and special characters.
* **Lowercasing**.
* **Tokenization**: splitting into words.
* **Stopwords Removal**.
* **Handling Missing Data**: dropping empty or null reviews.

### 🔹 Example Preprocessing Code

```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))  # keep only letters
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

df['clean_review'] = df['review'].apply(preprocess)
```

---

## 2️⃣ TF-IDF Vectorisation

### 🔹 Why TF-IDF?

* Converts cleaned text into numerical vectors.
* Emphasizes informative words.
* Reduces weight of common filler words.

### 🔹 Example Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_review'])
```

---

## 3️⃣ Models

### 🔹 Logistic Regression

* Probabilistic model, efficient for text classification.
* Handles sparse TF-IDF features well.

### 🔹 Naive Bayes (MultinomialNB)

* Based on Bayes’ theorem with independence assumption.
* Very fast and effective baseline for NLP tasks.

### 🔹 Support Vector Classifier (SVC)

* Maximizes margin between classes.
* Works well with high-dimensional data.
* Can be slower than Logistic Regression/Naive Bayes on large corpora.

### 🔹 Example Training Function

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name} Accuracy: {acc:.4f}")
    return model

lr_model = train_and_evaluate_model(LogisticRegression(), X_train, y_train, X_test, y_test, "Logistic Regression")
nb_model = train_and_evaluate_model(MultinomialNB(), X_train, y_train, X_test, y_test, "Naive Bayes")
svc_model = train_and_evaluate_model(SVC(), X_train, y_train, X_test, y_test, "SVC")
```

---

## 4️⃣ Evaluation

### 🔹 Metrics

* Accuracy comparison across models.
* Example predictions on unseen reviews:

```python
def predict_movie_review(text, model):
    text = preprocess(text)
    vectorized = vectorizer.transform([text])
    return model.predict(vectorized)[0]

print(predict_movie_review("The movie was a waste of time", lr_model))
print(predict_movie_review("Absolutely fantastic film!", nb_model))
```

---

## 📌 Applications

* **Sentiment Analysis** for movie/product reviews.
* **Opinion Mining** from social media posts.
* **Customer Feedback Classification**.
* **Reputation Monitoring** in businesses.

---

## 💡 Tips & Best Practices

* Naive Bayes is **fast** and works surprisingly well on text.
* Logistic Regression usually achieves **higher accuracy** with good preprocessing.
* SVC can perform well but may require **tuning kernel and regularization**.
* Try adding **n-grams** (bi-grams, tri-grams) to capture word order context.
* Use **cross-validation** for robust comparison.

---

## 📖 References

* [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* [Scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)

---


