# 📝 NLP Vectorisation — Reference Notes

## 🔍 Introduction

Machine learning models cannot directly process raw text; we need to convert text into **numerical representations (vectors)**. This process is called **vectorisation**.

This notebook demonstrates two classical text vectorisation techniques:

1. **Bag of Words (BoW)**
2. **TF-IDF (Term Frequency – Inverse Document Frequency)**

Both methods are widely used for **document classification, clustering, sentiment analysis, and information retrieval**.

---

## 1️⃣ Bag of Words (BoW)

### 🔹 What is it?

* Represents text as a **frequency distribution of words**.
* Ignores grammar and word order, but keeps track of word counts.

**Example corpus:**

```text
1. "NLP is fun"
2. "NLP is powerful"
```

**BoW Representation:**

| Word     | Doc1 | Doc2 |
| -------- | ---- | ---- |
| NLP      | 1    | 1    |
| is       | 1    | 1    |
| fun      | 1    | 0    |
| powerful | 0    | 1    |

### 🔹 How in Python (scikit-learn)?

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["NLP is fun", "NLP is powerful"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 🔹 Applications

* Text classification (spam detection, sentiment analysis).
* Document similarity comparisons.
* Feature extraction for clustering and topic modeling.

### 🔹 Tips

* BoW creates **sparse matrices** for large corpora.
* Doesn’t capture semantic meaning (just frequency).
* Apply preprocessing: lowercasing, stopwords removal, lemmatization.

---

## 2️⃣ TF-IDF (Term Frequency – Inverse Document Frequency)

### 🔹 What is it?

TF-IDF improves BoW by **weighting words by importance**:

* **TF (Term Frequency):** how often a word appears in a document.
* **IDF (Inverse Document Frequency):** how rare the word is across all documents.
* Words common across all docs (like *is, the, of*) get low weight.
* Rare but meaningful words get higher weight.

**Formula:**

$$
TF\text{-}IDF(t, d) = TF(t, d) \times \log\frac{N}{DF(t)}
$$

where $N$ = total number of documents, $DF(t)$ = number of documents containing term $t$.

### 🔹 How in Python (scikit-learn)?

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["NLP is fun", "NLP is powerful"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 🔹 Applications

* Document ranking in search engines (e.g., Google).
* Keyword extraction.
* Text classification.

### 🔹 Tips

* TF-IDF balances frequency with **informativeness**.
* Good for tasks where **rare terms matter**.
* Still bag-of-words based → loses word order/context.

---

## 📌 Comparison

| Aspect               | Bag of Words | TF-IDF                     |
| -------------------- | ------------ | -------------------------- |
| Captures frequency   | ✅            | ✅                          |
| Captures importance  | ❌            | ✅                          |
| Handles stopwords    | Needs manual | Weighted low automatically |
| Produces sparse data | ✅            | ✅                          |
| Semantic meaning     | ❌            | ❌                          |

---

## 📖 References

* [Scikit-learn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [Scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

