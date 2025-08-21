# 📝 NLP with NLTK — Reference Notes

## 🔍 Introduction

**Natural Language Processing (NLP)** is the field of AI concerned with teaching machines to understand, interpret, and generate human language.
**NLTK (Natural Language Toolkit)** is a widely used Python library that provides:

* Access to corpora and lexical resources (e.g., WordNet).
* Tools for text processing: tokenization, stemming, lemmatization, POS tagging, parsing.
* Functions for classification, clustering, and information retrieval.

This notebook demonstrates **fundamental NLP preprocessing steps** with NLTK:

* Tokenization
* Stopwords removal
* Stemming

These steps form the **foundation for any downstream NLP task** such as text classification, sentiment analysis, or topic modeling.

---

## 1️⃣ Tokenization

### 🔹 What is it?

Tokenization is splitting raw text into **smaller units (tokens)** such as words or sentences.

* Input: `"Chess is a board game for two players."`
* Word tokens: `["Chess", "is", "a", "board", "game", "for", "two", "players", "."]`

### 🔹 Why is it important?

* Converts unstructured text → structured form.
* Enables statistical/ML models to process language.
* Basis for further steps (stopword removal, stemming, vectorization).

### 🔹 How in NLTK?

```python
from nltk.tokenize import word_tokenize

text = "Chess is a board game for two players."
tokens = word_tokenize(text)
print(tokens)
```

### 🔹 Applications

* Building a **bag-of-words model**.
* Input preparation for ML pipelines (classification, topic modeling).
* Sentence tokenization for **summarization** or **translation**.

### 🔹 Tips

* Always **lowercase** tokens for uniformity.
* Tokenization rules differ across languages (use language-specific tokenizers).
* For production, spaCy or Hugging Face tokenizers may handle edge cases better.

---

## 2️⃣ Stopwords Removal

### 🔹 What is it?

Stopwords are **common words** that usually carry little meaning in text analysis.
Examples: *the, is, and, of, in, on, with*.

Removing them reduces dataset size and emphasizes meaningful words.

### 🔹 Why is it important?

* Eliminates noise.
* Focuses analysis on content words.
* Improves efficiency and accuracy in text classification.

### 🔹 How in NLTK?

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
print(filtered_tokens)
```

### 🔹 Applications

* Information retrieval (e.g., search engines).
* Sentiment analysis (focus on adjectives, verbs).
* Keyword extraction.

### 🔹 Tips

* Stopword lists are **language-specific**.
* Customize stopword lists (domain-specific terms may be meaningful).
* Sometimes retaining stopwords helps (e.g., authorship analysis, negation handling like "not good").

---

## 3️⃣ Stemming

### 🔹 What is it?

Stemming reduces words to their **root/base form** by chopping off suffixes.

* Example:

  * *players → player*
  * *strategy → strategi*

### 🔹 Why is it important?

* Reduces vocabulary size.
* Groups similar words → improves generalization in models.
* Essential for search engines, indexing, clustering.

### 🔹 How in NLTK?

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
print(stemmed_tokens)
```

### 🔹 Applications

* Search engines (retrieving documents regardless of tense/plural).
* Topic modeling and clustering.
* Preprocessing before vectorization (TF-IDF, embeddings).

### 🔹 Tips

* Stemming can produce non-words ("strategy → strategi").
* Use **lemmatization** (with POS tags) for more accurate base forms.
* PorterStemmer is fast but aggressive; SnowballStemmer provides better balance.

---

## 📌 Summary Table

| Step                  | Purpose                         | NLTK Function       | Applications                       | Tips                                      |
| --------------------- | ------------------------------- | ------------------- | ---------------------------------- | ----------------------------------------- |
| **Tokenization**      | Break text into tokens          | `word_tokenize`     | Classification, Summarization      | Lowercase, handle language-specific rules |
| **Stopwords Removal** | Remove frequent low-value words | `stopwords.words()` | Sentiment analysis, IR             | Customize list, sometimes keep negations  |
| **Stemming**          | Reduce words to root            | `PorterStemmer`     | Search, clustering, topic modeling | Use lemmatization for accuracy            |

---

## 📖 References

* [NLTK Documentation](https://www.nltk.org/)
* [NLTK Book](https://www.nltk.org/book/)

