# 📝 NLP with spaCy — Reference Notes

## 🔍 Introduction

**spaCy** is an industrial-strength NLP library designed for efficiency and real-world usage. Unlike NLTK, which is often used for teaching/research, spaCy is optimized for **production pipelines** and includes:

* Tokenization
* Stopword removal
* Lemmatization (instead of stemming)
* POS tagging, Named Entity Recognition (NER), Dependency Parsing (notebook covers the basics)

This notebook demonstrates **fundamental NLP preprocessing** with spaCy:

* Tokenization
* Stopwords filtering
* Lemmatization

---

## 1️⃣ Tokenization

### 🔹 What is it?

Tokenization = splitting text into meaningful units (words, punctuation).
spaCy’s tokenizer handles:

* Words
* Punctuation
* Contractions (e.g., *don’t → do + n’t*)

### 🔹 How in spaCy?

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Chess is a board game for two players."
doc = nlp(text)

tokens = [token.text for token in doc]
print(tokens)
```

### 🔹 Applications

* Preparing inputs for ML models.
* Sentence segmentation in summarization.
* Word-level features in classification tasks.

### 🔹 Tips

* spaCy’s tokenizer is **rule-based + statistical** → handles edge cases better than NLTK.
* Language-specific models handle tokenization across languages.

---

## 2️⃣ Stopwords Removal

### 🔹 What is it?

Stopwords = frequent words with little semantic meaning.

### 🔹 How in spaCy?

```python
filtered_tokens = [token.text for token in doc if not token.is_stop]
print(filtered_tokens)
```

### 🔹 Applications

* Sentiment analysis
* Search/retrieval
* Topic modeling

### 🔹 Tips

* spaCy comes with built-in stopword lists for each language model.
* You can **add/remove stopwords** (`nlp.Defaults.stop_words.add("customword")`).
* Keep negations (e.g., *not*) for sentiment tasks.

---

## 3️⃣ Lemmatization

### 🔹 What is it?

Lemmatization = reducing words to their **dictionary form (lemma)** while considering context (POS tags).

* Example:

  * *players → player*
  * *was → be*
  * *better → good*

### 🔹 Why spaCy uses lemmatization over stemming

* Lemmatization produces **valid words** (vs stemming’s chopped roots).
* Context-aware → considers part-of-speech.

### 🔹 How in spaCy?

```python
lemmatized = [token.lemma_ for token in doc if not token.is_stop]
print(lemmatized)
```

### 🔹 Applications

* Improves search engines (match *run*, *ran*, *running* → *run*).
* Reduces feature space for ML models.
* Essential for chatbots, translation, IR.

### 🔹 Tips

* spaCy lemmatizer requires POS tags → automatically handled by pipeline.
* Always compare lemmas to original tokens when analyzing output.

---

## 📌 Comparison: spaCy vs NLTK

| Feature           | NLTK                      | spaCy                              |
| ----------------- | ------------------------- | ---------------------------------- |
| **Ease of Use**   | Teaching & prototyping    | Production-ready                   |
| **Tokenization**  | Rule-based                | Rule-based + statistical           |
| **Stopwords**     | `stopwords.words()`       | `token.is_stop`                    |
| **Stemming**      | Porter, Snowball stemmers | Not supported (lemmatization only) |
| **Lemmatization** | WordNetLemmatizer         | Built-in, POS-aware                |
| **NER & Parsing** | Limited support           | Built-in, fast                     |

---

## 📖 References

* [spaCy Documentation](https://spacy.io/)
* [spaCy 101: Everything You Need to Know](https://spacy.io/usage/spacy-101)

---

