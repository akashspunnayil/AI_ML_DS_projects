#!/usr/bin/env python
# coding: utf-8

# # RNN - Restaurant Review dataset

# In[6]:


# pip install tensorflow pandas scikit-learn pytz


# In[7]:


import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


# In[8]:


# df = pd.DataFrame(data)
df = pd.read_csv("./data/Restaurant_Reviews.tsv", sep='\t')
df


# In[9]:


# # -----------------------------
# # 0) Data: use existing df or sample
# # -----------------------------
# try:
#     assert 'df' in globals()
#     assert {'text','label'}.issubset(df.columns)
#     print("Using provided DataFrame `df`.")
# except:
#     print("No `df` found — using a small sample dataset.")
#     df = pd.DataFrame({
#         "text": [
#             "I love this movie",
#             "Worst film ever",
#             "Not bad could be better",
#             "Absolutely fantastic",
#             "Terrible acting and boring",
#             "I really enjoyed it",
#         ],
#         "label": [1, 0, 1, 1, 0, 1]
#     })


# In[10]:


df.columns


# In[11]:


# -----------------------------
# 1) Clean text (simple)
# -----------------------------
def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z\s']", " ", s) # keep letters/apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["Review"] = df["Review"].astype(str).apply(clean_text)
df["Liked"] = df["Liked"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df["Review"], df["Liked"], test_size=0.33, random_state=42, stratify=df["Liked"]
)


# In[12]:


# -----------------------------
# 2) Tokenize + PAD (many-to-one requires fixed length for batching)
# -----------------------------
MAX_VOCAB = 10000 # keep top words
MAX_LEN = 20 # pad / truncate to this length

tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tok.fit_on_texts(X_train)

def to_padded(texts):
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post", value=0)

Xtr = to_padded(X_train)
Xte = to_padded(X_test)
ytr = y_train.values
yte = y_test.values

print("\nSample sequences BEFORE padding:", tok.texts_to_sequences(X_train[:2]))
print("Sample sequences AFTER padding:\n", Xtr[:2])
print("Each row length:", Xtr.shape[1], "(= MAX_LEN)")


# In[13]:


# -----------------------------
# 3) Simple RNN (many-to-one)
# Embedding: converts token IDs to vectors
# SimpleRNN: returns FINAL hidden state (sequence summary)
# -----------------------------
EMBED_DIM = 32
RNN_UNITS = 32

model = Sequential([
    Embedding(
        input_dim=min(MAX_VOCAB, len(tok.word_index) + 1),
        output_dim=EMBED_DIM,
        input_length=MAX_LEN,
        mask_zero=True # <-- tells RNN to ignore padding (zeros)
    ),
    SimpleRNN(RNN_UNITS), # final hidden state only -> many-to-one
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("\nModel summary:")
model.summary()


# In[14]:


get_ipython().system('nvcc --version')
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


# In[15]:


# -----------------------------
# 4) Train
# -----------------------------
history = model.fit(
    Xtr, ytr,
    epochs=8,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# 5) Evaluate
# -----------------------------
loss, acc = model.evaluate(Xte, yte, verbose=0)
print(f"\nTest accuracy: {acc:.3f}")


# In[16]:


# -----------------------------
# 6) Predict on new sentences
# -----------------------------
def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_text(t) for t in texts]
    pad = to_padded(cleaned)
    probs = model.predict(pad, verbose=0).ravel()
    labels = (probs >= 0.5).astype(int)
    return list(zip(texts, probs, ["positive" if i==1 else "negative" for i in labels]))

examples = [
    "I am not happy with the service",
    "Rude customer service",
    "Food is tasty",
    "Ambient atmosphere",
    "Low price",
    "Highly unhygenic",
    "Food was cold when served"
]
for txt, p, lab in predict_sentiment(examples):
    print(f"{lab:9s} | {p:.3f} | {txt}")


# In[ ]:





# In[ ]:





# In[ ]:




