#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('jupyter nbconvert --to python ChatBot.ipynb')


# # Using Rules

# In[1]:


# -------------------- RULE BASED --------------------
def rule_based_chatbot(user_input):
    responses = {
        "hi": "Hello! How can I help you?",
        "hello": "Hi there!",
        "bye": "Goodbye! Have a great day!",
        "how are you": "I’m fine, thanks for asking!"
    }
    return responses.get(user_input.lower(), "Sorry, I don't understand that.")

# Example
print("User: hi")
print("Bot:", rule_based_chatbot("hi"))


# # Using NLP

# In[2]:


# -------------------- ML / NLP --------------------
# Dataset (toy intents)
training_data = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("bye", "goodbye"),
    ("see you", "goodbye"),
    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("how are you", "greeting")
]

X_train = [x[0] for x in training_data]
y_train = [x[1] for x in training_data]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import random

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X_train)

clf = LogisticRegression()
clf.fit(X_vec, y_train)

responses = {
    "greeting": ["Hello!", "Hey!", "Hi there!"],
    "goodbye": ["Goodbye!", "See you soon!"],
    "thanks": ["You're welcome!", "No problem!"]
}

def ml_chatbot(user_input):
    vec = vectorizer.transform([user_input])
    intent = clf.predict(vec)[0]
    return random.choice(responses[intent])

print("User: hi")
print("Bot:", ml_chatbot("how are you"))


# # Using Transformers

# ####DialoGPT model

# In[13]:


get_ipython().run_cell_magic('time', '', '\n# -------------------- TRANSFORMER --------------------\n# pip install transformers torch\nfrom transformers import pipeline\n\n# chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")\nchatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")\n# chatbot = pipeline("text-generation", model="microsoft/DialoGPT-large")\n\ndef transformer_chatbot(user_input):\n    response = chatbot(user_input, max_length=50, do_sample=True)[0]["generated_text"]\n    # return response\n    return response.split("Bot:")[-1].strip()\n\nprint(" ")\nprint(" ")\n\n# prompt = "hello!" #"How are you there?"\n\nprint("User:", "hello!")\nprint("Bot:", transformer_chatbot("hello!"))\n\nprint(" ")\nprint(" ")\n')


# #### Blenderbot

# In[22]:


from transformers import pipeline

# Load BlenderBot
chatbot = pipeline("text2text-generation", model="facebook/blenderbot-90M") #"facebook/blenderbot-400M-distill"

def transformer_chatbot(user_input):
    response = chatbot(user_input, max_length=60)[0]["generated_text"]
    return response

prompt = "hello"
print("User:", prompt)
print("Bot:", transformer_chatbot(prompt))

prompt = "who are you?"
print("User:", prompt)
print("Bot:", transformer_chatbot(prompt))


prompt = "you are a chatbot model. do you realise that?"
print("User:", prompt)
print("Bot:", transformer_chatbot(prompt))

prompt = "yes"
print("User:", prompt)
print("Bot:", transformer_chatbot(prompt))


# #### continuos chatbot

# # Using API

# #### using openai API
# 
# FYI: openai no longer give free models

# In[ ]:


get_ipython().system('python -m openai migrate')


# In[ ]:


# # -------------------- API --------------------
# # https://platform.openai.com/api-keys
# # pip install openai
# import openai
# import os
# openai.api_key = os.environ.get("OPENAI_API_KEY")

# def api_chatbot(user_input):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": user_input}]
#     )
#     return response["choices"][0]["message"]["content"]

# print("User: hello")
# print("Bot:", api_chatbot("hello"))


# In[2]:


# from openai import OpenAI
# from google.colab import userdata
# apikey = userdata.get('Secret_key')

# # client = OpenAI()
# client = OpenAI(api_key=apikey)

# def api_chatbot(user_input):
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",   # or gpt-4o-mini, etc.
#         messages=[
#             {"role": "user", "content": user_input}
#         ]
#     )
#     return response.choices[0].message.content # response["choices"][0]["message"]["content"]

# print("User: hello")
# print("Bot:", api_chatbot("hello"))


# #### using open router API
# Successful so far than openai

# In[9]:


from openai import OpenAI
from google.colab import userdata

# Get your OpenRouter API key from Colab secrets
apikey = userdata.get('openrouter_api')

# Create client with OpenRouter base URL
client = OpenAI(
    api_key=apikey,
    base_url="https://openrouter.ai/api/v1"
)

def api_chatbot(user_input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "openrouter/auto", "mistralai/mistral-7b-instruct", etc.
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content

print("User: Hello! Who are you?")
print("Bot:", api_chatbot("Hello! Who are you?"))


# In[1]:


from openai import OpenAI
from google.colab import userdata

# https://openrouter.ai/settings/keys
# https://openrouter.ai/models

# Get your OpenRouter API key from Colab secrets
apikey = userdata.get('openrouter_api')

# Create client with OpenRouter base URL
client = OpenAI(
    api_key=apikey,
    base_url="https://openrouter.ai/api/v1"
)

# Start chat history
messages = [{"role": "system", "content": "You are a helpful chatbot."}] #  roles can be 'system' and 'user' and 'assistant'

while True:
    print(" ")
    print(" ")
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye!")
        print(" ")
        print(" ")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # change if needed "deepseek/deepseek-chat-v3.1:free" https://openrouter.ai/models?fmt=table&order=pricing-low-to-high
        messages=messages
    )

    bot_reply = response.choices[0].message.content
    print("Bot:", bot_reply)
    print("(Type 'quit', 'exit', or 'bye' to end the chat)")


    messages.append({"role": "assistant", "content": bot_reply})


# In[ ]:





# # rapid api hub
