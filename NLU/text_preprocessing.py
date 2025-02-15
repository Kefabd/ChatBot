import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from langdetect import detect, detect_langs
import yaml
import math
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
 
english_stop_words = list(set(stopwords.words('english')))

# french_stop_words = list(set(stopwords.words('french')))
# global_stop_words = english_stop_words + french_stop_words

# Load the YAML file
with open("conversations.yml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)  # Parse YAML file safely

# Extract conversations
global conversations 
conversations = data["conversations"]

print(conversations)


def to_lowercase(prompt):
    return prompt.lower()

def delete_stopwords(prompt):
    return " ".join([word for word in prompt.split() if word not in english_stop_words])

def text_cleaning(prompt):
    # List of characters to ignore
    ignore_character = list(string.punctuation)
    
    # Create a regex pattern for characters to ignore
    pattern = f"[{re.escape(''.join(ignore_character))}]"

    # Replace matching characters with an empty string
    cleaned_prompt = re.sub(pattern, ' ', prompt)

    # Regular expression to match single-letter words
    cleaned_prompt = re.sub(r'\b[a-z]\b', '', cleaned_prompt)
    
    # Remove extra spaces caused by removing single-letter words
    cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt)

    print(ignore_character)  # For debugging purposes
    return cleaned_prompt

def tokenization(sentence):
    language = "french"
    language_detected = detect(sentence)
    if language_detected == "en":
        language = "english"

    return nltk.word_tokenize(sentence, language=language)

def lemmatization(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def Tf(token, corpus, num_doc):
    current_doc = corpus[num_doc]
    return current_doc.count(token) / len(current_doc.split())

def IDF(terme, corpus, num_doc):
    count = 0
    for doc in corpus:
        if terme in doc:
            count += 1
    
    return math.log(len(corpus) / count)

def nottoyage_corpus(corpus):
    conversations = [text_cleaning(delete_stopwords(to_lowercase(doc))) for doc in corpus]
    return conversations

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(conversations[0])
print(X.shape)

# Example usage
sentence = "I'm happy to be here today as an data engineering student and I will try to do my best to be the first one ! on the next day of this week?"

print("LowerCase Sentence: ", to_lowercase(sentence))

print("Without Stop words: ", delete_stopwords(sentence))

print("Cleaned text: ", text_cleaning(sentence))

test = tokenization(sentence)
print("Tokenization: ", test)

print("Lemmatization: ", lemmatization(tokenization(sentence)))

