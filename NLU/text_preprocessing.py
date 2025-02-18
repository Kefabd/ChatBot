import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from langdetect import detect
import yaml
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

english_stop_words = list(set(stopwords.words("english")))

# Load the YAML file
with open("conversations.yml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)  # Parse YAML file safely

# Extract conversations from the YAML data
conversations = data["conversations"]
print(conversations)


def to_lowercase(prompt):
    return prompt.lower()


def delete_stopwords(prompt):
    return " ".join([word for word in prompt.split() if word not in english_stop_words])


def text_cleaning(prompt):
    # List of punctuation characters to ignore
    ignore_character = list(string.punctuation)

    # Create a regex pattern for characters to ignore
    pattern = f"[{re.escape(''.join(ignore_character))}]"

    # Replace matching characters with a space
    cleaned_prompt = re.sub(pattern, " ", prompt)

    # Remove single-letter words
    cleaned_prompt = re.sub(r"\b[a-z]\b", "", cleaned_prompt)

    # Remove extra spaces
    cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt)

    return cleaned_prompt.strip()


def tokenization(sentence):
    # Default language is French
    language = "french"
    language_detected = detect(sentence)
    if language_detected == "en":
        language = "english"
    return nltk.word_tokenize(sentence, language=language)


def lemmatization(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def nettoyage_corpus(corpus):
    """
    Expects corpus to be an iterable (e.g., list) of documents.
    """
    cleaned_conversations = [
        lemmatization(tokenization(text_cleaning(delete_stopwords(to_lowercase(doc)))))
        for doc in corpus
    ]
    return cleaned_conversations


# Example usage
sentence = "I'm happy to be here today as an data engineering working student and I will try to do my best to be the first one ! on the next day of this week?"

print(100 * "#")

# IMPORTANT: Wrap the sentence in a list so it's treated as a corpus with one document
cleaned_corpus = nettoyage_corpus([sentence])
print(f"Nettoyage Corpus of Sentence:\n{cleaned_corpus}")

