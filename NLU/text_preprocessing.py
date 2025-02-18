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
import gensim.downloader as api

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


"""
Embeddings Part

We have 2 choices :
1- We can train our Word2Vec model if we need a chatbot for a specific task like giving medical advice or smthg
2- We use a pretrained Word2Vec model thats already trained on massive datasets like Google News Word2Vec, GloVe, or FastText

I will opt for the 2nd choice since we aren't really developping a task specific chatbot
"""


def get_sentence_embedding(model, tokenized_sentence):
    """
    Convert a tokenized sentence into a single embedding by averaging its word vectors.

    Parameters:
      model: A pre-trained word2vec model from gensim.
      tokenized_sentence: A list of tokens (words) from a sentence.

    Returns:
      A numpy array representing the sentence embedding or None if no valid tokens are found.
    """
    valid_tokens = [token for token in tokenized_sentence if token in model]

    missing_tokens_from_model = [
        token for token in tokenized_sentence if token not in model
    ]

    print(50 * "#", " Debugging ", 50 * "#")
    print(f"Missing Tokens from Model :\n{missing_tokens_from_model}")

    if not valid_tokens:
        return None
    word_vectors = [model[token] for token in valid_tokens]
    return np.mean(word_vectors, axis=0)


# Example usage
sentence = "I'm happy to be here today as an data engineering working student and I will try to do my best to be the first one ! on the next day of this week?"

print(100 * "#")

# IMPORTANT: Wrap the sentence in a list so it's treated as a corpus with one document
cleaned_corpus = nettoyage_corpus([sentence])
print(f"Nettoyage Corpus of Sentence:\n{cleaned_corpus}")

# Load a pre-trained word embedding model.
# pretrained_model = api.load("glove-wiki-gigaword-100")

# Load the Google News Word2Vec model (300-dimensional)
# pretrained_model = api.load("word2vec-google-news-300")

# Load the GloVe Wiki Gigaword models (you can choose "50", "100", "200", or "300")
pretrained_model = api.load("glove-wiki-gigaword-100")

# Load the GloVe Twitter model (e.g., 25-dimensional)
# pretrained_model = api.load("glove-twitter-25")

# Load the FastText model (300-dimensional with subword information)
# pretrained_model = api.load("fasttext-wiki-news-subwords-300")

# Load the ConceptNet Numberbatch model (300-dimensional)
# pretrained_model = api.load("conceptnet-numberbatch-17-06-300")

# Compute the sentence embedding using the pre-trained model.
# We pass the tokenized version of our cleaned document (first element of cleaned_corpus).
sentence_tokens = cleaned_corpus[0]
sentence_embedding = get_sentence_embedding(pretrained_model, sentence_tokens)
if sentence_embedding is not None:
    print("\nSentence Embedding (averaged vector):")
    print(sentence_embedding)
else:
    print("No valid tokens found for computing sentence embedding.")
