import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from langdetect import detect


data = pd.read_csv("SMSSpamCollection.txt", sep='\t', header=0)


X = data[["message"]]
X = X.to_numpy()
X = X.reshape(-1,1)

nltk.download('stopwords')
nltk.download('punkt_tab')
 
english_stop_words = list(set(stopwords.words('english')))
french_stop_words = list(set(stopwords.words('french')))
global_stop_words = english_stop_words + french_stop_words




def to_lowercase(prompt):
    return prompt.lower()

def delete_stopwords(prompt):
    return "".join([word for word in prompt.split() if word not in global_stop_words])

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
    language_detected = detect(sentence)
    if language_detected == "en":
        language = "english"
    else:
        language = "french"

    return nltk.word_tokenize(sentence, language=language)

def lemmatization(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]




# Example usage
sentence = "hello world! how's it going?"

cleaned = tokenization(sentence)
print("Cleaned Sentence:", cleaned)
