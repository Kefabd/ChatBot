# NLG.py

import os
import re
import string
import yaml
import numpy as np
import nltk
import gensim.downloader as api
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression


# -------------------- Setup and Downloads --------------------
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load pre-trained GloVe model for retrieval-based generation (100-dimensional)
print("Loading GloVe embeddings...")
pretrained_model = api.load("glove-wiki-gigaword-100")
embed_dim = pretrained_model.vector_size

# -------------------- Text Pre-Processing Functions --------------------
english_stop_words = list(set(stopwords.words("english")))


def to_lowercase(prompt):
    return prompt.lower()


def delete_stopwords(prompt):
    return " ".join([word for word in prompt.split() if word not in english_stop_words])


def text_cleaning(prompt):
    pattern = f"[{re.escape(''.join(string.punctuation))}]"
    cleaned_prompt = re.sub(pattern, " ", prompt)
    cleaned_prompt = re.sub(r"\b[a-z]\b", "", cleaned_prompt)
    cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt)
    return cleaned_prompt.strip()


def tokenization(sentence):
    return word_tokenize(sentence)


def lemmatization(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def nettoyage_corpus(corpus):
    """
    Process a list of documents: lowercasing, stopword deletion,
    cleaning, tokenization, and lemmatization.
    """
    cleaned_conversations = [
        lemmatization(tokenization(text_cleaning(delete_stopwords(to_lowercase(doc)))))
        for doc in corpus
    ]
    return cleaned_conversations


# -------------------- Sentence Embedding --------------------
def get_sentence_embedding(model, sentence):
    tokens = word_tokenize(sentence.lower())
    valid_tokens = [token for token in tokens if token in model]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    embeddings = [model[token] for token in valid_tokens]
    return np.mean(embeddings, axis=0)


# -------------------- Retrieval-Based Response Generation --------------------
# Candidate responses for intents
candidate_responses = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! How's it going?",
    ],
    "goodbye": ["Goodbye! Have a great day.", "Bye! Take care.", "See you later!"],
    "get_time": [
        "The current time is 3:45 PM.",
        "It's 3:45 in the afternoon right now.",
        "Right now, it's 3:45 PM.",
    ],
    "get_weather": [
        "It's sunny and 25°C outside.",
        "Currently, it's sunny with a temperature of 25°C.",
        "The weather is clear and warm at 25°C.",
    ],
    "thanks": ["You're welcome!", "No problem, happy to help!", "Anytime!"],
    "apology": ["No worries, it's okay.", "Apology accepted.", "Don't worry about it."],
    "unknown": [
        "I'm not sure I understand. Could you please clarify?",
        "Sorry, I didn't catch that. Can you rephrase?",
        "I don't understand. Can you explain a bit more?",
    ],
}


# Calculate cosine similarity between vectors
def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


# Prepare a small intent dataset for ML-based classification
intent_phrases = {
    "greeting": [
        "Hello",
        "Hi",
        "Hey there",
        "Good morning",
        "Good afternoon",
        "Good evening",
        "What's up",
        "Greetings",
        "Howdy",
        "Hi, how are you?",
        "Hey",
        "Hello there",
        "Hey, what's going on?",
        "Yo",
        "Hiya",
        "Hello, nice to see you!",
        "Hey buddy",
        "Good to see you",
        "Hi, hope you're well",
        "Hello, how do you do?",
    ],
    "goodbye": [
        "Goodbye",
        "Bye",
        "See you later",
        "Talk to you soon",
        "Farewell",
        "Take care",
        "Catch you later",
        "See ya",
        "Bye bye",
        "Adios",
        "Later",
        "So long",
        "Good night",
        "I'm off",
        "Peace out",
        "Ciao",
        "Until next time",
        "Farewell for now",
        "See you around",
        "Later alligator",
    ],
    "get_time": [
        "What time is it?",
        "Tell me the current time",
        "Could you give me the time?",
        "I need to know the time",
        "Time please",
        "Do you know what time it is?",
        "Can you tell me the time?",
        "What's the time now?",
        "Please share the time",
        "Current time?",
        "Time update",
        "What's the clock saying?",
        "Show me the time",
        "Time check",
        "What's the time, please?",
        "May I know the time?",
        "Could you update me with the time?",
        "Time now?",
        "Let me know the time",
        "Time?",
    ],
    "get_weather": [
        "What's the weather like today?",
        "Tell me the weather forecast",
        "How is the weather?",
        "Is it going to rain?",
        "Weather update please",
        "What's the temperature outside?",
        "Do I need an umbrella today?",
        "Weather report",
        "Current weather conditions?",
        "How's the weather outside?",
        "Forecast for today?",
        "Is it sunny or rainy?",
        "Weather status",
        "What's the climate like today?",
        "Do I need a jacket today?",
        "How's the weather looking?",
        "Any rain expected today?",
        "Weather check",
        "Let me know today's weather",
        "Weather update",
    ],
    "thanks": [
        "Thank you",
        "Thanks a lot",
        "Much appreciated",
        "Thanks",
        "Thank you very much",
        "I appreciate it",
        "Thanks a million",
        "Thank you so much",
        "Cheers",
        "Thanks a bunch",
        "Many thanks",
        "I'm grateful",
        "Thank you kindly",
        "I owe you one",
        "Appreciate it",
        "Thanks for everything",
        "Thanks, that was helpful",
        "Thank you, really appreciate it",
        "Thanks a ton",
        "Sincere thanks",
    ],
    "apology": [
        "I'm sorry",
        "My apologies",
        "Sorry for that",
        "I apologize",
        "Please forgive me",
        "Sorry about that",
        "My bad",
        "I didn't mean that",
        "I am really sorry",
        "Apologies",
        "I regret that",
        "So sorry",
        "Excuse me",
        "Pardon me",
        "I beg your pardon",
        "I sincerely apologize",
        "Forgive me, please",
        "I apologize for any inconvenience",
        "I'm truly sorry",
        "Sorry, my mistake",
    ],
    "unknown": [
        "I don't know",
        "Can you repeat that?",
        "What do you mean?",
        "I don't understand",
        "Could you say that again?",
        "Not sure what you mean",
        "I'm confused",
        "What?",
        "Huh?",
        "I have no idea",
        "Could you clarify?",
        "I didn't catch that",
        "Sorry, what did you say?",
        "I am not sure I follow",
        "Please explain",
        "I don't follow",
        "Could you rephrase that?",
        "I don't comprehend",
        "Unclear to me",
        "Not sure",
    ],
}

# Prepare training data for intent classification
texts = []
labels = []
for intent, phrases in intent_phrases.items():
    for phrase in phrases:
        texts.append(phrase)
        labels.append(intent)

# Create sentence embeddings for each training phrase
X = np.array([get_sentence_embedding(pretrained_model, text) for text in texts])
y = np.array(labels)

# Train the classifier (using Logistic Regression)
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)


def ml_intent(sentence):
    embedding = get_sentence_embedding(pretrained_model, sentence).reshape(1, -1)
    return clf.predict(embedding)[0]


def select_response(user_input, predicted_intent):
    """
    Given a user input and predicted intent, choose the best candidate response by measuring
    cosine similarity between embeddings.
    """
    input_embedding = get_sentence_embedding(pretrained_model, user_input)
    responses = candidate_responses.get(
        predicted_intent, candidate_responses["unknown"]
    )
    best_response = None
    best_similarity = -1
    for response in responses:
        resp_embedding = get_sentence_embedding(pretrained_model, response)
        similarity = cosine_similarity(input_embedding, resp_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_response = response
    return best_response


def retrieval_based_response(user_input):
    """
    Generate a response using the retrieval-based approach.
    """
    cleaned_corpus = nettoyage_corpus([user_input])
    cleaned_text = " ".join(cleaned_corpus[0])
    predicted_intent = ml_intent(cleaned_text)
    response = select_response(cleaned_text, predicted_intent)
    return response


# -------------------- Seq2Seq Model --------------------
# --- Data Preparation ---
def clean_text(text, remove_stopwords=False):
    if isinstance(text, list):
        return [clean_text(t, remove_stopwords) for t in text]
    text = text.lower()
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in english_stop_words]
    return tokens


# Sample conversational dataset (input-response pairs)
pairs = [
    ("hello", "hi there"),
    ("hi", "hello, how can I help you?"),
    ("good morning", "good morning! how can I assist you today?"),
    ("hey", "hey there, what can I do for you?"),
    ("how are you", "i am fine, thanks for asking."),
    ("what's up", "not much, how about you?"),
    ("what is your name", "i am a chatbot, here to assist you."),
    ("who are you", "i am your virtual assistant, ready to help."),
    ("goodbye", "see you later, take care!"),
    ("bye", "goodbye, have a nice day!"),
    ("thanks", "you're welcome!"),
    ("thank you", "no problem, happy to help!"),
    ("i need help", "sure, what do you need help with?"),
    ("can you help me", "of course, how can i assist you?"),
    ("what time is it", "the current time is 3:45 pm."),
    ("what's the weather", "it's sunny and 25°C outside."),
    (
        "i am having a technical issue",
        "i'm sorry to hear that, can you describe the problem?",
    ),
    (
        "i want to check my order status",
        "please provide your order number so i can check.",
    ),
    (
        "i would like a refund",
        "i'm sorry for the inconvenience. please share your order number for processing.",
    ),
    ("i don't understand", "could you please rephrase that?"),
    ("can you repeat that", "sure, let me repeat that for you."),
    ("what is your purpose", "i am here to assist you with any questions or tasks."),
    (
        "tell me a joke",
        "why did the scarecrow win an award? because he was outstanding in his field!",
    ),
    (
        "what can you do",
        "i can help answer your questions, provide information, and assist with tasks.",
    ),
    (
        "how can i reset my password",
        "you can reset your password by clicking on 'forgot password' on the login page.",
    ),
    ("i am bored", "maybe try a new hobby, or i can share a fun fact with you."),
    ("tell me a fun fact", "did you know that honey never spoils?"),
    ("i need some advice", "what kind of advice are you looking for?"),
    ("what is the meaning of life", "that's a deep question! some say it's 42."),
    (
        "do you know any good restaurants",
        "i can recommend some if you tell me your location.",
    ),
    (
        "i want to book a flight",
        "sure, i can help with that. can you provide your travel dates?",
    ),
    ("can i talk to a human", "i can connect you with a human agent, please hold on."),
]

# Preprocess input and target texts
input_texts = [pair[0] for pair in pairs]
target_texts = [pair[1] for pair in pairs]

# Add special tokens for decoder
START_TOKEN = "<start>"
END_TOKEN = "<end>"
target_tokens = [[START_TOKEN] + clean_text([t])[0] + [END_TOKEN] for t in target_texts]

input_tokens = clean_text(input_texts, remove_stopwords=False)


# Build vocabulary
def build_vocab(tokenized_texts, min_freq=1):
    freq = {}
    for tokens in tokenized_texts:
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
    vocab = {token for token, count in freq.items() if count >= min_freq}
    vocab = sorted(list(vocab))
    word2idx = {
        word: idx + 2 for idx, word in enumerate(vocab)
    }  # Reserve index 0 for <pad> and 1 for <unk>
    word2idx["<pad>"] = 0
    word2idx["<unk>"] = 1
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


all_tokens = input_tokens + target_tokens
word2idx, idx2word = build_vocab(all_tokens)
vocab_size = len(word2idx)
print("Vocabulary size:", vocab_size)

# Create embedding matrix using GloVe
embedding_matrix = np.zeros((vocab_size, embed_dim))
for word, idx in word2idx.items():
    if word in pretrained_model:
        embedding_matrix[idx] = pretrained_model[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)


# Convert tokens to indices and pad sequences
def tokens_to_indices(tokens, word2idx):
    return [word2idx.get(token, word2idx["<unk>"]) for token in tokens]


def pad_sequence(seq, max_len):
    return seq + [word2idx["<pad>"]] * (max_len - len(seq))


encoder_inputs = [tokens_to_indices(tokens, word2idx) for tokens in input_tokens]
decoder_inputs = [tokens_to_indices(tokens, word2idx) for tokens in target_tokens]

encoder_max_len = max(len(seq) for seq in encoder_inputs)
decoder_max_len = max(len(seq) for seq in decoder_inputs)

encoder_inputs = [pad_sequence(seq, encoder_max_len) for seq in encoder_inputs]
decoder_inputs = [pad_sequence(seq, decoder_max_len) for seq in decoder_inputs]

encoder_inputs = torch.tensor(encoder_inputs, dtype=torch.long)
decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.long)


# -------------------- Seq2Seq Model Definition --------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, embedding_matrix):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, embedding_matrix):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token, hidden, cell):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


hidden_size = 256
num_epochs = 300
learning_rate = 0.001
batch_size = encoder_inputs.size(0)  # using the entire dataset in one batch

encoder = Encoder(vocab_size, embed_dim, hidden_size, embedding_matrix)
decoder = Decoder(vocab_size, embed_dim, hidden_size, embedding_matrix)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
)

# -------------------- Model Training / Loading --------------------
encoder_weights_file = "encoder_weights.pth"
decoder_weights_file = "decoder_weights.pth"


def train_seq2seq():
    encoder.train()
    decoder.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        hidden, cell = encoder(encoder_inputs)
        decoder_input = decoder_inputs[:, 0]  # initial input tokens (start tokens)
        loss = 0
        for t in range(1, decoder_max_len):
            output, hidden, cell = decoder(decoder_input, hidden, cell)
            loss += criterion(output, decoder_inputs[:, t])
            decoder_input = decoder_inputs[:, t]  # teacher forcing

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item() / (decoder_max_len - 1):.4f}"
            )
    # Save model weights after training
    torch.save(encoder.state_dict(), encoder_weights_file)
    torch.save(decoder.state_dict(), decoder_weights_file)
    print("Seq2Seq model trained and weights saved.")


if os.path.exists(encoder_weights_file) and os.path.exists(decoder_weights_file):
    print("Loading saved Seq2Seq model weights...")
    encoder.load_state_dict(torch.load(encoder_weights_file))
    decoder.load_state_dict(torch.load(decoder_weights_file))
else:
    print("Saved weights not found. Training Seq2Seq model...")
    train_seq2seq()


def generate_response(input_sentence, max_len=20):
    """
    Generate a response using the Seq2Seq model.
    """
    encoder.eval()
    decoder.eval()
    tokens = clean_text([input_sentence])[0]
    indices = tokens_to_indices(tokens, word2idx)
    indices = pad_sequence(indices, encoder_max_len)
    input_tensor = torch.tensor([indices], dtype=torch.long)

    with torch.no_grad():
        hidden, cell = encoder(input_tensor)
        decoder_input = torch.tensor([word2idx[START_TOKEN]], dtype=torch.long)
        output_sentence = []
        for _ in range(max_len):
            output, hidden, cell = decoder(decoder_input, hidden, cell)
            predicted_idx = output.argmax(1).item()
            if predicted_idx == word2idx.get(END_TOKEN, None):
                break
            output_sentence.append(idx2word.get(predicted_idx, "<unk>"))
            decoder_input = torch.tensor([predicted_idx], dtype=torch.long)
    return " ".join(output_sentence)


def seq2seq_response(user_input):
    """
    Wrapper function for Seq2Seq model response generation.
    """
    return generate_response(user_input)


# -------------------- Additional Utilities --------------------
class ContextManager:
    def __init__(self):
        self.user_context = {}

    def update_context(self, user_id, intent):
        self.user_context[user_id] = intent

    def get_context(self, user_id):
        return self.user_context.get(user_id, None)


# -------------------- Wrapper Functions for Integration --------------------
def retrieval_based_response_wrapper(user_input):
    """
    Wrapper function to generate response using retrieval-based method.
    """
    return retrieval_based_response(user_input)


def seq2seq_response_wrapper(user_input):
    """
    Wrapper function to generate response using Seq2Seq model.
    """
    return seq2seq_response(user_input)


# -------------------- Example Testing (Optional) --------------------
if __name__ == "__main__":
    context_manager = ContextManager()
    user_id = "user123"
    test_sentences = [
        "hello",
        "What's up?",
        "good morning",
        "how are you today?",
        "what is your name?",
        "can you help me with my order?",
        "i need a refund",
        "i don't understand",
        "tell me a joke",
        "what's the weather like?",
        "i am having a technical issue",
        "can i talk to a human?",
        "goodbye",
    ]

    print("\n--- Retrieval-Based Responses ---")
    for sent in test_sentences:
        response = retrieval_based_response(sent)
        print(f"User: {sent}\nBot: {response}\n")

    print("\n--- Seq2Seq Generated Responses ---")
    for sent in test_sentences:
        response = generate_response(sent)
        print(f"User: {sent}\nBot: {response}\n")
