import streamlit as st
from NLU.text_preprocessing import nettoyage_prompt, get_sentence_embedding
import gensim.downloader as api


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return api.load("word2vec-google-news-300")

# Load the Google News Word2Vec model (300-dimensional)
pretrained_model = load_embedding_model()


def initialize_session_state():
    """Initialize the chat history if not already present."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_title():
    """Display the chatbot title."""
    st.title("🤖 Chatbot")

def display_chat_history():
    """Render the chat history stored in session_state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_user_input():
    """Prompt the user to enter a message."""
    return st.chat_input("Type your message here...")

def generate_bot_response(user_input):
    """
    Generate the bot's response.
    Replace this with your actual chatbot logic or API call.
    """
    return f"I'm just a placeholder bot, but I heard: **{user_input}**"

def append_message(role, content):
    """Append a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content})

def handle_chat(user_input):
    """Process the user input and generate/display bot response."""
    append_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    tokenized_prompt = nettoyage_prompt(user_input)
    embeddings = get_sentence_embedding(pretrained_model, tokenized_prompt)
    #Akram start working here to get the response

    bot_response = generate_bot_response(user_input)
    append_message("assistant", bot_response)
    with st.chat_message("assistant"):
        st.markdown(bot_response)

# ---- Main App ----

def main():
    initialize_session_state()
    display_title()
    display_chat_history()

    user_input = get_user_input()
    if user_input:
        handle_chat(user_input)

if __name__ == "__main__":
    main()