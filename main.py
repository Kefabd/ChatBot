import sys
import asyncio
import streamlit as st
from NLU.text_preprocessing import nettoyage_prompt, get_sentence_embedding
from NLG.NLG import (
    retrieval_based_response,
    seq2seq_response,
)  # import the new functions
import gensim.downloader as api


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return api.load("word2vec-google-news-300")


# Load the Google News Word2Vec model (300-dimensional)
pretrained_model = load_embedding_model()


def initialize_session_state():
    """Initialize the chat history if not already present."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gen_method" not in st.session_state:
        # Default method is Retrieval Based
        st.session_state.gen_method = "Retrieval Based"


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


def append_message(role, content):
    """Append a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content})


def handle_chat(user_input):
    """Process the user input and generate/display bot response."""
    append_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process the prompt if needed (for pre-processing, for example)
    tokenized_prompt = nettoyage_prompt(user_input)
    # (You can use get_sentence_embedding if needed by passing pretrained_model)

    # Choose generation method based on selection
    if st.session_state.gen_method == "Retrieval Based":
        bot_response = retrieval_based_response(user_input)
    else:
        bot_response = seq2seq_response(user_input)

    # Include a notice on which algorithm generated the response.
    detailed_response = (
        f"**Response from {st.session_state.gen_method}:**\n\n{bot_response}"
    )

    append_message("assistant", detailed_response)
    with st.chat_message("assistant"):
        st.markdown(detailed_response)


def main():
    initialize_session_state()
    display_title()

    # Sidebar: Select Generation Method
    st.sidebar.header("Generation Settings")
    st.session_state.gen_method = st.sidebar.selectbox(
        "Select Generation Method", ["Retrieval Based", "Seq2Seq"], index=0
    )

    display_chat_history()
    user_input = get_user_input()
    if user_input:
        handle_chat(user_input)


if __name__ == "__main__":
    main()
