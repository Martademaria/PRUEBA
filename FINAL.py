import streamlit as st
from openai import AzureOpenAI
from transformers import pipeline
import os
from dotenv import load_dotenv
from ktrain import text
import re

# Load environment variables from .env file
load_dotenv()

# Retrieve keys from environment variables
ENDPOINT = os.getenv("AZURE_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("SEARCH_KEY")
SUBSCRIPTION_KEY = os.getenv("SUBSCRIPTION_KEY")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
    api_version="2024-05-01-preview",
)

@st.cache_resource
def load_classifier():
    """Load the zero-shot classification model once."""
    return pipeline("zero-shot-classification", model="microsoft/deberta-large-mnli")

@st.cache_resource
def load_qa_model():
    """Initialize and load the SimpleQA model."""
    index_dir = '/tmp/myindex'
    if not os.path.exists(index_dir):
        os.makedirs(index_dir, exist_ok=True)
        text.SimpleQA.initialize_index(index_dir)
    return text.SimpleQA(index_dir)

def classify_query(classifier, user_text):
    """Classify the query into categories."""
    labels = ['Menu Related', 'Not Menu Related', 'Operational Related']
    contextualized_text = f"Classify this question as related to menu, operations, or other: '{user_text}'"
    result = classifier(contextualized_text, labels)
    predicted_label = result['labels'][0]
    return predicted_label

def generate_response(client, conversation_history):
    """Generate response using Azure OpenAI."""
    completion = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=conversation_history,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        extra_body={
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "endpoint": SEARCH_ENDPOINT,
                    "index_name": "try1ragnlp",
                    "semantic_configuration": "azureml-default",
                    "authentication": {
                        "type": "api_key",
                        "key": SEARCH_KEY
                    },
                    "embedding_dependency": {
                        "type": "endpoint",
                        "endpoint": f"{ENDPOINT}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-07-01-preview",
                        "authentication": {
                            "type": "api_key",
                            "key": SUBSCRIPTION_KEY
                        }
                    },
                    "query_type": "vector_simple_hybrid",
                    "top_n_documents": 5
                }
            }]
        }
    )
    response_content = completion.choices[0].message.content.strip()
    response_content = re.sub(r'\[doc\d+\]', '', response_content)
    return response_content

def ask_question(classifier, qa_model, client, question):
    """Route the query based on classification."""
    classification = classify_query(classifier, question)
    if classification == "Menu Related":
        conversation_history = [{"role": "user", "content": question}]
        return generate_response(client, conversation_history), "azure"
    else:
        answers = qa_model.ask(question)
        return answers[0]['full_answer'], "bert"

def show_chatbot(classifier, qa_model, client):
    """Chatbot UI and interaction logic."""
    st.image("va.png", use_column_width=True)
    st.markdown("""
    Our assistant is here to help you with:
    - **Menu questions** ️
    - **Popular dishes** ️
    - **Allergies** ️
    - **Vegan options & Daily specials** 
    """)

    user_query = st.text_input("How can I assist you today?")

    if user_query:
        # Generate response
        answer, source = ask_question(classifier, qa_model, client, user_query)

        # Display the current interaction
        st.chat_message("user").markdown(f"**You:** {user_query}", unsafe_allow_html=True)
        st.chat_message("assistant").markdown(f"**Assistant:** {answer}", unsafe_allow_html=True)

        # Handle feedback for BERT responses
        if source == "bert":
            feedback = st.radio("Was this answer helpful?", ('Yes', 'No'), index=0, key="bert_feedback")
            if feedback == "No":
                st.write("Regenerating answer using Azure AI...")
                conversation_history = [{"role": "user", "content": user_query}]
                new_answer = generate_response(client, conversation_history)
                st.chat_message("assistant").markdown(f"**Assistant (Azure):** {new_answer}", unsafe_allow_html=True)

def show_restaurant_info():
    """Display restaurant info."""
    st.image("labellavita2.png", use_column_width=True)
    st.image("aboutus3.png", use_column_width=True)
    st.image("aaa.png", use_column_width=True)

def main():
    """Main function."""
    client = load_azure_openai_client()
    classifier = load_classifier()
    qa_model = load_qa_model()

    show_restaurant_info()

    # Button that occupies full width
    if st.button("See Menu", key="random_menu_button", use_container_width=True):
        st.image("themenu.png", use_column_width=True)

    st.image("loc.png", use_column_width=True)
    show_chatbot(classifier, qa_model, client)

if __name__ == "__main__":
    main()
