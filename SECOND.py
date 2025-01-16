import os
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import re
import ktrain
from ktrain import text
import shutil
import docx

# Establecer la variable de entorno para usar Keras en modo legado
os.environ['TF_USE_LEGACY_KERAS'] = 'True'

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

# Function to show restaurant information
def show_restaurant_info():
    st.image("labellavita2.png", use_column_width=True)
    st.image("aboutus3.png", use_column_width=True)
    st.image("aaa.png", use_column_width=True)

# Function to handle chatbot interactions
def show_chatbot():
    # Replace the title with an image
    st.image("va.png", use_column_width=True)

    # Add chatbot description with emojis
    st.markdown("""
    Our assistant is here to help you with:
    - **Menu questions** ️
    - **Popular dishes** ️
    - **Allergies** ️
    - **Vegan options & Daily specials**

    Just type your question, and our chatbot will assist you!
    """)

    # Initialize conversation history in session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = [
            {"role": "system", "content": "You are a multilingual restaurant assistant."}
        ]

    user_query = st.text_input("How can I assist you today?")

    if user_query:
        # Add user query to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_query})

        # Try generating assistant response, handle rate limit error
        try:
            assistant_response = generate_response(st.session_state.conversation_history)
            # Add assistant response to conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": assistant_response})

            # Display user and assistant messages
            user_message = st.session_state.conversation_history[-2]
            assistant_message = st.session_state.conversation_history[-1]

            st.chat_message("user").markdown(f"**You:** {user_message['content']}", unsafe_allow_html=True)
            st.chat_message("assistant").markdown(f"**Assistant:** {assistant_message['content']}", unsafe_allow_html=True)
        except Exception as e:
            # Handle rate limit error specifically
            if "429" in str(e):
                st.warning("The assistant is currently busy. Please try again in 60 seconds.", icon="⏳")
            else:
                st.error("An error occurred while processing your request. Please try again later.", icon="⚠️")

# Function to generate chatbot responses
def generate_response(conversation_history):
    # Generate response using Azure OpenAI
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
                    "index_name": "try1ragnlp",  # Ensure this index exists in Azure
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

    # Remove document references like [doc1], [doc2], etc.
    response_content = re.sub(r'\[doc\d+\]', '', response_content)

    return response_content

# Function to display the menu image
def show_random_menu_image():
    st.image("themenu.png", use_column_width=True)

# Function to index documents from a folder
def index_documents(docs_folder):
    index_dir = '/tmp/myindex'
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

    text.SimpleQA.initialize_index(index_dir)

    # Index documents from folder
    text.SimpleQA.index_from_folder(
        docs_folder,
        index_dir,
        commit_every=1,
        use_text_extraction=True
    )

# Function to handle the QA model choice
def ask_question_with_model(model_choice, question):
    if model_choice == "bert":
        # Use BERT for answering
        qa = text.SimpleQA('/tmp/myindex')
        answers = qa.ask(question)
        full_answer = answers[0]['full_answer']
        return full_answer
    elif model_choice == "azure":
        # Use Azure for answering
        conversation_history = [{"role": "user", "content": question}]
        return generate_response(conversation_history)

# Main function to display content and chatbot
def main():
    show_restaurant_info()

    # Add space before the button
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # Show menu button
    if st.button("See Menu", key="random_menu_button", use_container_width=True):
        show_random_menu_image()

    # Show location image below the button
    st.image("loc.png", use_column_width=True)

    # Ask user to choose model
    model_choice = st.selectbox("Choose model for answering:", ["bert", "azure"])

    # Get user's question
    question = st.text_input("Ask a question:")

    if question:
        answer = ask_question_with_model(model_choice, question)
        st.write("Answer:", answer)

    # Show chatbot
    show_chatbot()

if __name__ == "__main__":
    main()
