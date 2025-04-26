import streamlit as st
from transformers import AutoModel
from numpy.linalg import norm
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import numpy as np
from langchain.schema import Document
import base64
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt_tab')
import fitz 
import re
import uuid

# Define cosine similarity function
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b)) if isinstance(a, np.ndarray) else (np.array(a) @ np.array(b).T) / (norm(np.array(a)) * norm(b))

# Load Jina embeddings model
@st.cache_resource
def load_model():
    return AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

# Load ChatGroq model
@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key='Your_key',
        model_name="llama-3.3-70b-versatile"
    )

model = load_model()
llm = load_llm()

# Streamlit UI
st.title("AI-Powered PDF Q&A and MCQ Generator with RAG")

# Custom CSS to center the input at the bottom with solid background
st.markdown(
    """
    <style>
    .stTextInput {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        padding: 10px;
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.2);
        z-index: 10;
    }

    .stTextInput input {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "summary_history" not in st.session_state:
    st.session_state["summary_history"] = []
# Initialize session state for page navigation
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = 1

def render_text_with_buttons(text):
    """
    Split the text at occurrences of "(Page X)" and render it with buttons embedded.
    Handles multiple occurrences of the same page number.
    """
    pattern = r"\(Page (\d+)\)"
    segments = re.split(pattern, text)  # Split the text into parts based on the pattern

    for i, segment in enumerate(segments):
        if i % 2 == 0:
            # Render plain text
            st.write(segment)
        else:
            # Render button for the page number
            page_number = int(segment)
            button_key = f"page_{page_number}_{i}"
            if st.button(f"Go to Page {page_number}", key=button_key):
                st.session_state["selected_page"] = page_number
                st.rerun()


def options_gen(response_content):
    options_prompt = f"generate 3 different questions from this summary: {response_content}"
    options_response = llm.invoke(options_prompt)
    # Split the content into a list based on new lines or specific delimiters
    options_list = options_response.content.split("\n")
    return options_list

# Function to display chat history
def display_chat_history():
    for summary in st.session_state["summary_history"]:
            st.write(f"*Summary:* {summary}")
    for i, (question, response,mcq_content) in enumerate(st.session_state["chat_history"], start=1):
        st.write('---')
        st.write(f"Q{i}: {question}")
        render_text_with_buttons(response)
        st.write(f"A{i}: {mcq_content}")
        generated_questions = options_gen(response)
        for idx, question in enumerate(generated_questions):
            # Assign a unique key to each button using the index
            if  st.button(question, key=f"button_{idx}_{question[:10]}"): 
                
                response_content_g, mcq_content_g, retrieved_docs_g = generate_answer(question)
                render_text_with_buttons(response_content_g)
                st.write(f"mcqs:{mcq_content_g}")
                


# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    if "temp_file_path" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            st.session_state["temp_file_path"] = temp_file.name

    temp_file_path = st.session_state["temp_file_path"]
    # Split layout: Left column for PDF, right column for processing
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display PDF with page navigation
        st.subheader("PDF Preview")
        doc = fitz.open(temp_file_path)
        total_pages = doc.page_count



        # Page navigation dropdown
        selected_page = st.selectbox(
            "Select a page to view",
            [str(i + 1) for i in range(total_pages)],
            index=st.session_state["selected_page"] - 1,
            key="page_selector",
        )
        st.session_state["selected_page"] = int(selected_page)

        # Display the selected page
        page = doc.load_page(st.session_state["selected_page"] - 1)
        pix = page.get_pixmap()
        img_data = pix.tobytes()
        st.image(img_data)

    with col2:
        # Load PDF content using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks_with_pages = []
        for doc in documents:
            page_number = doc.metadata.get('page', 'Unknown')
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                text_chunks_with_pages.append(Document(page_content=chunk, metadata={"page": page_number}))

        # Initialize FAISS vector store for RAG
        embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
        vector_store = FAISS.from_documents(text_chunks_with_pages, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        st.success("PDF loaded and indexed successfully!")

        if 'summary' not in st.session_state:
            full_text = " ".join([doc.page_content for doc in text_chunks_with_pages])
            parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary_p = summarizer(parser.document, 10) 
            summary_prompt = f"summarize the following text:\n\n{summary_p}\n\nKeep it concise but cover the main points."
            summary = llm.invoke(summary_prompt)
            st.session_state.summary = summary.content
            if 'summary_history' not in st.session_state:
                st.session_state.summary_history = []
            
            st.session_state.summary_history.append(summary.content)
        else:
            summary = st.session_state.summary

        def generate_answer(question_to_use):
            # Retrieve relevant context with page numbers
            retrieved_docs = retriever.get_relevant_documents(question_to_use)
            context_with_pages = "\n\n".join(
                [f"Page {doc.metadata.get('page', 'Unknown')}: {doc.page_content}" for doc in retrieved_docs]
            )

            # Generate response
            prompt = (f"Based on the following context, provide a structured response:\n\n{context_with_pages}\n\n"
                      "Follow this format:\n"
                      "1. Overview: Provide a brief summary of the topic.\n"
                      "2. Details:\n"
                      "   - For each subtopic, provide details using bullet points.\n"
                      "   - Include page numbers in parentheses after each detail.\n"
                      )
            response = llm.invoke(prompt)

            # Generate MCQs
            mcq_prompt = f"Generate 3 multiple-choice questions from the following text:\n\n{response.content}\n\nFormat: Each question with four options, one correct answer clearly marked."
            mcq_response = llm.invoke(mcq_prompt)

            return response.content, mcq_response.content, retrieved_docs

        question = st.text_input("Enter your question", key="question_input")
        if question:
            response_content, mcq_content, retrieved_docs = generate_answer(question)

            # Add question and response to chat history
            st.session_state["chat_history"].append((question, response_content,mcq_content))



        # Display chat history
        st.subheader("Chat History")
        display_chat_history()

    # Clean up the temporary file after all processing
    if "temp_file_path" in st.session_state:
        try:
            os.remove(st.session_state["temp_file_path"])
            del st.session_state["temp_file_path"]
        except PermissionError:
            st.warning("Temporary file is still in use and cannot be removed.")