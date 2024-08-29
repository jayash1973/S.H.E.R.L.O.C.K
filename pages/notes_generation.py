import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
from typing import List, Dict
import json
from datetime import datetime

# Load environment variables
load_dotenv()

AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = os.getenv('AI71_API_KEY')

# Initialize the Falcon model
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="tiiuae/falcon-180B-chat",
        api_key=AI71_API_KEY,
        base_url=AI71_BASE_URL,
        streaming=True,
    )

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings()

def process_document(file_content, file_type):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
        if isinstance(file_content, str):
            tmp_file.write(file_content.encode('utf-8'))
        else:
            tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

    if file_type == 'pdf':
        loader = PyPDFLoader(tmp_file_path)
    elif file_type == 'txt':
        loader = TextLoader(tmp_file_path)
    elif file_type == 'md':
        loader = UnstructuredMarkdownLoader(tmp_file_path)
    elif file_type == 'docx':
        loader = Docx2txtLoader(tmp_file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(texts, get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    os.unlink(tmp_file_path)
    return retriever

def generate_notes(retriever, topic, style, length):
    prompt_template = f"""
    You are an expert note-taker and summarizer. Your task is to create {style} and {length} notes on the given topic.
    Use the following guidelines:
    1. Focus on key concepts and important details.
    2. Use bullet points or numbered lists for clarity.
    3. Include relevant examples or explanations where necessary.
    4. Organize the information in a logical and easy-to-follow structure.
    5. Aim for clarity without sacrificing important information.

    Context: {{context}}
    Topic: {{question}}
    
    Notes:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs
    )
    
    result = qa_chain({"query": topic})
    return result['result']

def save_notes(notes: str, topic: str):
    notes_data = load_notes_data()
    timestamp = datetime.now().isoformat()
    notes_data.append({"topic": topic, "notes": notes, "timestamp": timestamp})
    with open("saved_notes.json", "w") as f:
        json.dump(notes_data, f)

def load_notes_data() -> List[Dict]:
    try:
        with open("saved_notes.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. Notes Generator", layout="wide")

    st.title("S.H.E.R.L.O.C.K. Notes Generator")

    st.markdown("""
    This tool helps you generate concise and relevant notes on specific topics. 
    You can upload a document or enter text directly.
    """)

    # Sidebar content
    st.sidebar.title("About S.H.E.R.L.O.C.K.")
    st.sidebar.markdown("""
    S.H.E.R.L.O.C.K. (Summarizing Helper & Effective Research Liaison for Organizing Comprehensive Knowledge) 
    is an advanced AI-powered tool designed to assist you in generating comprehensive notes from various sources.

    Key Features:
    - Multi-format support (PDF, TXT, MD, DOCX)
    - Customizable note generation
    - Intelligent text processing
    - Save and retrieve notes

    How to use:
    1. Choose your input method
    2. Process your document or text
    3. Enter a topic and customize note style
    4. Generate and save your notes

    Enjoy your enhanced note-taking experience!
    """)

    input_method = st.radio("Choose input method:", ("Upload Document", "Enter Text"))

    if input_method == "Upload Document":
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md", "docx"])
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            file_content = uploaded_file.read()
            st.success("Document uploaded successfully!")
            
            with st.spinner("Processing document..."):
                retriever = process_document(file_content, file_type)
                st.session_state.retriever = retriever
            st.success("Document processed!")
    elif input_method == "Enter Text":
        text_input = st.text_area("Enter your text here:", height=200)
        if text_input:
            with st.spinner("Processing text..."):
                retriever = process_document(text_input, 'txt')
                st.session_state.retriever = retriever
            st.success("Text processed!")

    topic = st.text_input("Enter the topic for note generation:")

    col1, col2 = st.columns(2)
    with col1:
        style = st.selectbox("Note Style", ["Concise", "Detailed", "Academic", "Casual"])
    with col2:
        length = st.selectbox("Note Length", ["Short", "Medium", "Long"])

    if st.button("Generate Notes"):
        if topic and hasattr(st.session_state, 'retriever'):
            with st.spinner("Generating notes..."):
                try:
                    notes = generate_notes(st.session_state.retriever, topic, style, length)
                    st.subheader("Generated Notes:")
                    st.markdown(notes)
                    
                    # Download button for the generated notes
                    st.download_button(
                        label="Download Notes",
                        data=notes,
                        file_name=f"{topic.replace(' ', '_')}_notes.txt",
                        mime="text/plain"
                    )

                    # Save notes
                    if st.button("Save Notes"):
                        save_notes(notes, topic)
                        st.success("Notes saved successfully!")
                except Exception as e:
                    st.error(f"An error occurred while generating notes: {str(e)}")
        else:
            st.warning("Please upload a document or enter text, and specify a topic before generating notes.")

    # Display saved notes
    st.sidebar.subheader("Saved Notes")
    saved_notes = load_notes_data()
    for i, note in enumerate(saved_notes):
        if st.sidebar.button(f"{note['topic']} - {note['timestamp'][:10]}", key=f"saved_note_{i}"):
            st.subheader(f"Saved Notes: {note['topic']}")
            st.markdown(note['notes'])

    st.sidebar.markdown("---")
    st.sidebar.markdown("Powered by Falcon-180B and Streamlit")

    # Add a footer
    st.markdown("---")
    st.markdown("Created by Your Team Name | © 2024")

if __name__ == "__main__":
    main()