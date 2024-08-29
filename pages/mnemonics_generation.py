import streamlit as st
import random
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import tempfile
from PIL import Image
import io

# Load environment variables
load_dotenv()

AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = os.getenv('AI71_API_KEY')

# Initialize the Falcon model
chat = ChatOpenAI(
    model="tiiuae/falcon-180B-chat",
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
    streaming=True,
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

def process_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(temp_file_path)
        elif file_extension in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(temp_file_path)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            continue

        documents.extend(loader.load())
        os.unlink(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def generate_mnemonic(topic, user_preferences):
    prompt = f"""
    Generate a memorable mnemonic for the topic: {topic}.
    Consider the user's preferences: {user_preferences}.
    The mnemonic should be easy to remember and relate to the topic.
    Also provide a brief explanation of how the mnemonic relates to the topic.
    """
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_quiz_question(mnemonic):
    quiz_prompt = f"""
    Create a quiz question based on the mnemonic: {mnemonic}
    Format your response as follows:
    Question: [Your question here]
    Answer: [Your answer here]
    """
    quiz_response = chat.invoke([HumanMessage(content=quiz_prompt)])
    content = quiz_response.content.strip()
    
    try:
        question_part, answer_part = content.split("Answer:", 1)
        question = question_part.replace("Question:", "").strip()
        answer = answer_part.strip()
    except ValueError:
        question = content
        answer = "Unable to generate a specific answer. Please refer to the mnemonic."
    
    return question, answer

def generate_image_prompt(mnemonic):
    prompt = f"""
    Create a detailed image prompt for Midjourney based on the mnemonic: {mnemonic}
    The image should visually represent the key elements of the mnemonic.
    """
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. Mnemonic Generator", page_icon="🧠", layout="wide")

    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 S.H.E.R.L.O.C.K. Mnemonic Generator")

    # Initialize session state
    if 'generated_mnemonic' not in st.session_state:
        st.session_state.generated_mnemonic = None
    if 'quiz_question' not in st.session_state:
        st.session_state.quiz_question = None
    if 'quiz_answer' not in st.session_state:
        st.session_state.quiz_answer = None
    if 'image_prompt' not in st.session_state:
        st.session_state.image_prompt = None

    # Sidebar
    with st.sidebar:
        st.header("📚 Document Upload")
        uploaded_files = st.file_uploader("Upload documents (optional)", type=["pdf", "txt", "md", "doc", "docx"], accept_multiple_files=True)
        if uploaded_files:
            qa_chain = process_documents(uploaded_files)
            st.success(f"{len(uploaded_files)} document(s) processed successfully!")
        else:
            qa_chain = None

        st.header("🎨 User Preferences")
        user_preferences = st.text_area("Enter your interests or preferences:")

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🔍 Generate Mnemonic")
        topic = st.text_input("Enter the topic for your mnemonic:")
        
        if st.button("Generate Mnemonic"):
            if topic:
                with st.spinner("Generating mnemonic..."):
                    mnemonic = generate_mnemonic(topic, user_preferences)
                st.session_state.generated_mnemonic = mnemonic
                
                with st.spinner("Generating quiz question..."):
                    question, answer = generate_quiz_question(mnemonic)
                st.session_state.quiz_question = question
                st.session_state.quiz_answer = answer

                with st.spinner("Generating image prompt..."):
                    image_prompt = generate_image_prompt(mnemonic)
                st.session_state.image_prompt = image_prompt
            else:
                st.warning("Please enter a topic to generate a mnemonic.")

    with col2:
        if st.session_state.generated_mnemonic:
            st.header("📝 Generated Mnemonic")
            st.write(st.session_state.generated_mnemonic)

    # Quiz section
    if st.session_state.quiz_question:
        st.header("🧠 Mnemonic Quiz")
        st.write(st.session_state.quiz_question)
        user_answer = st.text_input("Your answer:")
        if st.button("Submit Answer"):
            if user_answer.lower() == st.session_state.quiz_answer.lower():
                st.success("🎉 Correct! Well done.")
            else:
                st.error(f"❌ Not quite. The correct answer is: {st.session_state.quiz_answer}")

    # Image prompt section
    if st.session_state.image_prompt:
        st.header("🖼️ Image Prompt")
        st.write(st.session_state.image_prompt)
        st.info("You can use this prompt with Midjourney or other image generation tools to create a visual representation of your mnemonic.")

    # Document Q&A section
    if qa_chain:
        st.header("📚 Document Q&A")
        user_question = st.text_input("Ask a question about the uploaded document(s):")
        if st.button("Get Answer"):
            with st.spinner("Searching for the answer..."):
                result = qa_chain({"query": user_question})
                st.subheader("Answer:")
                st.write(result["result"])
                st.subheader("Sources:")
                for source in result["source_documents"]:
                    st.write(source.page_content)

    # Mnemonic visualization
    if st.session_state.generated_mnemonic:
        st.header("🎨 Mnemonic Visualization")
        visualization_type = st.selectbox("Choose visualization type:", ["Word Cloud", "Mind Map"])
        if st.button("Generate Visualization"):
            with st.spinner("Generating visualization..."):
                visualization_prompt = f"""
                Create a detailed description of a {visualization_type} based on the mnemonic:
                {st.session_state.generated_mnemonic}
                Describe the layout, key elements, and their relationships.
                """
                visualization_description = chat.invoke([HumanMessage(content=visualization_prompt)]).content
                st.write(visualization_description)
                st.info("You can use this description to create a visual representation of your mnemonic using tools like Canva or Mindmeister.")

    # Export options
    if st.session_state.generated_mnemonic:
        st.header("📤 Export Options")
        export_format = st.selectbox("Choose export format:", ["Text", "PDF", "Markdown"])
        if st.button("Export Mnemonic"):
            export_content = f"""
            Topic: {topic}
            
            Mnemonic:
            {st.session_state.generated_mnemonic}
            
            Quiz Question:
            {st.session_state.quiz_question}
            
            Quiz Answer:
            {st.session_state.quiz_answer}
            
            Image Prompt:
            {st.session_state.image_prompt}
            """
            
            if export_format == "Text":
                st.download_button("Download Text", export_content, file_name="mnemonic_export.txt")
            elif export_format == "PDF":
                # You'd need to implement PDF generation here, for example using reportlab
                st.warning("PDF export not implemented in this example.")
            elif export_format == "Markdown":
                st.download_button("Download Markdown", export_content, file_name="mnemonic_export.md")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Powered by Falcon-180B and Streamlit")

if __name__ == "__main__":
    main()