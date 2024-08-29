import streamlit as st
import random
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import tempfile

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

# Expanded list of predefined topics
PREDEFINED_TOPICS = [
    "Quantum Computing", "Artificial Intelligence Ethics", "Blockchain Technology",
    "Neuroscience", "Climate Change Mitigation", "Space Exploration",
    "Renewable Energy", "Genetic Engineering", "Cybersecurity",
    "Machine Learning", "Nanotechnology", "Robotics",
    "Virtual Reality", "Augmented Reality", "Internet of Things",
    "5G Technology", "Autonomous Vehicles", "Bioinformatics",
    "Cloud Computing", "Data Science", "Artificial General Intelligence",
    "Quantum Cryptography", "3D Printing", "Smart Cities",
    "Biotechnology", "Fusion Energy", "Sustainable Agriculture",
    "Space Tourism", "Quantum Sensors", "Brain-Computer Interfaces",
    "Personalized Medicine", "Synthetic Biology", "Exoplanets",
    "Dark Matter", "CRISPR Technology", "Quantum Internet",
    "Deep Learning", "Edge Computing", "Humanoid Robots",
    "Drone Technology", "Quantum Supremacy", "Neuromorphic Computing",
    "Asteroid Mining", "Bionic Implants", "Smart Materials",
    "Quantum Dots", "Lab-grown Meat", "Vertical Farming",
    "Hyperloop Transportation", "Molecular Nanotechnology", "Quantum Metrology",
    "Artificial Photosynthesis", "Cognitive Computing", "Swarm Robotics",
    "Metamaterials", "Neuroplasticity", "Quantum Machine Learning",
    "Green Hydrogen", "Organ-on-a-Chip", "Bioprinting",
    "Plasma Physics", "Quantum Simulation", "Soft Robotics",
    "Geoengineering", "Exoskeletons", "Programmable Matter",
    "Graphene Applications", "Quantum Sensing", "Neuralink",
    "Holographic Displays", "Quantum Error Correction", "Synthetic Genomes",
    "Carbon Capture and Storage", "Quantum Memory", "Organoids",
    "Artificial Synapses", "Quantum Imaging", "Biosensors",
    "Memristors", "Quantum Annealing", "DNA Data Storage",
    "Cultured Meat", "Quantum Radar", "Neuromorphic Hardware",
    "Quantum Entanglement", "Phytomining", "Biohacking",
    "Topological Quantum Computing", "Neuroprosthetics", "Optogenetics",
    "Quantum Gravity", "Molecular Machines", "Biomimicry",
    "Quantum Teleportation", "Neurogenesis", "Bioelectronics",
    "Quantum Tunneling", "Tissue Engineering", "Bioremediation",
    "Quantum Photonics", "Synthetic Neurobiology", "Nanomedicine",
    "Quantum Biology", "Biogeochemistry", "Molecular Gastronomy",
    "Quantum Thermodynamics", "Nutrigenomics", "Biomechatronics",
    "Quantum Chemistry", "Psychoneuroimmunology", "Nanophotonics",
    "Quantum Optics", "Neuroeconomics", "Bionanotechnology"
]

def process_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(temp_file_path)
    else:
        loader = TextLoader(temp_file_path)

    documents = loader.load()
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
    
    os.unlink(temp_file_path)
    return qa_chain

def get_sherlock_analysis(topic, qa_chain=None):
    system_prompt = """
    You are Sherlock Holmes, the world's greatest detective and master of observation and deduction. 
    Your task is to provide an in-depth analysis of the given topic, offering unique insights on how to approach learning it from the ground up. 
    Your analysis should:
    1. Break down the topic into its fundamental components.
    2. Identify key concepts and their relationships.
    3. Suggest a structured approach to learning, starting from first principles.
    4. Highlight potential challenges and how to overcome them.
    5. Provide a unique point of view that encourages critical thinking.
    Your response should be detailed, insightful, and encourage a deep understanding of the subject.
    """

    if qa_chain:
        result = qa_chain({"query": f"Provide a Sherlock Holmes style analysis of the topic: {topic}"})
        response = result['result']
    else:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze the following topic: {topic}")
        ]
        response = chat.invoke(messages).content
    
    return response

def chunk_text(text, max_chunk_size=4000):
    chunks = []
    current_chunk = ""
    for sentence in text.split(". "):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. Observation", page_icon="🔍", layout="wide")

    st.title("🕵️ S.H.E.R.L.O.C.K. Observation")
    st.markdown("*Uncover the depths of any subject with the keen insight of Sherlock Holmes*")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Choose Your Method")
        method = st.radio("Select input method:", ["Enter Topic", "Upload Document", "Choose from List"])

        if method == "Enter Topic":
            topic = st.text_input("Enter your topic of interest:")
        elif method == "Upload Document":
            uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
            if uploaded_file:
                topic = uploaded_file.name
        else:
            topic = st.selectbox("Choose a topic:", PREDEFINED_TOPICS)

        if st.button("Analyze", key="analyze_button"):
            if method == "Upload Document" and uploaded_file:
                qa_chain = process_document(uploaded_file)
                analysis = get_sherlock_analysis(topic, qa_chain)
            elif topic:
                analysis = get_sherlock_analysis(topic)
            else:
                st.warning("Please provide a topic or upload a document.")
                return

            col1.markdown("## Sherlock's Analysis")
            chunks = chunk_text(analysis)
            for chunk in chunks:
                col1.markdown(chunk)

    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/cd/Sherlock_Holmes_Portrait_Paget.jpg", use_column_width=True)
    st.sidebar.title("About S.H.E.R.L.O.C.K. Observation")
    st.sidebar.markdown("""
    S.H.E.R.L.O.C.K. Observation is your personal detective for any subject. 
    It provides:
    - In-depth analysis of topics
    - Unique perspectives on learning approaches
    - First principles breakdown of subjects
    - Critical thinking encouragement
    
    Let Sherlock guide you through the intricacies of any field of study!
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Powered by Falcon-180B and Streamlit")

if __name__ == "__main__":
    main()