import streamlit as st
import random
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from streamlit_chat import message
from gtts import gTTS
import io
from PyPDF2 import PdfReader
import docx2txt
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = os.getenv('AI71_API_KEY')

# Initialize the models
chat = ChatOpenAI(
    model="tiiuae/falcon-180B-chat",
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
    streaming=True,
)

# Use SentenceTransformers for embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_document(file):
    content = ""
    file_extension = file.name.split('.')[-1].lower()

    if file_extension == 'txt':
        content = file.getvalue().decode('utf-8')
    elif file_extension == 'pdf':
        try:
            pdf_reader = PdfReader(io.BytesIO(file.getvalue()))
            for page in pdf_reader.pages:
                content += page.extract_text()
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    elif file_extension == 'docx':
        content = docx2txt.process(io.BytesIO(file.getvalue()))
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

    if not content.strip():
        st.warning("The uploaded file appears to be empty or unreadable. Please check the file and try again.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)
    
    if not chunks:
        st.warning("Unable to extract meaningful content from the file. Please try a different file.")
        return None

    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    return vectorstore, content

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_mind_palace(topic, learning_style, user_preferences, content=None):
    system_message = f"""
    You are an expert in creating memorable and personalized mind palaces to aid in learning and retention. 
    The user wants to learn about '{topic}' and their preferred learning style is '{learning_style}'.
    Their personal preferences are: {user_preferences}
    Create a vivid and easy-to-remember mind palace description that incorporates the topic, caters to the user's learning style, and aligns with their preferences.
    The mind palace should have 5-7 interconnected rooms or areas, each representing a key aspect of the topic.
    For each room, provide:
    1. A catchy and memorable name related to the topic
    2. A vivid description that incorporates the user's preferences and makes use of multiple senses
    3. 3-5 key elements or objects in the room that represent important information
    4. How these elements relate to the topic
    5. A simple and effective memory technique or association specific to the user's learning style
    
    Ensure that the mind palace is coherent, with a logical flow between rooms. Use vivid imagery, familiar concepts, and emotional connections to make it more memorable.
    
    Format your response as a JSON object with the following structure:
    {{
        "palace_name": "Catchy Name of the Mind Palace",
        "rooms": [
            {{
                "name": "Memorable Room Name",
                "description": "Vivid description of the room",
                "elements": [
                    {{
                        "name": "Striking Element Name",
                        "description": "How this element relates to the topic",
                        "memory_technique": "A simple and effective memory technique or association"
                    }}
                ]
            }}
        ]
    }}
    
    Ensure that your response is a valid JSON object. Do not include any text before or after the JSON object.
    """
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"Create a memorable mind palace for the topic: {topic}")
    ]
    
    if content:
        messages.append(HumanMessage(content=f"Use this additional context to enhance the mind palace, focusing on the most important and memorable aspects: {content[:2000]}"))
    
    try:
        response = chat.invoke(messages)
        json_response = json.loads(response.content)
        return json_response
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response: {str(e)}")
        st.error("Raw response content:")
        st.error(response.content)
        raise

def generate_audio_description(mind_palace_data):
    description = f"Welcome to your personalized and memorable mind palace: {mind_palace_data['palace_name']}. Let's take a journey through your palace, using vivid imagery and your preferred learning style to make it unforgettable. "
    for room in mind_palace_data['rooms']:
        description += f"We're entering the {room['name']}. {room['description']} "
        for element in room['elements']:
            description += f"Focus on the {element['name']}. {element['description']} To remember this, use this simple technique: {element['memory_technique']} Take a moment to really visualize and feel this connection. "
        description += "Now, let's move to the next room, carrying these vivid images with us. "
    description += "We've completed our tour of your mind palace. Take a deep breath and recall the journey we've just taken, visualizing each room and its striking elements."

    tts = gTTS(text=description, lang='en', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    return fp

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. Memorable Mind Palace Generator", layout="wide")
    
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input, .stTextArea textarea {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .room-expander {
        background-color: #2E2E2E;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stSelectbox>div>div>select {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("S.H.E.R.L.O.C.K.")
        st.subheader("Memorable Mind Palace Generator")
        
        st.markdown("---")
        st.markdown("How to use your Memorable Mind Palace:")
        st.markdown("""
        1. Choose to enter a topic or upload a document.
        2. Select your learning style and enter your preferences.
        3. Generate your personalized, easy-to-remember mind palace.
        4. Listen to the vivid audio description and imagine each room.
        5. Explore the detailed text description, focusing on the striking elements.
        6. Use the chat to reinforce your understanding of the mind palace.
        7. Practice recalling information by mentally walking through your palace, using the memory techniques provided.
        """)
        
        st.markdown("---")
        st.markdown("Powered by Falcon-180B and SentenceTransformers")

    # Main content
    st.title("S.H.E.R.L.O.C.K. Memorable Mind Palace Generator")
    
    st.write("""
    Welcome to the Memorable Mind Palace Generator! This tool will help you create a vivid and easy-to-remember
    mind palace to enhance your learning and memory retention. Choose to enter a topic or upload a document,
    select your preferred learning style, and enter your personal preferences. We'll generate a unique,
    unforgettable mind palace tailored just for you.
    """)
    
    input_method = st.radio("Choose your input method:", ["Enter a topic", "Upload a document"])
    
    if input_method == "Enter a topic":
        topic = st.text_input("Enter the topic you want to learn:")
        uploaded_file = None
    else:
        topic = None
        uploaded_file = st.file_uploader("Upload a document to memorize", type=['txt', 'md', 'pdf', 'docx'])
    
    learning_style = st.selectbox("Choose your preferred learning style:", 
                                  ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"])
    
    st.write("""
    Learning Styles:
    - Visual: You learn best through images, diagrams, and spatial understanding. We'll create vivid mental pictures.
    - Auditory: You prefer learning through listening and speaking. We'll focus on memorable sounds and verbal associations.
    - Kinesthetic: You learn by doing, moving, and touching. We'll incorporate imaginary physical sensations and movements.
    - Reading/Writing: You learn best through words. We'll use powerful written descriptions and word associations.
    """)
    
    user_preferences = st.text_area("Enter your personal preferences (e.g., favorite places, hobbies, movies, or anything that resonates with you):")
    
    if st.button("Generate Memorable Mind Palace"):
        with st.spinner("Crafting your unforgettable mind palace..."):
            content = None
            if uploaded_file is not None:
                vectorstore, content = process_document(uploaded_file)
                if vectorstore is None:
                    st.error("Failed to process the uploaded document. Please try again with a different file.")
                    return
                topic = "Document Content"
            elif topic is None or topic.strip() == "":
                st.error("Please enter a topic or upload a document.")
                return
            
            try:
                mind_palace_data = generate_mind_palace(topic, learning_style, user_preferences, content)
                if mind_palace_data is None:
                    st.error("Failed to generate the mind palace. Please try again.")
                    return
                
                st.session_state.mind_palace = mind_palace_data
                st.session_state.chat_history = []
                
                # Generate audio description with selected voice
                with st.spinner("Creating a vivid audio guide for your mind palace..."):
                    audio_fp = generate_audio_description(mind_palace_data)
                    st.session_state.mind_palace_audio = audio_fp
            except Exception as e:
                logger.error(f"An error occurred while generating the mind palace: {str(e)}")
                st.error(f"An error occurred while generating the mind palace. Please try again.")
                return

    if 'mind_palace' in st.session_state:
        mind_palace_data = st.session_state.mind_palace
        
        st.subheader(f"Your Memorable Mind Palace: {mind_palace_data['palace_name']}")
        
        # Audio player
        if 'mind_palace_audio' in st.session_state:
            try:
                st.audio(st.session_state.mind_palace_audio, format='audio/wav')
                st.write("👆 Listen to the vivid audio guide and imagine your mind palace. Close your eyes and immerse yourself in this mental journey.")
            except Exception as e:
                logger.error(f"Error playing audio: {str(e)}")
                st.warning("There was an issue playing the audio. You can still explore the text description of your mind palace.")
        
        # Text description
        for room in mind_palace_data['rooms']:
            with st.expander(f"Room: {room['name']}", expanded=True):
                st.markdown(f"**Description:** {room['description']}")
                st.markdown("**Key Elements:**")
                for element in room['elements']:
                    st.markdown(f"- **{element['name']}:** {element['description']}")
                    st.markdown(f"  *Memory Technique:* {element['memory_technique']}")
        
        st.success("Your memorable mind palace has been generated successfully! Take some time to walk through it mentally, focusing on the vivid details and connections.")
        
        # Chat interface
        st.subheader("Reinforce Your Mind Palace")
        
        # Initialize input key if not present
        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0
        
        # Display chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        for i, (sender, message_text) in enumerate(st.session_state.chat_history):
            if sender == "user":
                message(message_text, is_user=True, key=f"{i}_user")
            else:
                message(message_text, key=f"{i}_assistant")
        
        # User input text box with dynamic key
        user_input = st.text_input("Ask a question or request a memory reinforcement exercise:", key=f"user_input_{st.session_state.input_key}")
        
        # Ask button below the text input
        ask_button = st.button("Ask")
        
        if ask_button and user_input:
            with st.spinner("Generating response to enhance your memory..."):
                # Prepare context for the AI
                context = f"Mind Palace Data: {json.dumps(mind_palace_data)}\n\n"
                if 'uploaded_content' in st.session_state:
                    context += f"Uploaded Document Content: {st.session_state.uploaded_content}\n\n"
                
                system_message = f"""
                You are an AI assistant helping the user understand and remember their personalized mind palace.
                Use the following context to provide responses that reinforce the vivid imagery and memory techniques used in the mind palace.
                If asked about specific content from the uploaded document, refer to it in your response.
                
                {context}
                """
                
                response = chat.invoke([
                    SystemMessage(content=system_message),
                    HumanMessage(content=user_input)
                ])
                
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response.content))
            
            # Increment the input key to reset the input field
            st.session_state.input_key += 1
            
            # Force a rerun to update the chat history display and reset the input
            st.experimental_rerun()

if __name__ == "__main__":
    main()