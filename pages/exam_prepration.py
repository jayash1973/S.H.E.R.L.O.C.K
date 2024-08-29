import streamlit as st
import random
import time
from typing import List, Dict
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.graphs import NetworkxEntityGraph
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = os.getenv('AI71_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Initialize the Falcon model
chat = ChatOpenAI(
    model="tiiuae/falcon-180B-chat",
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
    streaming=True,
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

FIELDS = [
    "Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
    "History", "Geography", "Literature", "Philosophy", "Psychology",
    "Sociology", "Economics", "Business", "Finance", "Accounting",
    "Law", "Political Science", "Environmental Science", "Astronomy", "Geology",
    "Linguistics", "Anthropology", "Art History", "Music Theory", "Film Studies",
    "Medical Science", "Nursing", "Public Health", "Nutrition", "Physical Education",
    "Engineering", "Architecture", "Urban Planning", "Agriculture", "Veterinary Science",
    "Oceanography", "Meteorology", "Statistics", "Data Science", "Artificial Intelligence",
    "Cybersecurity", "Renewable Energy", "Quantum Physics", "Neuroscience", "Genetics",
    "Biotechnology", "Nanotechnology", "Robotics", "Space Exploration", "Cryptography"
]

# List of educational resources
EDUCATIONAL_RESOURCES = [
    "https://www.coursera.org",
    "https://www.khanacademy.org",
    "https://scholar.google.com",
    "https://www.edx.org",
    "https://www.udacity.com",
    "https://www.udemy.com",
    "https://www.futurelearn.com",
    "https://www.lynda.com",
    "https://www.skillshare.com",
    "https://www.codecademy.com",
    "https://www.brilliant.org",
    "https://www.duolingo.com",
    "https://www.ted.com/talks",
    "https://ocw.mit.edu",
    "https://www.open.edu/openlearn",
    "https://www.coursebuffet.com",
    "https://www.academicearth.org",
    "https://www.edutopia.org",
    "https://www.saylor.org",
    "https://www.openculture.com",
    "https://www.gutenberg.org",
    "https://www.archive.org",
    "https://www.wolframalpha.com",
    "https://www.quizlet.com",
    "https://www.mathway.com",
    "https://www.symbolab.com",
    "https://www.lessonplanet.com",
    "https://www.teacherspayteachers.com",
    "https://www.brainpop.com",
    "https://www.ck12.org"
]

def search_web(query: str, num_results: int = 30, max_retries: int = 3) -> List[Dict[str, str]]:
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'
    ]
    
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': random.choice(user_agents)}
            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
            
            results = []
            if "items" in res:
                for item in res["items"]:
                    result = {
                        "title": item["title"],
                        "link": item["link"],
                        "snippet": item.get("snippet", "")
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"An error occurred: {e}. Attempt {attempt + 1} of {max_retries}")
            time.sleep(2 ** attempt)
    
    print("Max retries reached. No results found.")
    return []

def scrape_webpage(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def process_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(uploaded_file)
        elif file_extension in ['.txt', '.md']:
            loader = TextLoader(uploaded_file)
        elif file_extension in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            continue
        
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    graph = NetworkxEntityGraph()
    graph.add_documents(texts)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain, graph

def generate_questions(topic, difficulty, num_questions, include_answers, qa_chain=None, graph=None):
    system_prompt = f"""You are an expert exam question generator. Generate {num_questions} {difficulty}-level questions about {topic}. 
    {"Each question should be followed by its correct answer." if include_answers else "Do not include answers."}
    Format your response as follows:
    Q1. [Question]
    {"A1. [Answer]" if include_answers else ""}
    Q2. [Question]
    {"A2. [Answer]" if include_answers else ""}
    ... and so on.
    """
    
    if qa_chain and graph:
        context = graph.get_relevant_documents(topic)
        context_text = "\n".join([doc.page_content for doc in context])
        
        result = qa_chain({"query": system_prompt, "context": context_text})
        questions = result['result']
    else:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please generate {num_questions} {difficulty} questions about {topic}.")
        ]
        questions = chat(messages).content
    
    return questions

def gather_resources(field: str) -> List[Dict[str, str]]:
    resources = []
    for resource_url in EDUCATIONAL_RESOURCES:
        search_results = search_web(f"site:{resource_url} {field}", num_results=1)
        if search_results:
            result = search_results[0]
            content = scrape_webpage(result['link'])
            resources.append({
                "title": result['title'],
                "link": result['link'],
                "content": content[:500] + "..." if len(content) > 500 else content
            })
    
    # YouTube search
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    youtube_results = youtube.search().list(q=field, type='video', part='id,snippet', maxResults=5).execute()
    for item in youtube_results.get('items', []):
        video_id = item['id']['videoId']
        resources.append({
            "title": item['snippet']['title'],
            "link": f"https://www.youtube.com/watch?v={video_id}",
            "content": item['snippet']['description'],
            "thumbnail": item['snippet']['thumbnails']['medium']['url']
        })
    
    return resources

def main():
    st.set_page_config(page_title="Advanced Exam Preparation System", layout="wide")
    
    st.sidebar.title("Advanced Exam Prep")
    st.sidebar.markdown("""
    Welcome to our advanced exam preparation system! 
    Here you can generate practice questions, explore educational resources, 
    and interact with an AI tutor to enhance your learning experience.
    """)
    
    # Main area tabs
    tab1, tab2, tab3 = st.tabs(["Question Generator", "Resource Explorer", "Academic Tutor"])
    
    with tab1:
        st.header("Question Generator")
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Enter the exam topic:")
            exam_type = st.selectbox("Select exam type:", ["General", "STEM", "Humanities", "Business", "Custom"])
        with col2:
            difficulty = st.select_slider(
                "Select difficulty level:",
                options=["Super Easy", "Easy", "Beginner", "Intermediate", "Higher Intermediate", "Master", "Advanced"]
            )
            num_questions = st.number_input("Number of questions:", min_value=1, max_value=50, value=5)
        include_answers = st.checkbox("Include answers", value=True)
        
        if st.button("Generate Questions", key="generate_questions"):
            if topic:
                with st.spinner("Generating questions..."):
                    questions = generate_questions(topic, difficulty, num_questions, include_answers)
                st.success("Questions generated successfully!")
                st.markdown(questions)
            else:
                st.warning("Please enter a topic.")
    
    with tab2:
        st.header("Resource Explorer")
        selected_field = st.selectbox("Select a field to explore:", FIELDS)
        if st.button("Explore Resources", key="explore_resources"):
            with st.spinner("Gathering resources..."):
                resources = gather_resources(selected_field)
            st.success(f"Found {len(resources)} resources!")
            
            for i, resource in enumerate(resources):
                col1, col2 = st.columns([1, 3])
                with col1:
                    if "thumbnail" in resource:
                        st.image(resource["thumbnail"], use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/150", use_column_width=True)
                with col2:
                    st.subheader(f"[{resource['title']}]({resource['link']})")
                    st.write(resource['content'])
                st.markdown("---")
    
    with tab3:
        st.header("Academic Tutor")
        uploaded_files = st.file_uploader("Upload documents (PDF, TXT, MD, DOC, DOCX)", type=["pdf", "txt", "md", "doc", "docx"], accept_multiple_files=True)
        
        if uploaded_files:
            qa_chain, graph = process_documents(uploaded_files)
            st.success("Documents processed successfully!")
        else:
            qa_chain, graph = None, None
        
        st.subheader("Chat with AI Tutor")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                with st.chat_message(role):
                    st.write(message)

        user_input = st.chat_input("Ask a question or type 'search: your query' to perform a web search:")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                if user_input.lower().startswith("search:"):
                    search_query = user_input[7:].strip()
                    search_results = search_web(search_query, num_results=3)
                    response = f"Here are some search results for '{search_query}':\n\n"
                    for result in search_results:
                        response += f"- [{result['title']}]({result['link']})\n  {result['snippet']}\n\n"
                else:
                    response = chat([HumanMessage(content=user_input)]).content
                st.write(response)
                st.session_state.chat_history.append(("assistant", response))

        # Scroll to bottom of chat
        js = f"""
        <script>
            function scroll_to_bottom() {{
                var chatElement = window.parent.document.querySelector('.stChatFloatingInputContainer');
                chatElement.scrollIntoView({{behavior: 'smooth'}});
            }}
            scroll_to_bottom();
        </script>
        """
        st.components.v1.html(js)

if __name__ == "__main__":
    main()
