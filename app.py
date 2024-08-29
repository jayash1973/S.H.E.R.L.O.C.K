import streamlit as st
import os
import importlib

# Custom CSS for improved styling
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Move set_page_config to the very beginning of the script
st.set_page_config(
    page_title="S.H.E.R.L.O.C.K.",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the pages with icons
PAGES = {
    "Home": {"icon": "🏠", "module": None},
    "Web RAG Powered Chatbot": {"icon": "💬", "module": "chatbot"},
    "Notes Generation": {"icon": "📝", "module": "notes_generation"},
    "Exam Preparation": {"icon": "📚", "module": "exam_preparation"},
    "Mnemonics Generation": {"icon": "🧠", "module": "mnemonics_generation"},
    "Study Roadmap": {"icon": "🗺️", "module": "study_roadmap"},
    "Interview Preparation": {"icon": "🎤", "module": "interview_preparation"},
    "AI Buddy": {"icon": "🤖🧘", "module": "ai_buddy"},
    "Mind Palace Builder": {"icon": "🏛️", "module": "mind_palace"},
    "Sherlock Style Observation": {"icon": "🔍", "module": "sherlock_observation"},
    "Research Paper Finder": {"icon": "🏛️", "module": "research_paper_finder"},
    "Lecture Finder": {"icon": "🔍", "module": "lecture_finder"},
    "Resume Generator": {"icon": "📝", "module": "resume_generator"}
}

def load_module(module_name):
    if module_name is None:
        return None
    try:
        return importlib.import_module(f"pages.{module_name}")
    except ImportError:
        st.error(f"Unable to load module: {module_name}. Make sure the file exists in the 'pages' directory.")
        return None

def main():
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/cd/Sherlock_Holmes_Portrait_Paget.jpg", use_column_width=True)
    st.sidebar.title("S.H.E.R.L.O.C.K. 🕵️")
    st.sidebar.markdown("*Study Helper & Educational Resource for Learning & Observational Knowledge*")
    
    # Apply custom CSS
    st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f0f2f6;
        }
        .stButton button {
            background-color: #0e1117;
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #2e7d32;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .sidebar .sidebar-content {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3 {
            color: #1e3a8a;
        }
        .stRadio > label {
            font-weight: bold;
            color: #333;
        }
        .feature-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        .feature-icon {
            font-size: 2em;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    selection = st.sidebar.radio(
        "Navigate",
        list(PAGES.keys()),
        format_func=lambda x: f"{PAGES[x]['icon']} {x}"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app is part of the S.H.E.R.L.O.C.K. project. "
        "For more information, visit [our website](https://sherlock.vercel.app/)."
    )
    st.sidebar.text("Version 1.0")
    
    # Main content area
    if selection == "Home":
        st.title("Welcome to S.H.E.R.L.O.C.K. 🕵️")
        st.markdown("""
        *Systematic Holistic Educational Resource for Learning and Optimizing Cognitive Knowledge*
        
        S.H.E.R.L.O.C.K. is an advanced AI-powered personalized learning assistant designed to revolutionize your educational journey. By combining cutting-edge artificial intelligence with time-tested learning techniques, S.H.E.R.L.O.C.K. aims to enhance your cognitive abilities, strengthen your memory, and deepen your subject-specific knowledge.
        
        Our platform offers a comprehensive suite of tools and features that cater to various aspects of learning and personal development. From AI-driven chatbots and customized study plans to innovative memory techniques and mindfulness practices, S.H.E.R.L.O.C.K. is your all-in-one companion for academic success and personal growth.
        
        Explore our features below and embark on a journey to unlock your full learning potential!
        """)
        st.markdown("## Features")
        cols = st.columns(3)
        for idx, (feature, details) in enumerate(list(PAGES.items())[1:]):  # Skip "Home"
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon">{details['icon']}</div>
                    <h3>{feature}</h3>
                    <p>{get_feature_description(feature)}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.title(f"{PAGES[selection]['icon']} {selection}")
        st.markdown(f"*{get_feature_description(selection)}*")
        st.markdown("---")
        
        # Load and run the selected module
        module = load_module(PAGES[selection]['module'])
        if module and hasattr(module, 'main'):
            module.main()
        else:
            st.error(f"Unable to load the {selection} feature. Please check the module implementation.")

def get_feature_description(feature):
    descriptions = {
        "Web RAG Powered Chatbot": "Engage with our state-of-the-art AI-powered chatbot that leverages Retrieval-Augmented Generation (RAG) technology. This intelligent assistant provides interactive learning experiences by retrieving and synthesizing information from vast web-based resources, offering you accurate, context-aware responses to your queries.",
        "Notes Generation": "Transform complex documents and lengthy lectures into concise, easy-to-understand notes. Our AI analyzes the content, extracts key points, and presents them in a clear, structured format, helping you grasp essential concepts quickly and efficiently.",
        "Exam Preparation": "Ace your exams with our intelligent test preparation system. Generate custom question papers, practice tests, and quizzes tailored to your specific syllabus and learning progress. Receive instant feedback and targeted recommendations to improve your performance.",
        "Mnemonics Generation": "Boost your memory and retention with personalized mnemonic devices. Our AI creates custom memory aids, including acronyms, rhymes, and vivid imagery, helping you remember complex information effortlessly and enhancing your long-term recall abilities.",
        "Study Roadmap": "Navigate your learning journey with a personalized, adaptive study plan. Our AI analyzes your goals, strengths, and areas for improvement to create a tailored learning path, optimizing your study time and ensuring efficient progress towards your educational objectives.",
        "Interview Preparation": "Master the art of interviewing with our advanced simulation system. Experience realistic interview scenarios, receive real-time feedback on your responses, and gain insights into improving your communication skills, body language, and overall performance.",
        "AI Buddy": "Connect with a compassionate AI companion designed to provide emotional support, motivation, and guidance. Cultivate mental clarity and emotional balance with our curated collection of meditation and mindfulness resources. Engage in thoughtful conversations, receive personalized advice, and enjoy a judgment-free space for self-reflection and personal growth.",
        "Mind Palace Builder": "Construct powerful mental frameworks to enhance your memory and learning capabilities. Our interactive system guides you through creating and populating your own virtual 'mind palace', allowing you to organize and recall vast amounts of information with ease.",
        "Sherlock Style Observation": "Sharpen your critical thinking and observational skills using techniques inspired by the legendary detective. Learn to notice details, make logical deductions, and approach problems from unique perspectives, enhancing your analytical abilities across various subjects.",
        "Research Paper Finder": "Discover relevant academic literature effortlessly with our intelligent research paper finder. Input your topic of interest, and our AI will scour databases to present you with a curated list of papers, saving you valuable research time and ensuring you stay up-to-date with the latest findings in your field.",
        "Lecture Finder": "Expand your knowledge horizons with our smart lecture discovery tool. Find high-quality video lectures from reputable sources on YouTube, covering any topic you wish to explore. Our AI curates content based on your learning preferences and academic level.",
        "Resume Generator": "Create a standout resume tailored to your unique skills and experiences. Our AI-powered resume generator analyzes your input and crafts a professional, ATS-friendly document that highlights your strengths and aligns with industry standards, increasing your chances of landing your dream job."
    }
    return descriptions.get(feature, "Description not available.")

if __name__ == "__main__":
    main()