import streamlit as st
import random
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import base64
import cv2
import numpy as np
from PIL import Image
import io
import time
import PyPDF2
import docx
import markdown

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
    timeout=60,
)

# Expanded list of roles
roles = [
    "Software Engineer", "Data Scientist", "Product Manager", "UX Designer", "Marketing Manager",
    "Sales Representative", "Human Resources Manager", "Financial Analyst", "Project Manager",
    "Business Analyst", "Content Writer", "Graphic Designer", "Customer Service Representative",
    "Operations Manager", "Research Scientist", "Legal Counsel", "Network Administrator",
    "Quality Assurance Tester", "Supply Chain Manager", "Public Relations Specialist"
]

def generate_interview_questions(role):
    system_message = f"""You are an experienced interviewer for the role of {role}. 
    Generate 5 challenging and relevant interview questions for this position. 
    The questions should cover a range of skills and experiences required for the role."""

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content="Please provide 5 interview questions for this role.")
    ]

    response = chat.invoke(messages).content
    questions = response.split('\n')
    return [q.strip() for q in questions if q.strip()]

def get_interview_response(role, question, answer):
    system_message = f"""You are an experienced interviewer for the role of {role}. 
    Your task is to evaluate the candidate's response to the following question: '{question}'
    
    The candidate's answer was: '{answer}'
    
    Please provide:
    1. A brief evaluation of the answer (2-3 sentences)
    2. Specific feedback on how to improve (if needed) or praise for a good answer
    3. A follow-up question based on their response
    4. A score out of 10 for their answer
    
    Format your response as follows:
    Evaluation: [Your evaluation here]
    Feedback: [Your specific feedback or praise here]
    Follow-up: [Your follow-up question here]
    Score: [Score out of 10]
    """

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content="Please provide your evaluation, feedback, follow-up question, and score.")
    ]

    response = chat.invoke(messages).content
    return response

def analyze_appearance(image):
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    analysis = []
    
    if len(faces) == 0:
        analysis.append("No face detected in the image. Please ensure your face is clearly visible.")
    else:
        analysis.append(f"Detected {len(faces)} face(s) in the image.")
        
        # Analyze facial positioning
        for (x, y, w, h) in faces:
            face_center = (x + w//2, y + h//2)
            image_center = (cv_image.shape[1]//2, cv_image.shape[0]//2)
            
            if abs(face_center[0] - image_center[0]) > cv_image.shape[1]//8:
                analysis.append("Your face is not centered horizontally. Try to position yourself in the middle of the frame.")
            
            if abs(face_center[1] - image_center[1]) > cv_image.shape[0]//8:
                analysis.append("Your face is not centered vertically. Adjust your camera or seating position.")
            
            if w * h < (cv_image.shape[0] * cv_image.shape[1]) // 16:
                analysis.append("Your face appears too small in the frame. Consider moving closer to the camera.")
            elif w * h > (cv_image.shape[0] * cv_image.shape[1]) // 4:
                analysis.append("Your face appears too large in the frame. Consider moving slightly away from the camera.")
    
    # Analyze image brightness
    brightness = np.mean(gray)
    if brightness < 100:
        analysis.append("The image appears too dark. Consider improving your lighting for better visibility.")
    elif brightness > 200:
        analysis.append("The image appears too bright. You might want to reduce harsh lighting or adjust your camera settings.")
    
    # Analyze image contrast
    contrast = np.std(gray)
    if contrast < 20:
        analysis.append("The image lacks contrast. This might make it difficult to see details. Consider adjusting your lighting or camera settings.")
    
    return "\n".join(analysis)

def extract_text_from_file(file):
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension == 'docx':
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    elif file_extension == 'txt':
        text = file.read().decode()
    elif file_extension == 'md':
        md_text = file.read().decode()
        text = markdown.markdown(md_text)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return text

def analyze_cv(cv_text):
    system_message = """You are an expert CV reviewer with extensive experience in various industries. 
    Analyze the given CV and provide:
    1. An overall assessment of the CV's strengths
    2. Areas that need improvement
    3. Specific suggestions for enhancing the CV
    4. Tips for tailoring the CV to specific job applications

    Be thorough, constructive, and provide actionable advice."""

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"Here's the text of the CV to review:\n\n{cv_text}\n\nPlease provide your analysis and suggestions.")
    ]

    response = chat.invoke(messages).content
    return response

def resize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio"""
    ratio = max_size / max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    return image.resize(new_size, Image.LANCZOS)

def get_mock_interview_tips():
    tips = [
        "Research the company and role thoroughly before the interview.",
        "Practice common interview questions with a friend or family member.",
        "Prepare specific examples to illustrate your skills and experiences.",
        "Dress professionally and ensure your background is tidy for video interviews.",
        "Have questions prepared to ask the interviewer about the role and company.",
        "Use the STAR method (Situation, Task, Action, Result) to structure your answers.",
        "Be aware of your body language and maintain good eye contact.",
        "Listen carefully to each question and take a moment to gather your thoughts before answering.",
        "Be honest about your experiences and skills, but focus on your strengths.",
        "Follow up with a thank-you note or email after the interview.",
    ]
    return tips

def get_interview_resources():
    resources = [
        {"name": "Glassdoor Interview Questions & Reviews", "url": "https://www.glassdoor.com/Interview/index.htm"},
        {"name": "LinkedIn Interview Preparation", "url": "https://www.linkedin.com/interview-prep/"},
        {"name": "Indeed Career Guide", "url": "https://www.indeed.com/career-advice"},
        {"name": "Coursera - How to Succeed in an Interview", "url": "https://www.coursera.org/learn/interview-preparation"},
        {"name": "Harvard Business Review - Interview Tips", "url": "https://hbr.org/topic/interviewing"},
    ]
    return resources

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. Interview Preparation", page_icon="🎙️", layout="wide")

    st.title("🎙️ S.H.E.R.L.O.C.K. Interview Preparation")
    st.markdown("### Streamlined Help for Enhancing Responsive Learning and Optimizing Career Knowledge")

    # Sidebar for user details and interview settings
    with st.sidebar:
        st.header("Interview Settings")
        name = st.text_input("Your Name")
        role = st.selectbox("Interview Role", roles)
        experience = st.slider("Years of Experience", 0, 20, 5)

        st.header("Quick Tips")
        if st.button("Get Mock Interview Tips"):
            tips = get_mock_interview_tips()
            for tip in tips:
                st.info(tip)

        st.header("Useful Resources")
        resources = get_interview_resources()
        for resource in resources:
            st.markdown(f"[{resource['name']}]({resource['url']})")

    # Appearance Analysis
    st.header("Appearance Analysis")
    uploaded_image = st.file_uploader("Upload your interview outfit image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            image = resize_image(image)
            st.image(image, caption="Your uploaded image", use_column_width=True)
            if st.button("Analyze Appearance"):
                with st.spinner("Analyzing your appearance..."):
                    appearance_feedback = analyze_appearance(image)
                    st.write(appearance_feedback)
                    
                    st.write("\nGeneral tips for professional appearance in video interviews:")
                    tips = [
                        "Dress professionally from head to toe, even if only your upper body is visible.",
                        "Choose solid colors over busy patterns for a less distracting appearance.",
                        "Ensure your background is tidy and professional.",
                        "Position your camera at eye level for the most flattering angle.",
                        "Use soft, diffused lighting to avoid harsh shadows.",
                        "Make eye contact by looking directly into the camera when speaking.",
                    ]
                    for tip in tips:
                        st.write(f"- {tip}")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
            st.info("Please make sure you've uploaded a valid image file.")

    # CV Analysis
    st.header("CV Analysis")
    uploaded_cv = st.file_uploader("Upload your CV", type=["pdf", "docx", "txt", "md"])
    if uploaded_cv is not None:
        try:
            cv_text = extract_text_from_file(uploaded_cv)
            if st.button("Analyze CV"):
                with st.spinner("Analyzing your CV..."):
                    cv_feedback = analyze_cv(cv_text)
                st.write(cv_feedback)
        except Exception as e:
            st.error(f"An error occurred while processing the CV: {str(e)}")

    # Initialize session state variables
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'feedback' not in st.session_state:
        st.session_state.feedback = []
    if 'scores' not in st.session_state:
        st.session_state.scores = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Start Interview button
    if not st.session_state.interview_started:
        if st.button("Start Mock Interview"):
            if name and role:
                st.session_state.interview_started = True
                with st.spinner("Generating interview questions..."):
                    st.session_state.questions = generate_interview_questions(role)
                st.rerun()
            else:
                st.warning("Please enter your name and select a role before starting the interview.")

    # Interview in progress
    if st.session_state.interview_started:
        st.header("Mock Interview")
        if st.session_state.current_question < len(st.session_state.questions):
            st.subheader(f"Question {st.session_state.current_question + 1}")
            st.write(st.session_state.questions[st.session_state.current_question])

            # Display chat history
            for i, (q, a, f) in enumerate(st.session_state.chat_history):
                with st.expander(f"Question {i+1}"):
                    st.write(f"Q: {q}")
                    st.write(f"Your Answer: {a}")
                    st.write(f"Feedback: {f}")

            answer = st.text_area("Your Answer", key=f"answer_{st.session_state.current_question}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Answer"):
                    if answer:
                        with st.spinner("Evaluating your answer..."):
                            response = get_interview_response(role, st.session_state.questions[st.session_state.current_question], answer)
                            st.session_state.answers.append(answer)
                            st.session_state.feedback.append(response)
                            
                            # Extract score from response
                            score_lines = [line for line in response.split('\n') if line.startswith('Score:')]
                            if score_lines:
                                score_str = score_lines[0].split(':')[1].strip()
                                try:
                                    score = int(score_str)
                                except ValueError:
                                    # If the score is a fraction like "6/10", extract the numerator
                                    score = int(score_str.split('/')[0])
                            else:
                                # If no score is found, use a default value
                                score = 5  # or any other default value you prefer
                                st.warning("No score was provided in the response. Using a default score of 5.")
                            
                            st.session_state.scores.append(score)

                            # Update chat history
                            st.session_state.chat_history.append((
                                st.session_state.questions[st.session_state.current_question],
                                answer,
                                response
                            ))

                        st.session_state.current_question += 1
                        if st.session_state.current_question < len(st.session_state.questions):
                            st.rerun()
                    else:
                        st.warning("Please provide an answer before submitting.")
            with col2:
                if st.button("Skip Question"):
                    st.session_state.current_question += 1
                    if st.session_state.current_question < len(st.session_state.questions):
                        st.rerun()

        else:
            st.success("Interview Completed!")
            total_score = sum(st.session_state.scores)
            average_score = total_score / len(st.session_state.scores)

            st.header("Interview Summary")
            st.subheader(f"Overall Score: {average_score:.2f}/10")

            for i, (q, a, f) in enumerate(st.session_state.chat_history):
                with st.expander(f"Question {i+1}"):
                    st.write(f"Q: {q}")
                    st.write(f"Your Answer: {a}")
                    st.write(f"Feedback: {f}")

            # Generate overall feedback
            overall_feedback_prompt = f"""
            You are an experienced career coach. Based on the candidate's performance in the interview for the role of {role},
            with {experience} years of experience, please provide:
            1. A summary of their strengths (2-3 points)
            2. Areas for improvement (2-3 points)
            3. Advice for future interviews (2-3 tips)
            4. Personalized tips for improving their professional appearance and body language
            5. Strategies for managing interview anxiety

            Their overall score was {average_score:.2f}/10.

            Format your response as follows:
            Strengths:
            - [Strength 1]
            - [Strength 2]
            - [Strength 3]

            Areas for Improvement:
            - [Area 1]
            - [Area 2]
            - [Area 3]

            Tips for Future Interviews:
            - [Tip 1]
            - [Tip 2]
            - [Tip 3]

            Professional Appearance and Body Language:
            - [Tip 1]
            - [Tip 2]
            - [Tip 3]

            Managing Interview Anxiety:
            - [Strategy 1]
            - [Strategy 2]
            - [Strategy 3]
            """

            messages = [
                SystemMessage(content=overall_feedback_prompt),
                HumanMessage(content="Please provide the overall feedback for the interview.")
            ]

            with st.spinner("Generating overall feedback..."):
                overall_feedback = chat.invoke(messages).content

            st.subheader("Overall Feedback")
            st.write(overall_feedback)

            if st.button("Start New Interview"):
                st.session_state.interview_started = False
                st.session_state.current_question = 0
                st.session_state.questions = []
                st.session_state.answers = []
                st.session_state.feedback = []
                st.session_state.scores = []
                st.session_state.chat_history = []
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("Powered by Falcon-180B and Streamlit")

    # Interview Preparation Checklist
    st.sidebar.header("Interview Preparation Checklist")
    checklist_items = [
        "Research the company",
        "Review the job description",
        "Prepare your elevator pitch",
        "Practice common interview questions",
        "Prepare questions for the interviewer",
        "Choose appropriate attire",
        "Test your technology (for virtual interviews)",
        "Gather necessary documents (resume, portfolio, etc.)",
        "Plan your route or set up your interview space",
        "Get a good night's sleep"
    ]
    for item in checklist_items:
        st.sidebar.checkbox(item)

    # Interview Timer
    if st.session_state.interview_started:
        st.sidebar.header("Interview Timer")
        if 'start_time' not in st.session_state:
            st.session_state.start_time = time.time()
        
        elapsed_time = int(time.time() - st.session_state.start_time)
        minutes, seconds = divmod(elapsed_time, 60)
        st.sidebar.write(f"Elapsed Time: {minutes:02d}:{seconds:02d}")

    # Confidence Boost
    st.sidebar.header("Confidence Boost")
    if st.sidebar.button("Get a Confidence Boost"):
        confidence_boosters = [
            "You've got this! Your preparation will pay off.",
            "Remember, the interviewer wants you to succeed too.",
            "Take deep breaths and stay calm. You're well-prepared.",
            "Your unique experiences make you a valuable candidate.",
            "Every interview is a learning opportunity. Embrace it!",
            "Believe in yourself. Your skills and knowledge are valuable.",
            "Stay positive and confident. Your attitude shines through.",
            "You've overcome challenges before. This is just another opportunity to shine.",
            "Focus on your strengths and what you can bring to the role.",
            "Remember your past successes. You're capable of greatness!"
        ]
        st.sidebar.success(random.choice(confidence_boosters))

    # Interview Do's and Don'ts
    st.sidebar.header("Interview Do's and Don'ts")
    dos_and_donts = {
        "Do": [
            "Arrive early or log in on time",
            "Maintain good eye contact",
            "Listen actively and ask thoughtful questions",
            "Show enthusiasm for the role and company",
            "Provide specific examples to support your answers"
        ],
        "Don't": [
            "Speak negatively about past employers",
            "Interrupt the interviewer",
            "Use filler words excessively (um, like, you know)",
            "Check your phone or watch frequently",
            "Provide vague or generic answers"
        ]
    }
    dos_tab, donts_tab = st.sidebar.tabs(["Do's", "Don'ts"])
    with dos_tab:
        for do_item in dos_and_donts["Do"]:
            st.write(f"✅ {do_item}")
    with donts_tab:
        for dont_item in dos_and_donts["Don't"]:
            st.write(f"❌ {dont_item}")

    # Personal Notes
    st.sidebar.header("Personal Notes")
    personal_notes = st.sidebar.text_area("Jot down your thoughts or reminders here:")

    # Initialize session state for saved notes if it doesn't exist
    if 'saved_notes' not in st.session_state:
        st.session_state.saved_notes = []

    # Save Notes button
    if st.sidebar.button("Save Notes"):
        if personal_notes.strip():  # Check if the note is not empty
            st.session_state.saved_notes.append(personal_notes)
            st.sidebar.success("Note saved successfully!")
            # Clear the text area after saving
            personal_notes = ""
        else:
            st.sidebar.warning("Please enter a note before saving.")

    # Display saved notes as checkboxes
    st.sidebar.subheader("Saved Notes")
    for i, note in enumerate(st.session_state.saved_notes):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.checkbox(note, key=f"note_{i}")
        with col2:
            if st.button("Delete", key=f"delete_{i}"):
                del st.session_state.saved_notes[i]
                st.rerun()
                
    # Follow-up Email Template
    if st.session_state.interview_started and st.session_state.current_question >= len(st.session_state.questions):
        st.header("Follow-up Email Template")
        interviewer_name = st.text_input("Interviewer's Name")
        company_name = st.text_input("Company Name")
        specific_topic = st.text_input("Specific topic discussed during the interview")
        
        if interviewer_name and company_name and specific_topic:
            email_template = f"""
            Subject: Thank you for the interview - {role} position

            Dear {interviewer_name},

            I hope this email finds you well. I wanted to express my sincere gratitude for taking the time to interview me for the {role} position at {company_name}. I thoroughly enjoyed our conversation and learning more about the role and the company.

            Our discussion about {specific_topic} was particularly interesting, and it reinforced my enthusiasm for the position. I am excited about the possibility of bringing my skills and experience to your team and contributing to {company_name}'s success.

            If you need any additional information or have any further questions, please don't hesitate to contact me. I look forward to hearing about the next steps in the process.

            Thank you again for your time and consideration.

            Best regards,
            {name}
            """
            st.text_area("Follow-up Email Template", email_template, height=300)
            if st.button("Copy to Clipboard"):
                st.write("Email template copied to clipboard!")
                # Note: In a web app, you'd use JavaScript to copy to clipboard

if __name__ == "__main__":
    main()