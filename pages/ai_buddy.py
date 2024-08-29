import streamlit as st
import random
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import plotly.express as px
import json
import tempfile
import time
import numpy as np
import threading
import sounddevice as sd
from playsound import playsound
import pygame
from scipy.io import wavfile

pygame.mixer.init()

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

# Expanded Therapy techniques
THERAPY_TECHNIQUES = {
    "CBT": "Use Cognitive Behavioral Therapy techniques to help the user identify and change negative thought patterns.",
    "Mindfulness": "Guide the user through mindfulness exercises to promote present-moment awareness and reduce stress.",
    "Solution-Focused": "Focus on the user's strengths and resources to help them find solutions to their problems.",
    "Emotion-Focused": "Help the user identify, experience, and regulate their emotions more effectively.",
    "Psychodynamic": "Explore the user's past experiences and unconscious patterns to gain insight into current issues.",
    "ACT": "Use Acceptance and Commitment Therapy to help the user accept their thoughts and feelings while committing to positive changes.",
    "DBT": "Apply Dialectical Behavior Therapy techniques to help the user manage intense emotions and improve relationships.",
    "Gestalt": "Use Gestalt therapy techniques to focus on the present moment and increase self-awareness.",
    "Existential": "Explore existential themes such as meaning, freedom, and responsibility to help the user find purpose.",
    "Narrative": "Use storytelling and narrative techniques to help the user reframe their life experiences and create new meaning.",
}

def get_ai_response(user_input, buddy_config, therapy_technique=None):
    system_message = f"You are {buddy_config['name']}, an AI companion with the following personality: {buddy_config['personality']}. "
    system_message += f"Additional details about you: {buddy_config['details']}. "
    
    if therapy_technique:
        system_message += f"In this conversation, {THERAPY_TECHNIQUES[therapy_technique]}"
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_input)
    ]
    response = chat.invoke(messages).content
    return response

def play_sound_loop(sound_file, stop_event):
    while not stop_event.is_set():
        playsound(sound_file)

def play_sound_for_duration(sound_file, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        playsound(sound_file, block=False)
        time.sleep(0.1)  # Short sleep to prevent excessive CPU usage
    # Ensure the sound stops after the duration
    pygame.mixer.quit()

def get_sound_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.mp3')]

def get_sound_file_path(sound_name, sound_dir):
    # Convert the sound name to a filename
    filename = f"{sound_name.lower().replace(' ', '_')}.mp3"
    return os.path.join(sound_dir, filename)

SOUND_OPTIONS = [
    "Gentle Rain", "Ocean Waves", "Forest Ambience", "Soft Wind Chimes",
    "Tibetan Singing Bowls", "Humming Song", "Crackling Fireplace",
    "Birdsong", "White Noise", "Zen River", "Heartbeat", "Deep Space",
    "Whale Songs", "Bamboo Flute", "Thunderstorm", "Cat Purring",
    "Campfire", "Windchimes", "Waterfall", "Beach Waves", "Cicadas",
    "Coffee Shop Ambience", "Grandfather Clock", "Rainstorm on Tent",
    "Tropical Birds", "Subway Train", "Washing Machine", "Fan White Noise",
    "Tibetan Bells", "Wind in Trees", "Meditation Bowl", "Meditation Bowl2", "Birds Singing Rainy Day"
]

def show_meditation_timer():
    st.subheader("🧘‍♀️ Enhanced Meditation Timer")
    
    sound_dir = os.path.join(os.path.dirname(__file__), "..", "sounds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Select duration (minutes)", 1, 60, 5)
        background_sound = st.selectbox("Background Sound", SOUND_OPTIONS)
        
    with col2:
        interval_options = ["None", "Every 5 minutes", "Every 10 minutes"]
        interval_reminder = st.selectbox("Interval Reminders", interval_options)
        end_sound = st.selectbox("End of Session Sound", SOUND_OPTIONS)
    
    if st.button("Start Meditation", key="start_meditation"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load background sound
        background_sound_file = get_sound_file_path(background_sound, sound_dir)
        if not os.path.exists(background_sound_file):
            st.error(f"Background sound file not found: {background_sound_file}")
            return
        
        # Load end of session sound
        end_sound_file = get_sound_file_path(end_sound, sound_dir)
        if not os.path.exists(end_sound_file):
            st.error(f"End sound file not found: {end_sound_file}")
            return
        
        # Play background sound on loop
        pygame.mixer.music.load(background_sound_file)
        pygame.mixer.music.play(-1)  # -1 means loop indefinitely
        
        start_time = time.time()
        end_time = start_time + (duration * 60)
        
        try:
            while time.time() < end_time:
                elapsed_time = time.time() - start_time
                progress = elapsed_time / (duration * 60)
                progress_bar.progress(progress)
                
                remaining_time = end_time - time.time()
                mins, secs = divmod(int(remaining_time), 60)
                status_text.text(f"Time remaining: {mins:02d}:{secs:02d}")
                
                if interval_reminder != "None":
                    interval = 5 if interval_reminder == "Every 5 minutes" else 10
                    if int(elapsed_time) > 0 and int(elapsed_time) % (interval * 60) == 0:
                        st.toast(f"{interval} minutes passed", icon="⏰")
                
                # Check if 10 seconds remaining
                if remaining_time <= 10 and remaining_time > 9:
                    pygame.mixer.music.stop()  # Stop background sound
                    pygame.mixer.Sound(end_sound_file).play()  # Play end sound
                
                if remaining_time <= 0:
                    break
                
                time.sleep(0.1)  # Update more frequently for smoother countdown
        finally:
            # Stop all sounds
            pygame.mixer.quit()
        
        # Ensure the progress bar is full and time remaining shows 00:00
        progress_bar.progress(1.0)
        status_text.text("Time remaining: 00:00")
        
        st.success("Meditation complete!")
        st.balloons()
        
        if 'achievements' not in st.session_state:
            st.session_state.achievements = set()
        st.session_state.achievements.add("Zen Master")
        st.success("Achievement Unlocked: Zen Master 🧘‍♀️")

def show_personalized_recommendations():
    st.subheader("🎯 Personalized Recommendations")
    
    recommendation_categories = [
        "Mental Health",
        "Physical Health",
        "Personal Development",
        "Relationships",
        "Career",
        "Hobbies",
    ]
    
    selected_category = st.selectbox("Choose a category", recommendation_categories)
    
    recommendations = {
        "Mental Health": [
            "Practice daily gratitude journaling",
            "Try a guided meditation for stress relief",
            "Explore cognitive behavioral therapy techniques",
            "Start a mood tracking journal",
            "Learn about mindfulness practices",
        ],
        "Physical Health": [
            "Start a 30-day yoga challenge",
            "Try intermittent fasting",
            "Begin a couch to 5K running program",
            "Experiment with new healthy recipes",
            "Create a sleep hygiene routine",
        ],
        "Personal Development": [
            "Start learning a new language",
            "Read personal development books",
            "Take an online course in a subject you're interested in",
            "Practice public speaking",
            "Start a daily writing habit",
        ],
        "Relationships": [
            "Practice active listening techniques",
            "Plan regular date nights or friend meetups",
            "Learn about love languages",
            "Practice expressing gratitude to loved ones",
            "Join a local community or interest group",
        ],
        "Career": [
            "Update your resume and LinkedIn profile",
            "Network with professionals in your industry",
            "Set SMART career goals",
            "Learn a new skill relevant to your field",
            "Start a side project or freelance work",
        ],
        "Hobbies": [
            "Start a garden or learn about plant care",
            "Try a new art form like painting or sculpting",
            "Learn to play a musical instrument",
            "Start a DIY home improvement project",
            "Explore photography or videography",
        ],
    }
    
    st.write("Here are some personalized recommendations for you:")
    for recommendation in recommendations[selected_category]:
        st.markdown(f"- {recommendation}")
    
    if st.button("Get More Recommendations"):
        st.write("More tailored recommendations:")
        additional_recs = random.sample(recommendations[selected_category], 3)
        for rec in additional_recs:
            st.markdown(f"- {rec}")

def generate_binaural_beat(freq1, freq2, duration_seconds, sample_rate=44100):
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    left_channel = np.sin(2 * np.pi * freq1 * t)
    right_channel = np.sin(2 * np.pi * freq2 * t)
    stereo_audio = np.vstack((left_channel, right_channel)).T
    return (stereo_audio * 32767).astype(np.int16)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    b64 = base64.b64encode(bin_file).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}.wav" class="download-link">Download {file_label}</a>'

def show_binaural_beats():
    st.subheader("🎵 Binaural Beats Generator")
    
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .download-link {
        background-color: #008CBA;
        color: white;
        padding: 10px 15px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stop-button {
        background-color: #f44336;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("Binaural beats are created when two slightly different frequencies are played in each ear, potentially influencing brainwave activity.")
    
    preset_beats = {
        "Deep Relaxation (Delta)": {"base": 100, "beat": 2},
        "Meditation (Theta)": {"base": 150, "beat": 6},
        "Relaxation (Alpha)": {"base": 200, "beat": 10},
        "Light Focus (Low Beta)": {"base": 250, "beat": 14},
        "High Focus (Mid Beta)": {"base": 300, "beat": 20},
        "Alertness (High Beta)": {"base": 350, "beat": 30},
        "Gamma Consciousness": {"base": 400, "beat": 40},
        "Lucid Dreaming": {"base": 180, "beat": 3},
        "Memory Enhancement": {"base": 270, "beat": 12},
        "Creativity Boost": {"base": 220, "beat": 8},
        "Pain Relief": {"base": 130, "beat": 4},
        "Mood Elevation": {"base": 315, "beat": 18}
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        beat_type = st.selectbox("Choose a preset or custom:", ["Custom"] + list(preset_beats.keys()))
    
    with col2:
        duration = st.slider("Duration (minutes):", 1, 60, 15)
    
    if beat_type == "Custom":
        col3, col4 = st.columns(2)
        with col3:
            base_freq = st.slider("Base Frequency (Hz):", 100, 500, 200)
        with col4:
            beat_freq = st.slider("Desired Beat Frequency (Hz):", 1, 40, 10)
    else:
        base_freq = preset_beats[beat_type]["base"]
        beat_freq = preset_beats[beat_type]["beat"]
        st.info(f"Base Frequency: {base_freq} Hz, Beat Frequency: {beat_freq} Hz")
    
    if 'audio_playing' not in st.session_state:
        st.session_state.audio_playing = False
    
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    
    if 'end_time' not in st.session_state:
        st.session_state.end_time = None
    
    # Create persistent placeholders for UI elements
    progress_bar = st.empty()
    status_text = st.empty()
    stop_button = st.empty()
    
    generate_button = st.button("Generate and Play Binaural Beat")
    
    if generate_button:
        try:
            # Stop any currently playing audio
            if st.session_state.audio_playing:
                pygame.mixer.music.stop()
                st.session_state.audio_playing = False
            
            audio_data = generate_binaural_beat(base_freq, base_freq + beat_freq, duration * 60)
            
            # Save the generated audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_filename = temp_file.name
                wavfile.write(temp_filename, 44100, audio_data)
            
            # Initialize pygame mixer
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
            
            # Load and play the audio
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            st.session_state.audio_playing = True
            
            st.session_state.start_time = time.time()
            st.session_state.end_time = st.session_state.start_time + (duration * 60)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")
    
    if st.session_state.audio_playing:
        stop_button_active = stop_button.button("Stop Binaural Beat", key="stop_binaural", type="primary")
        current_time = time.time()
        
        if stop_button_active:
            pygame.mixer.music.stop()
            st.session_state.audio_playing = False
            st.session_state.start_time = None
            st.session_state.end_time = None
        
        elif current_time < st.session_state.end_time:
            elapsed_time = current_time - st.session_state.start_time
            progress = elapsed_time / (st.session_state.end_time - st.session_state.start_time)
            progress_bar.progress(progress)
            
            remaining_time = st.session_state.end_time - current_time
            mins, secs = divmod(int(remaining_time), 60)
            status_text.text(f"Time remaining: {mins:02d}:{secs:02d}")
        else:
            pygame.mixer.music.stop()
            st.session_state.audio_playing = False
            st.session_state.start_time = None
            st.session_state.end_time = None
            progress_bar.empty()
            status_text.text("Binaural beat session complete!")
    
    # Offer download of the generated audio
    if not st.session_state.audio_playing and 'audio_data' in locals():
        with io.BytesIO() as buffer:
            wavfile.write(buffer, 44100, audio_data)
            st.markdown(get_binary_file_downloader_html(buffer.getvalue(), f"binaural_beat_{base_freq}_{beat_freq}Hz"), unsafe_allow_html=True)

    # Ensure the app updates every second
    if st.session_state.audio_playing:
        time.sleep(1)
        st.experimental_rerun()

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. AI Buddy", page_icon="🕵️", layout="wide")
    
    # Custom CSS for improved styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
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
    
    st.title("🕵️ S.H.E.R.L.O.C.K. AI Buddy")
    st.markdown("Your personalized AI companion for conversation, therapy, and personal growth.")

    # Initialize session state
    if 'buddy_name' not in st.session_state:
        st.session_state.buddy_name = "Sherlock"
    if 'buddy_personality' not in st.session_state:
        st.session_state.buddy_personality = "Friendly, empathetic, and insightful"
    if 'buddy_details' not in st.session_state:
        st.session_state.buddy_details = "Knowledgeable about various therapy techniques and always ready to listen"
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar for AI Buddy configuration and additional features
    with st.sidebar:
        st.header("🤖 Configure Your AI Buddy")
        st.session_state.buddy_name = st.text_input("Name your AI Buddy", value=st.session_state.buddy_name)
        st.session_state.buddy_personality = st.text_area("Describe your buddy's personality", value=st.session_state.buddy_personality)
        st.session_state.buddy_details = st.text_area("Additional details about your buddy", value=st.session_state.buddy_details)
        
        st.header("🧘 Therapy Session")
        therapy_mode = st.checkbox("Enable Therapy Mode")
        if therapy_mode:
            therapy_technique = st.selectbox("Select Therapy Technique", list(THERAPY_TECHNIQUES.keys()))
        else:
            therapy_technique = None
        
        st.markdown("---")
        
        st.subheader("📋 Todo List")
        if 'todos' not in st.session_state:
            st.session_state.todos = []
        
        new_todo = st.text_input("Add a new todo:")
        if st.button("Add Todo", key="add_todo"):
            if new_todo:
                st.session_state.todos.append({"task": new_todo, "completed": False})
                st.success("Todo added successfully!")
            else:
                st.warning("Please enter a todo item.")
        
        for i, todo in enumerate(st.session_state.todos):
            col1, col2, col3 = st.columns([0.05, 0.8, 0.15])
            with col1:
                todo['completed'] = st.checkbox("", todo['completed'], key=f"todo_{i}")
            with col2:
                st.write(todo['task'], key=f"todo_text_{i}")
            with col3:
                if st.button("🗑️", key=f"delete_{i}", help="Delete todo"):
                    st.session_state.todos.pop(i)
                    st.experimental_rerun()

        st.subheader("⏱️ Pomodoro Timer")
        pomodoro_duration = st.slider("Pomodoro Duration (minutes)", 1, 60, 25)
        if st.button("Start Pomodoro"):
            progress_bar = st.progress(0)
            for i in range(pomodoro_duration * 60):
                time.sleep(1)
                progress_bar.progress((i + 1) / (pomodoro_duration * 60))
            st.success("Pomodoro completed!")
            if 'achievements' not in st.session_state:
                st.session_state.achievements = set()
            st.session_state.achievements.add("Consistent Learner")
        
        st.markdown("---")
        st.markdown("Powered by Falcon-180B and Streamlit")

        st.markdown("---")
        st.header("📔 Daily Journal")
        journal_entry = st.text_area("Write your thoughts for today")
        if st.button("Save Journal Entry"):
            if 'journal_entries' not in st.session_state:
                st.session_state.journal_entries = []
            st.session_state.journal_entries.append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'entry': journal_entry
            })
            st.success("Journal entry saved!")
            st.toast("Journal entry saved successfully!", icon="✅")
        
        if 'journal_entries' in st.session_state and st.session_state.journal_entries:
            st.subheader("Previous Entries")
            for entry in st.session_state.journal_entries[-5:]:  # Show last 5 entries
                st.text(entry['date'])
                st.write(entry['entry'])
                st.markdown("---")

    # Main content area
    tab1, tab2 = st.tabs(["Chat", "Tools"])
    
    with tab1:
        # Chat interface
        st.header("🗨️ Chat with Your AI Buddy")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # User input
        prompt = st.chat_input("What's on your mind?")
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            buddy_config = {
                "name": st.session_state.buddy_name,
                "personality": st.session_state.buddy_personality,
                "details": st.session_state.buddy_details
            }

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in chat.stream(get_ai_response(prompt, buddy_config, therapy_technique)):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Force a rerun to update the chat history immediately
            st.experimental_rerun()

    with tab2:
        tool_choice = st.selectbox("Select a tool", ["Meditation Timer", "Binaural Beats", "Recommendations"])
        if tool_choice == "Meditation Timer":
            show_meditation_timer()
        elif tool_choice == "Recommendations":
            show_personalized_recommendations()
        elif tool_choice == "Binaural Beats":
            show_binaural_beats()

    # Mood tracker
    st.sidebar.markdown("---")
    st.sidebar.header("😊 Mood Tracker")
    mood = st.sidebar.slider("How are you feeling today?", 1, 10, 5)
    if st.sidebar.button("Log Mood"):
        st.sidebar.success(f"Mood logged: {mood}/10")
        st.balloons()

    # Resources and Emergency Contact
    st.sidebar.markdown("---")
    st.sidebar.header("🆘 Resources")
    st.sidebar.info("If you're in crisis, please reach out for help:")
    st.sidebar.markdown("- [Mental Health Resources](https://www.mentalhealth.gov/get-help/immediate-help)")
    st.sidebar.markdown("- Emergency Contact: 911 or your local emergency number")

    # Inspiration Quote
    st.sidebar.markdown("---")
    st.sidebar.header("💡 Daily Inspiration")
    if st.sidebar.button("Get Inspirational Quote"):
        quotes = [
            "The only way to do great work is to love what you do. - Steve Jobs",
            "Believe you can and you're halfway there. - Theodore Roosevelt",
            "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
            "Strive not to be a success, but rather to be of value. - Albert Einstein",
            "The only limit to our realization of tomorrow will be our doubts of today. - Franklin D. Roosevelt",
            "Do not wait to strike till the iron is hot; but make it hot by striking. - William Butler Yeats",
            "What lies behind us and what lies before us are tiny matters compared to what lies within us. - Ralph Waldo Emerson",
            "Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston Churchill",
            "Life is what happens when you're busy making other plans. - John Lennon",
            "You miss 100% of the shots you don't take. - Wayne Gretzky",
            "The best way to predict the future is to create it. - Peter Drucker",
            "It is not the strongest of the species that survive, nor the most intelligent, but the one most responsive to change. - Charles Darwin",
            "Whether you think you can or you think you can't, you're right. - Henry Ford",
            "The only place where success comes before work is in the dictionary. - Vidal Sassoon",
            "Do what you can, with what you have, where you are. - Theodore Roosevelt",
            "The purpose of our lives is to be happy. - Dalai Lama",
            "Success usually comes to those who are too busy to be looking for it. - Henry David Thoreau",
            "Your time is limited, so don't waste it living someone else's life. - Steve Jobs",
            "Don't be afraid to give up the good to go for the great. - John D. Rockefeller",
            "I find that the harder I work, the more luck I seem to have. - Thomas Jefferson",
            "Success is not the key to happiness. Happiness is the key to success. - Albert Schweitzer",
            "It does not matter how slowly you go, as long as you do not stop. - Confucius",
            "If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success. - James Cameron",
            "Don't watch the clock; do what it does. Keep going. - Sam Levenson",
            "Hardships often prepare ordinary people for an extraordinary destiny. - C.S. Lewis",
            "Don't count the days, make the days count. - Muhammad Ali",
            "The best revenge is massive success. - Frank Sinatra",
            "The only impossible journey is the one you never begin. - Tony Robbins",
            "Act as if what you do makes a difference. It does. - William James",
            "You are never too old to set another goal or to dream a new dream. - C.S. Lewis",
            "If you're going through hell, keep going. - Winston Churchill",
            "Dream big and dare to fail. - Norman Vaughan",
            "In the middle of every difficulty lies opportunity. - Albert Einstein",
            "What we achieve inwardly will change outer reality. - Plutarch",
            "I have not failed. I've just found 10,000 ways that won't work. - Thomas Edison",
            "It always seems impossible until it's done. - Nelson Mandela",
            "The future depends on what you do today. - Mahatma Gandhi",
            "Don't wait. The time will never be just right. - Napoleon Hill",
            "Quality is not an act, it is a habit. - Aristotle",
            "Your life does not get better by chance, it gets better by change. - Jim Rohn",
            "The only thing standing between you and your goal is the story you keep telling yourself as to why you can't achieve it. - Jordan Belfort",
            "Challenges are what make life interesting; overcoming them is what makes life meaningful. - Joshua J. Marine",
            "Opportunities don't happen, you create them. - Chris Grosser",
            "I can't change the direction of the wind, but I can adjust my sails to always reach my destination. - Jimmy Dean",
            "Start where you are. Use what you have. Do what you can. - Arthur Ashe",
            "The secret of getting ahead is getting started. - Mark Twain",
            "You don’t have to be great to start, but you have to start to be great. - Zig Ziglar",
            "Keep your eyes on the stars, and your feet on the ground. - Theodore Roosevelt",
            "The only way to achieve the impossible is to believe it is possible. - Charles Kingsleigh"
        ]

        random_quote = random.choice(quotes)
        st.sidebar.success(random_quote)
        
    # Chat Export
    st.sidebar.markdown("---")
    if st.sidebar.button("Export Chat History"):
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        st.sidebar.download_button(
            label="Download Chat History",
            data=chat_history,
            file_name="ai_buddy_chat_history.txt",
            mime="text/plain"
        )
        
        st.sidebar.success("Chat history ready for download!")

    # Display achievements
    if 'achievements' in st.session_state and st.session_state.achievements:
        st.sidebar.markdown("---")
        st.sidebar.header("🏆 Achievements")
        for achievement in st.session_state.achievements:
            st.sidebar.success(f"Unlocked: {achievement}")

if __name__ == "__main__":
    main()