import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import pandas as pd
import time
import os
from datetime import datetime
import random
import re
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = os.getenv('AI71_API_KEY')

# Initialize the Falcon model
chat = ChatOpenAI(
    model="tiiuae/falcon-180B-chat",
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
    temperature=0.7,
)

class RoadmapStep(BaseModel):
    title: str
    description: str
    resources: List[Dict[str, str]] = Field(default_factory=list)
    estimated_time: str
    how_to_use: Optional[str] = None

class Roadmap(BaseModel):
    steps: Dict[str, RoadmapStep] = Field(default_factory=dict)

def clean_json(content):
    # Remove any leading or trailing whitespace
    content = content.strip()
    
    # Ensure the content starts and ends with curly braces
    if not content.startswith('{'):
        content = '{' + content
    if not content.endswith('}'):
        content = content + '}'
    
    # Remove any newline characters and extra spaces
    content = ' '.join(content.split())
    
    # Escape any unescaped double quotes within string values
    content = re.sub(r'(?<!\\)"(?=(?:(?:[^"]*"){2})*[^"]*$)', r'\"', content)
    
    return content

def ensure_valid_json(content):
    # First, apply our existing cleaning function
    content = clean_json(content)
    
    # Use regex to find and fix unquoted property names
    pattern = r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
    content = re.sub(pattern, r'\1 "\2":', content)
    
    # Replace single quotes with double quotes
    content = content.replace("'", '"')
    
    # Attempt to parse the JSON to catch any remaining issues
    try:
        json_obj = json.loads(content)
        return json.dumps(json_obj)  # Return a properly formatted JSON string
    except json.JSONDecodeError as e:
        # If we still can't parse it, log the error and return None
        logger.error(f"Failed to parse JSON after cleaning: {str(e)}")
        logger.debug(f"Problematic JSON: {content}")
        return None

def generate_roadmap(topic):
    levels = [
        "knowledge",
        "comprehension",
        "application",
        "analysis",
        "synthesis",
        "evaluation"
    ]

    roadmap = Roadmap()

    for level in levels:
        try:
            logger.info(f"Generating roadmap step for topic: {topic} at {level} level")
            step = generate_simplified_step(topic, level, chat)
            roadmap.steps[level] = step
            logger.info(f"Added step for {level} level")
            
        except Exception as e:
            logger.error(f"Error in generate_roadmap for {level}: {str(e)}")
            step = create_fallback_step(topic, level, chat)
            roadmap.steps[level] = step

    logger.info("Roadmap generation complete")
    return roadmap

def generate_diverse_resources(topic, level):
    encoded_topic = topic.replace(' ', '+')
    encoded_level = level.replace(' ', '+')
    
    resource_templates = [
        {"title": "Wikipedia", "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"},
        {"title": "YouTube Overview", "url": f"https://www.youtube.com/results?search_query={encoded_topic}+{encoded_level}"},
        {"title": "Coursera Courses", "url": f"https://www.coursera.org/search?query={encoded_topic}"},
        {"title": "edX Courses", "url": f"https://www.edx.org/search?q={encoded_topic}"},
        {"title": "Brilliant", "url": f"https://brilliant.org/search/?q={encoded_topic}"},
        {"title": "Google Scholar", "url": f"https://scholar.google.com/scholar?q={encoded_topic}"},
        {"title": "MIT OpenCourseWare", "url": f"https://ocw.mit.edu/search/?q={encoded_topic}"},
        {"title": "Khan Academy", "url": f"https://www.khanacademy.org/search?query={encoded_topic}"},
        {"title": "TED Talks", "url": f"https://www.ted.com/search?q={encoded_topic}"},
        {"title": "arXiv Papers", "url": f"https://arxiv.org/search/?query={encoded_topic}&searchtype=all"},
        {"title": "ResearchGate", "url": f"https://www.researchgate.net/search/publication?q={encoded_topic}"},
        {"title": "Academic Earth", "url": f"https://academicearth.org/search/?q={encoded_topic}"},
    ]
    
    # Randomly select 5-7 resources
    num_resources = random.randint(5, 7)
    selected_resources = random.sample(resource_templates, num_resources)
    
    return selected_resources

def create_fallback_step(topic, level, chat):
    def generate_component(prompt, default_value):
        try:
            response = chat.invoke([{"role": "system", "content": prompt}])
            return response.content.strip() or default_value
        except Exception as e:
            logger.error(f"Error generating component: {str(e)}")
            return default_value

    # Generate title
    title_prompt = f"Create a concise title (max 10 words) for a study step about {topic} at the {level} level of Bloom's Taxonomy."
    default_title = f"{level.capitalize()} Step for {topic}"
    title = generate_component(title_prompt, default_title)

    # Generate description
    description_prompt = f"""Write a detailed description (500-700 words) for a study step about {topic} at the {level} level of Bloom's Taxonomy. 
    Explain what this step entails, how the user should approach it, and why it's important for mastering the topic at this level. 
    The description should be specific to {topic} and not a generic explanation of the Bloom's Taxonomy level."""
    default_description = f"In this step, you will focus on {topic} at the {level} level. This involves understanding key concepts and theories related to {topic}. Engage with the provided resources to build a strong foundation."
    description = generate_component(description_prompt, default_description)

    # Generate estimated time
    time_prompt = f"Estimate the time needed to complete a study step about {topic} at the {level} level of Bloom's Taxonomy. Provide the answer in a format like '3-4 days' or '1-2 weeks'."
    default_time = "3-4 days"
    estimated_time = generate_component(time_prompt, default_time)

    # Generate how to use
    how_to_use_prompt = f"""Write a paragraph (100-150 words) on how to effectively use the {level} level of Bloom's Taxonomy when studying {topic}. 
    Include tips and strategies specific to {topic} at this {level} level."""
    default_how_to_use = f"Explore the provided resources and take notes on key concepts related to {topic}. Practice explaining these concepts in your own words to reinforce your understanding at the {level} level."
    how_to_use = generate_component(how_to_use_prompt, default_how_to_use)

    return RoadmapStep(
        title=title,
        description=description,
        resources=generate_diverse_resources(topic, level),
        estimated_time=estimated_time,
        how_to_use=how_to_use
    )

def create_interactive_graph(roadmap):
    G = nx.DiGraph()
    color_map = {
        'Knowledge': '#FF6B6B',
        'Comprehension': '#4ECDC4',
        'Application': '#45B7D1',
        'Analysis': '#FFA07A',
        'Synthesis': '#98D8C8',
        'Evaluation': '#F9D56E'
    }
    
    for i, (level, step) in enumerate(roadmap.steps.items()):
        G.add_node(step.title, level=level.capitalize(), pos=(i, -i))
    
    pos = nx.get_node_attributes(G, 'pos')
    
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=[],
            size=30,
            line_width=2
        ),
        text=[],
        textposition="top center"
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_info = f"{node}<br>Level: {G.nodes[node]['level']}"
        node_trace['text'] += (node_info,)
        node_trace['marker']['color'] += (color_map.get(G.nodes[node]['level'], '#CCCCCC'),)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Study Roadmap',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    ))
    
    # Add a color legend
    for level, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            showlegend=True,
            name=level
        ))
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    return fig

def get_user_progress(roadmap):
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = {}
    
    for level, step in roadmap.steps.items():
        if step.title not in st.session_state.user_progress:
            st.session_state.user_progress[step.title] = 0
    
    return st.session_state.user_progress

def update_user_progress(step_title, progress):
    st.session_state.user_progress[step_title] = progress

def calculate_overall_progress(progress_dict):
    if not progress_dict:
        return 0
    total_steps = len(progress_dict)
    completed_steps = sum(1 for progress in progress_dict.values() if progress == 100)
    return (completed_steps / total_steps) * 100

def generate_simplified_step(topic, level, chat):
    prompt = f"""Create a detailed study step for the topic: {topic} at the {level} level of Bloom's Taxonomy.
    
    Provide:
    1. A descriptive title (max 10 words)
    2. A detailed description (500-700 words) explaining what this step entails, how the user should approach it, and why it's important for mastering the topic at this level. The description should be specific to {topic} and not a generic explanation of the Bloom's Taxonomy level.
    3. Estimated time for completion (e.g., 3-4 days, 1-2 weeks, etc.)
    4. A paragraph (100-150 words) on how to use this level effectively, including tips and strategies specific to {topic} at this {level} level

    Format your response as a valid JSON object with the following structure:
    {{
        "title": "Step title",
        "description": "Step description",
        "estimated_time": "Estimated time",
        "how_to_use": "Paragraph on how to use this level effectively"
    }}
    """
    
    try:
        response = chat.invoke([{"role": "system", "content": prompt}])
        valid_json = ensure_valid_json(response.content)
        if valid_json is None:
            raise ValueError("Failed to create valid JSON")
        
        step_dict = json.loads(valid_json)
        
        # Generate diverse resources
        resources = generate_diverse_resources(topic, level)
        
        return RoadmapStep(
            title=step_dict["title"],
            description=step_dict["description"],
            resources=resources,
            estimated_time=step_dict["estimated_time"],
            how_to_use=step_dict["how_to_use"]
        )
    except Exception as e:
        logger.error(f"Error in generate_simplified_step for {level}: {str(e)}")
        return create_fallback_step(topic, level, chat)



def display_step(step, level, user_progress):
    with st.expander(f"{level.capitalize()}: {step.title}"):
        st.write(f"**Description:** {step.description}")
        st.write(f"**Estimated Time:** {step.estimated_time}")
        st.write("**Resources:**")
        for resource in step.resources:
            st.markdown(f"- [{resource['title']}]({resource['url']})")
            if 'contribution' in resource:
                st.write(f"  *{resource['contribution']}*")
        
        # Check if how_to_use exists before displaying it
        if step.how_to_use:
            st.write("**How to use this level effectively:**")
            st.write(step.how_to_use)
        
        progress = st.slider(f"Progress for {step.title}", 0, 100, user_progress.get(step.title, 0), key=f"progress_{level}")
        update_user_progress(step.title, progress)

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. Study Roadmap Generator", layout="wide")

    # Custom CSS for dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .streamlit-expanderHeader {
            background-color: #2E2E2E;
            color: #FFFFFF;
        }
        .streamlit-expanderContent {
            background-color: #2E2E2E;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 S.H.E.R.L.O.C.K. Study Roadmap Generator")
    st.write("Generate a comprehensive study roadmap based on first principles for any topic.")

    # Sidebar
    with st.sidebar:
        st.image("https://placekitten.com/300/200", caption="S.H.E.R.L.O.C.K.", use_column_width=True)
        st.markdown("""
        ## About S.H.E.R.L.O.C.K.
        **S**tudy **H**elper for **E**fficient **R**oadmaps and **L**earning **O**ptimization using **C**omprehensive **K**nowledge

        S.H.E.R.L.O.C.K. is your AI-powered study companion, designed to create personalized learning roadmaps for any topic. It breaks down complex subjects into manageable steps, ensuring a comprehensive understanding from fundamentals to advanced concepts.
        """)

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

    topic = st.text_input("📚 Enter the topic you want to master:")
    
    if st.button("🚀 Generate Roadmap"):
        if topic:
            with st.spinner("🧠 Generating your personalized study roadmap..."):
                try:
                    logger.info(f"Starting roadmap generation for topic: {topic}")
                    roadmap = generate_roadmap(topic)
                    if roadmap and roadmap.steps:
                        logger.info("Roadmap generated successfully")
                        st.session_state.current_roadmap = roadmap
                        st.session_state.current_topic = topic
                        st.success("Roadmap generated successfully!")
                    else:
                        logger.warning("Generated roadmap is empty or invalid")
                        st.error("Failed to generate a valid roadmap. Please try again with a different topic.")
                except Exception as e:
                    logger.error(f"Error during roadmap generation: {str(e)}", exc_info=True)
                    st.error(f"An error occurred while generating the roadmap: {str(e)}")
    
    if 'current_roadmap' in st.session_state:
        st.subheader(f"📊 Study Roadmap for: {st.session_state.current_topic}")
        
        roadmap = st.session_state.current_roadmap
        fig = create_interactive_graph(roadmap)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        user_progress = get_user_progress(roadmap)
        
        levels_description = {
            "knowledge": "Understanding and remembering basic facts and concepts",
            "comprehension": "Grasping the meaning and interpreting information",
            "application": "Using knowledge in new situations",
            "analysis": "Breaking information into parts and examining relationships",
            "synthesis": "Combining elements to form a new whole",
            "evaluation": "Making judgments based on criteria and standards"
        }
        
        for level, step in roadmap.steps.items():
            st.header(f"{level.capitalize()} Level")
            st.write(f"**Description:** {levels_description[level]}")
            st.write("**How to master this level:**")
            st.write(f"To master the {level} level, focus on {levels_description[level].lower()}. Engage with the resources provided, practice applying the concepts, and gradually build your understanding. Remember that mastery at this level is crucial before moving to the next.")
            display_step(step, level, user_progress)
        
        overall_progress = calculate_overall_progress(user_progress)
        st.progress(overall_progress / 100)
        st.write(f"Overall progress: {overall_progress:.2f}%")
        
        roadmap_json = json.dumps(roadmap.dict(), indent=2)
        st.download_button(
            label="📥 Download Roadmap as JSON",
            data=roadmap_json,
            file_name="study_roadmap.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()