import streamlit as st
import googleapiclient.discovery
from dotenv import load_dotenv
from datetime import timedelta

# Load environment variables
load_dotenv()

# Set up YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = os.getenv('DEVELOPER_KEY')
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

def search_youtube(query, max_results=50):
    try:
        request = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=max_results,
            fields="items(id(videoId),snippet(title,description,thumbnails))"
        )
        response = request.execute()
        return response.get('items', [])
    except googleapiclient.errors.HttpError as e:
        st.error(f"An error occurred: {e}")
        return []

def get_video_details(video_id):
    try:
        request = youtube.videos().list(
            part="contentDetails,statistics",
            id=video_id,
            fields="items(contentDetails(duration),statistics(viewCount))"
        )
        response = request.execute()
        return response['items'][0] if response['items'] else None
    except googleapiclient.errors.HttpError as e:
        st.error(f"An error occurred while fetching video details: {e}")
        return None

def format_duration(duration):
    duration = duration.replace('PT', '')
    hours = 0
    minutes = 0
    seconds = 0
    if 'H' in duration:
        hours, duration = duration.split('H')
        hours = int(hours)
    if 'M' in duration:
        minutes, duration = duration.split('M')
        minutes = int(minutes)
    if 'S' in duration:
        seconds = int(duration.replace('S', ''))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def parse_duration(duration_str):
    parts = duration_str.split(':')
    if len(parts) == 3:
        return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
    elif len(parts) == 2:
        return timedelta(minutes=int(parts[0]), seconds=int(parts[1]))
    else:
        return timedelta(seconds=int(parts[0]))

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. Learning Assistant", page_icon="🕵️", layout="wide")
    st.sidebar.title("S.H.E.R.L.O.C.K.")
    st.sidebar.markdown("""
    **S**ystematic **H**olistic **E**ducational **R**esource for **L**earning and **O**ptimizing **C**ognitive **K**nowledge
    
    Enhance your cognitive abilities, memory techniques, and subject-specific knowledge with AI-powered personalized learning.
    """)
    
    query = st.sidebar.text_input("What would you like to learn about?", "")
    
    min_duration = st.sidebar.selectbox(
        "Minimum video duration",
        ["Any", "5:00", "10:00", "15:00", "30:00", "45:00", "1:00:00"],
        index=0
    )
    
    search_button = st.sidebar.button("Search for Learning Resources")
    
    st.title("Learning Resources")
    
    if search_button and query:
        with st.spinner("Searching for the best learning resources..."):
            results = search_youtube(query)
            
        if results:
            filtered_results = []
            for item in results:
                video_id = item['id']['videoId']
                video_details = get_video_details(video_id)
                
                if video_details:
                    duration = video_details['contentDetails']['duration']
                    formatted_duration = format_duration(duration)
                    views = int(video_details['statistics']['viewCount'])
                    
                    if min_duration == "Any" or parse_duration(formatted_duration) >= parse_duration(min_duration):
                        filtered_results.append((item, formatted_duration, views))
                
            if filtered_results:
                for item, duration, views in filtered_results:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(item['snippet']['thumbnails']['medium']['url'], use_column_width=True)
                    with col2:
                        st.markdown(f"### [{item['snippet']['title']}](https://www.youtube.com/watch?v={item['id']['videoId']})")
                        st.markdown(f"**Duration:** {duration} | **Views:** {views:,}")
                        st.markdown(item['snippet']['description'])
                    
                    st.markdown("---")
            else:
                st.warning("No results found matching your duration criteria. Try adjusting the minimum duration or search query.")
        else:
            st.warning("No results found. Please try a different search query.")

if __name__ == "__main__":
    main()