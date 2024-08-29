import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Scopus API key
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY')

def search_scopus(query, start_year, end_year, max_results=50):
    base_url = "https://api.elsevier.com/content/search/scopus"
    
    params = {
        "query": query,
        "date": f"{start_year}-{end_year}",
        "count": max_results,
        "sort": "citedby-count desc",
        "field": "title,author,year,publicationName,description,citedby-count,doi,eid"
    }
    
    headers = {
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()["search-results"]["entry"]
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while searching Scopus: {e}")
        return []

def format_authors(author_info):
    if isinstance(author_info, list):
        return ", ".join([author.get("authname", "") for author in author_info])
    elif isinstance(author_info, dict):
        return author_info.get("authname", "")
    else:
        return "N/A"

def safe_get(dictionary, keys, default="N/A"):
    for key in keys:
        if isinstance(dictionary, dict) and key in dictionary:
            dictionary = dictionary[key]
        else:
            return default
    return dictionary

def get_paper_link(paper):
    doi = safe_get(paper, ["prism:doi"])
    if doi != "N/A":
        return f"https://doi.org/{doi}"
    eid = safe_get(paper, ["eid"])
    if eid != "N/A":
        return f"https://www.scopus.com/record/display.uri?eid={eid}&origin=resultslist"
    return "#"

def main():
    st.set_page_config(page_title="S.H.E.R.L.O.C.K. Research Assistant", page_icon="🔬", layout="wide")
    
    st.sidebar.title("S.H.E.R.L.O.C.K.")
    st.sidebar.markdown("""
    **S**ystematic **H**olistic **E**ducational **R**esource for **L**iterature and **O**ptimizing **C**ognitive **K**nowledge
    
    Enhance your research capabilities with AI-powered literature search and analysis.
    """)
    
    query = st.sidebar.text_input("What topic would you like to research?", "")
    
    current_year = datetime.now().year
    start_year, end_year = st.sidebar.slider(
        "Publication Year Range",
        min_value=1900,
        max_value=current_year,
        value=(current_year-5, current_year)
    )
    
    max_results = st.sidebar.slider("Maximum number of results", 10, 100, 50)
    
    search_button = st.sidebar.button("Search for Research Papers")
    
    st.title("Research Papers and Articles")
    
    if search_button and query:
        with st.spinner("Searching for the most relevant research papers..."):
            results = search_scopus(query, start_year, end_year, max_results)
        
        if results:
            papers = []
            for paper in results:
                papers.append({
                    "Title": safe_get(paper, ["dc:title"]),
                    "Authors": format_authors(safe_get(paper, ["author"])),
                    "Year": safe_get(paper, ["prism:coverDate"])[:4],
                    "Journal": safe_get(paper, ["prism:publicationName"]),
                    "Abstract": safe_get(paper, ["dc:description"]),
                    "Citations": safe_get(paper, ["citedby-count"], "0"),
                    "Link": get_paper_link(paper)
                })
            
            df = pd.DataFrame(papers)
            
            st.markdown(f"### Found {len(results)} papers on '{query}'")
            
            for _, paper in df.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### [{paper['Title']}]({paper['Link']})")
                        st.markdown(f"**Authors:** {paper['Authors']}")
                        st.markdown(f"**Published in:** {paper['Journal']} ({paper['Year']})")
                        st.markdown(f"**Abstract:** {paper['Abstract']}")
                    with col2:
                        st.metric("Citations", paper["Citations"])
                    
                    st.markdown("---")
            
            # Download results as CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name=f"{query.replace(' ', '_')}_research_papers.csv",
                mime="text/csv",
            )
        else:
            st.warning("No results found. Please try a different search query or adjust the year range.")

if __name__ == "__main__":
    main()