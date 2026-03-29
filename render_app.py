import streamlit as st
import pandas as pd
# Import your logic directly
from src.pipelines.prediction_pipeline import PredictionPipeline

# Initialize the pipeline once (Caching saves RAM!)
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()

# Load movie list for the dropdown
movies_df = pd.read_csv("models/movie_list.csv")
movie_titles = movies_df['title'].values

st.title("🎬 Movie Recommender")

selected_movie = st.selectbox("Pick a movie:", movie_titles)

if st.button("Recommend"):
    # Call the logic directly without an API request
    recommendations = pipeline.predict(selected_movie)
    
    for i, movie in enumerate(recommendations):
        st.success(f"{i+1}. {movie}")