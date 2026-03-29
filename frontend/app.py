import streamlit as st
import requests
import pandas as pd

# 1. Load the movie list for the dropdown
# We need this so the user can see the available titles
movies_df = pd.read_csv("models/movie_list.csv")
movie_titles = movies_df['title'].values

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Pick a movie you liked, and we'll suggest 5 similar ones!")

# 2. Search Box with Auto-complete
selected_movie = st.selectbox(
    "Type or select a movie from the list:",
    movie_titles
)

# 3. Prediction Button
if st.button("Recommend"):
    # We send a POST request to your FastAPI backend
    payload = {"movie_name": selected_movie}
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            recommendations = result['recommendations']
            
            st.subheader(f"If you liked '{selected_movie}', you might also enjoy:")
            
            # Displaying recommendations in a nice list
            for i, movie in enumerate(recommendations):
                st.success(f"{i+1}. {movie}")
        else:
            st.error("Error connecting to the Backend API.")
            
    except Exception as e:
        st.error(f"Could not connect to the server. Make sure FastAPI is running! Error: {e}")

# Footer
st.markdown("---")
st.caption("Powered by KNN and TMDB Dataset")