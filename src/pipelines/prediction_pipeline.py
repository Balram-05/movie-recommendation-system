import joblib
import pandas as pd
import numpy as np

class PredictionPipeline:
    def __init__(self):
        # 1. Load the artifacts we saved during training
        self.model_pipeline = joblib.load("models/model.pkl")
        self.movie_list = pd.read_csv("models/movie_list.csv")

    def predict(self, movie_name):
        try:
            # 2. Find the index of the movie the user typed
            # Example: If user types 'Avatar', find which row that is
            movie_index = self.movie_list[self.movie_list['title'] == movie_name].index[0]
            
            # 3. Get the "Tags" for that movie
            movie_tags = self.movie_list.iloc[movie_index]['tags']

            # 4. Use the model to find the 5 nearest neighbors
            # The model returns 'distances' and 'indices' of similar movies
            distances, indices = self.model_pipeline.named_steps['model'].kneighbors(
                self.model_pipeline.named_steps['preprocessor'].transform([movie_tags]),
                n_neighbors=6 # 6 because the first one is the movie itself
            )

            # 5. Convert indices back to Movie Titles
            recommendations = []
            for i in indices[0][1:]: # Start from 1 to skip the input movie
                recommendations.append(self.movie_list.iloc[i]['title'])

            return recommendations

        except Exception as e:
            return ["Movie not found in database! Please check spelling."]