import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

class ModelTrainer:

    def TrainModel(self, new_df, preprocessor):

        # 1. Unlike Loan, we don't split for Recommenders 
        # (We need the whole matrix to find neighbors)
        X = new_df['tags']

        # 2. Define pipeline (Just like your Loan project)
        # We put CountVectorizer and KNN in one pipeline
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", NearestNeighbors(metric='cosine', algorithm='brute'))
            ]
        )

        print("Starting Movie Recommendation Model Training...")

        # 3. Fitting the pipeline
        # Note: KNN 'fit' just stores the vectors for later distance calculation
        model_pipeline.fit(X)

        # 4. Save artifacts (Equivalent to your Step 11 & 12)
        os.makedirs("models", exist_ok=True)
        
        # We save the dataframe because we need titles for the UI
        new_df.to_csv("models/movie_list.csv", index=False)
        
        # Save the full pipeline
        joblib.dump(model_pipeline, "models/model.pkl")

        print("Movie Recommender model saved successfully!")

        return model_pipeline