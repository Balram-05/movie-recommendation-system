import pandas as pd
import os

class DataIngestion:
    def Ingest_data(self, movies_path, credits_path):
        # 1. Load both CSVs
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)

        # 2. Merge on 'title'
        data = movies.merge(credits, on='title')
        
        print(f"Data Ingestion Successful: {data.shape[0]} movies loaded.")
        return data