import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer

class DataPreprocessing:
    
    # Simple helper to clean the messy TMDB columns
    def _extract_names(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'].replace(" ", "")) # Remove spaces so 'Sci Fi' becomes 'SciFi'
        return " ".join(L)

    def PreProcessData(self, data):
        # 1. Select only what we need
        data = data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        data.dropna(inplace=True)

        # 2. Clean the text columns (Turning lists into simple strings)
        data['genres'] = data['genres'].apply(self._extract_names)
        data['keywords'] = data['keywords'].apply(self._extract_names)
        data['cast'] = data['cast'].apply(self._extract_names)
        data['crew'] = data['crew'].apply(self._extract_names)

        # 3. Combine everything into one 'tags' column
        data['tags'] = data['overview'] + " " + data['genres'] + " " + \
                       data['keywords'] + " " + data['cast'] + " " + data['crew']
        
        # 4. Final Clean DataFrame
        new_df = data[['movie_id', 'title', 'tags']]
        new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

        # Note: We return the dataframe and a Vectorizer (replaces your 'preprocessor')
        cv = CountVectorizer(max_features=5000, stop_words='english')
        
        return new_df, cv