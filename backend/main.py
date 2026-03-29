from fastapi import FastAPI
from src.pipelines.prediction_pipeline import PredictionPipeline
from pydantic import BaseModel

app = FastAPI()

# 1. Initialize the Pipeline (This loads the .pkl files into memory once)
pipeline = PredictionPipeline()

# 2. Schema for the request (Similar to your LoanData schema)
class MovieRequest(BaseModel):
    movie_name: str

@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running!"}

@app.post("/predict")
def predict(request: MovieRequest):
    """
    Endpoint to receive a movie name and return 5 similar movies.
    """
    # Use the pipeline logic we discussed
    recommendations = pipeline.predict(request.movie_name)
    
    return {
        "movie_requested": request.movie_name,
        "recommendations": recommendations
    }