from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainPipeline:

    def run_pipeline(self):

        # Step 1 : Data Ingestion
        ingestion = DataIngestion()
        # Make sure to update these paths to where your TMDB files are located
        data = ingestion.Ingest_data(
            "data/tmdb_5000_movies.csv", 
            "data/tmdb_5000_credits.csv"
        )

        print("Data Ingestion Completed")

        # Step 2 : Data Preprocessing
        preprocessing = DataPreprocessing()
        # This returns our cleaned dataframe and the CountVectorizer
        new_df, preprocessor = preprocessing.PreProcessData(data)

        print("Data Preprocessing Completed")

        # Step 3 : Model Training
        trainer = ModelTrainer()
        trainer.TrainModel(new_df, preprocessor)

        print("Model Training Completed")


if __name__ == "__main__": 
    # Calling the pipeline to run ingestion -> preprocessing -> training
    pipeline = TrainPipeline()
    pipeline.run_pipeline()