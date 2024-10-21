from steps.data_ingestion_step import data_ingestion_step

from dotenv import load_dotenv
import os
from zenml import Model, pipeline, step

load_dotenv()

@pipeline(
    model = Model(
        name = 'house_price_predictor',
    ),
)

def ml_pipeline():
    file_path = os.getenv('ZIP_FILE_PATH')

    raw_data = data_ingestion_step(
        file_path=file_path
    )

    