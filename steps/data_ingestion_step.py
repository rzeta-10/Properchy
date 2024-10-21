import pandas as pd
from src.ingest_data import DataIngesterFactory
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    Ingest the data from the file and return a DataFrame

    Parameters:
    file_path (str): The path to the file to ingest

    Returns:
    df (pd.DataFrame): The ingested data as a DataFrame
    """

    file_extension = '.zip'

    data_ingestor = DataIngesterFactory.create_ingester(file_extension)

    df = data_ingestor.ingest(file_path)

    return df