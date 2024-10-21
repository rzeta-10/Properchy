import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd

class DataIngester(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest the data from the file and return a DataFrame

        Parameters:
        file_path (str): The path to the file to ingest

        Returns:
        None : The ingested data as a DataFrame        
        """
        pass

class ZipIngester(DataIngester):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest the data from the zip file and return a DataFrame

        Parameters:
        file_path (str): The path to the zip file to ingest

        Returns:
        df (pd.DataFrame): The ingested data as a DataFrame
        """
        if not zipfile.is_zipfile(file_path):
            raise ValueError("File is not a zip file")
        
        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall("extracted_data")
        
        extracted_files = os.listdir("extracted_data")
        csv_files = [file for file in extracted_files if file.endswith(".csv")]

        if len(csv_files) == 0:
            raise ValueError("No csv files found in the zip file")
        elif len(csv_files) > 1:
            raise ValueError("Multiple csv files found in the zip file")
        
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)
        return df
    
class DataIngesterFactory:
    @staticmethod
    def create_ingester(file_path: str) -> DataIngester:
        """
        Ingest the data from the file and return a DataFrame

        Parameters:
        file_path (str): The path to the file to ingest

        Returns:
        Calls the appropriate ingester based on the file type
        """
        if zipfile.is_zipfile(file_path):
            return ZipIngester()
        else:
            raise ValueError("Unsupported file type")
        
    
if __name__ == "__main__":

    file_path = os.path.join("data", "archive.zip")

    file_extension = os.path.splitext(file_path)[1]

    ingester = DataIngesterFactory.create_ingester(file_path)

    df = ingester.ingest(file_path)

    print(df.head())

    pass