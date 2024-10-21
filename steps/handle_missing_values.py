import logging
from abc import ABC, abstractmethod

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MissingValuesHandler(ABC):
    @abstractmethod
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataframe and return a new dataframe

        Parameters:
        df (pd.DataFrame): The dataframe with missing values

        Returns:
        df (pd.DataFrame): The dataframe with missing values handled
        """
        pass

class DropMissingValues(MissingValuesHandler):
    def __init__(self, axis=0, thresh=None):
        """
        Initialize the DropMissingValues handler
        
        Parameters:
        axis (int): The axis along which to drop missing values (0 for rows, 1 for columns)
        thresh (int): The minimum number of non-missing values required to keep the axis
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop missing values from the dataframe and return a new dataframe
        
        Parameters:
        df (pd.DataFrame): The dataframe with missing values
        
        Returns:
        df (pd.DataFrame): The dataframe with missing values dropped
        """
        logging.info(f"Dropping missing values along axis {self.axis} with threshold {self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info(f"Number of rows dropped: {df.shape[0] - df_cleaned.shape[0]}")
        return df_cleaned