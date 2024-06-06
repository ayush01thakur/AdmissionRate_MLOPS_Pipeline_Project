import logging

import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivide, DataPreProcessing
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Cleans the data and divides it into train and test

    Args: pandas DataFrame of raw data
    Returns :
        X_train: Training Data
        X_test: Testing Data
        y_train: Training values
        y_test: Testing values
    """

    try:
        process_strategy = DataPreProcessing()
        data_cleaning= DataCleaning(df, process_strategy)
        preProcessed_data= data_cleaning.handle_data()

        divide_strategy = DataDivide()
        data_cleaning = DataCleaning(preProcessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed, Returned X_train, X_test, y_train, y_test")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error("Error while cleaning data: {}".format(e))
        raise e
    
