import logging
from typing import Union
from abc import ABC, abstractmethod
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from sklearn.model_selection import train_test_split


# parent class (DataStrategy) which will be followed for the preprocessing, cleaning, and splitting data
class DataStrategy(ABC):
    """
    Abstract class defining the handling data strategy
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        # as child classes will inherit this handle data and will define it there
        pass

class DataPreProcessing(DataStrategy):
    """
    Preprocessing the data 
    """

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This handle_data function will preprocess the data
        """

        try:
            # deleting the rows with null values ; however we do not have any in the current dataset
            # df= df.dropna()

            # dropping the research column because that is not that relevant
            df.drop(columns=['Research'])

            # df= df.select_dtypes(include=['Int64', 'float32'])
            return df
        
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e
        

        
class DataDivide(DataStrategy):
    """
    Strategy for dividing the data into train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides data into train and test 
        Args:
            data: pd.DataFrame
        Returns:
            Union[pd.DataFrame, pd.Series]
        """
        try:
            X= data.drop(columns = ['Chance_of_Admission'], axis=1)
            y= data['Chance_of_Admission']

            X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.80, random_state=42)


            # scaling the data:
            # for now i am leaving the scaling operation; however I'll add this later;

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error("Error in dividing the data: {}".format(e))
            raise e
        
    
class DataCleaning:
    """
    class for cleaning data which process the data nd divides
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data= data
        self.strategy= strategy

    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        handle data of DataCleaning class
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
        
