import logging
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin

class Model(ABC):

    """ Abstract class for all different models"""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Trains on the training data
        Args:
            X_train: training data
            y_train: training labels
        """

        pass

class LinearRegressionModel(Model):

    """ Linear Regression Model """
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Trains on the linear regression model

        """
        try:
            regModel= LinearRegression()
            regModel.fit(X_train, y_train)
            logging.info("Linear Model creation Complete")
            return regModel
        
        except Exception as e:
            logging.error("Error occured while creating Linear Regression Model")
            raise e
        


