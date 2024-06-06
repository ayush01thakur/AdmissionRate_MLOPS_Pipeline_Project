import logging

import pandas as pd
from zenml import step

from sklearn.base import RegressorMixin
from src.model_development import LinearRegressionModel
from .config import ModelNameConfig

@step
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, config: ModelNameConfig) -> RegressorMixin:
    """
    train the model 
    Args:
        X_train: Training Data
        X_test: Testing Data
        y_train: Training values
        y_test: Testing values
        config: model_name
    Returns:
        None for now.
    """

    try:
        model = None
        if config.model_name== "LinearRegression":
            model= LinearRegressionModel()
            trained_model= model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supporting")
        
    except Exception as e:
        logging.error("Error occured while training {} model".format(config.model_name))
        raise e
    
    