import logging
import pandas as pd
import numpy as np
from abc import abstractmethod, ABC
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):

    @abstractmethod
    def calculated_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ 
        calculates the model's performance
        Args:
            y_true: actual values
            y_pred: predicted values
        """
        #  since we just implementing this for the linear Regression model and not for any classification or 
        # other model so will just go with basic metrics like mse, r2 score

class MSE(Evaluation):
    """
    Evaluation Strategy is Mean Squared Error
    """

    def calculated_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error occured while evaluation MSE: {}".format(e))
            raise e
        
class R2score(Evaluation):
    """
    Evaluation Strategy is R2_Score
    """
    def calculated_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating R2_Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2_Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error occured while evaluation r2_score: {}".format(e))
            raise e


