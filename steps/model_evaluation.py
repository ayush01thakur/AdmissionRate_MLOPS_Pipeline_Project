import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
import numpy as np
from zenml import step

from src.evaluation import R2score, MSE
from sklearn.base import RegressorMixin 

@step
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, 
                   y_test: pd.DataFrame)-> Tuple[Annotated[float, "r2_score"], Annotated[float, "mse"]]:
    """
    Evaluate the trained model on the test data and shows the evaluation scores
    Args: 
        model, X_test, and Y_test
    Returns:
        None for now
    """
    try:
        y_pred = model.predict(X_test)
        mse_class= MSE()
        mse= mse_class.calculated_scores(y_test, y_pred)

        r2_class= R2score()
        r2Score= r2_class.calculated_scores(y_test, y_pred)

        return r2Score, mse
    
    except Exception as e:
        logging.error("Error while evaluating the model: {}".format(e))
        raise e
    

    