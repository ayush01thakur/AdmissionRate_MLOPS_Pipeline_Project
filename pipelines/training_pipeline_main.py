import pandas as pd
from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_df import clean_data
from steps.model_training import train_model
from steps.model_evaluation import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    df= ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mse, r2Score= evaluate_model(model, X_test, y_test)
