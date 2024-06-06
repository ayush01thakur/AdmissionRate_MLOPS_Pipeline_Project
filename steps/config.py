from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """ Model Configurations: model name"""
    model_name: str= "LinearRegression"