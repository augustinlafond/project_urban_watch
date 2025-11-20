import os
import pickle
from urban_watch.params import *
import mlflow
from mlflow.tracking import MlflowClient

def save_model(model, model_name='model'):

    if MODEL_TARGET == "mlflow":
        mlflow.     .log_model(model=model,
                               artifact_path=MLFLOW_MODEL_NAME,
                               registered_model_name=MLFLOW_MODEL_NAME)
        print("model logged to mlflow")
        return
