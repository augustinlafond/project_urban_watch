import os
import pickle
from urban_watch.params import *
import mlflow
from mlflow.tracking import MlflowClient

def save_model(model, model_name='model'):

    if MODEL_TARGET == "mlflow":
        mlflow.sklearn.log_model(model, artifact_path=MLFLOW_MODEL_NAME)
        print("model logged to mlflow")
        return

    #ici la local save
    model_dir = os.path.join(LOCAL_REGISTRY_PATH, 'models')
    os.makedirs(model_dir)
