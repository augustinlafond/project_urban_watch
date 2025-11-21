import os
import pickle
from urban_watch.params import *
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient
from urban_watch.params import *
from colorama import Fore, Style


def save_model(model):
    mlflow.sklearn.log_model(sk_model=model,
                                artifact_path="model",
                                registered_model_name=MLFLOW_MODEL_NAME)
    print("model logged to mlflow")
    return None

def save_results(params: dict, metrics: dict):
    if params is not None:
        mlflow.log_params(params)
    if metrics is not None:
        mlflow.log_metrics(metrics)
    print("✅ Results saved on mlflow")



def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with sklearn auto-logging
    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.sklearn.autolog()
            results = func(*args, **kwargs)
        print("✅ mlflow_run auto-log done")
        return results
    return wrapper


def load_model(stage="Production"):
    print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

    # Load model from MLflow
    model = None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
        model_uri = model_versions[0].source
        assert model_uri is not None
    except:
        print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")
        return None

    model = mlflow.sklearn.load_model(model_uri=model_uri)
    print("✅ model loaded from mlflow")
    return model
