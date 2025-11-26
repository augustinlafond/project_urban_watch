"""
# MLflow Model Management Utilities

This module provides utilities for logging, saving, and loading machine learning models and their metrics with MLflow.

Main functionalities:

- **Model Logging**: Save trained `LogisticRegression`, `RandomForestClassifier`, or `XGBClassifier` models to MLflow, optionally registering them.
- **Results Logging**: Log hyperparameters and evaluation metrics to MLflow.
- **Run Wrapper**: Decorator (`mlflow_run`) to automatically log parameters, metrics, and models during a function run.
- **Model Loading**: Retrieve a trained model from MLflow given its name, type, and deployment stage.

"""


# Standard library
from colorama import Fore, Style

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Project-specific parameters
from urban_watch.params import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT


def save_model(model):
    if model is not None:
        if isinstance(model, LogisticRegression):
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="logistic_regression_model",
            )

        elif isinstance(model, RandomForestClassifier):
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="random_forest_model",
            )

        elif isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(
                xgb_model=model.get_booster(),
                artifact_path="model",
                registered_model_name="xgb_model",
            )
        else:
            raise ValueError(f"model not log")

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
            mlflow.xgboost.autolog()
            results = func(*args, **kwargs)
        print("✅ mlflow_run auto-log done")
        return results

    return wrapper


def load_model(model_name, model_type, stage="Production"):
    print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

    # Load model from MLflow
    model = None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        model_versions = client.get_latest_versions(name=model_name, stages=[stage])
        model_uri = model_versions[0].source
        assert model_uri is not None
    except:
        print(f"\n❌ No model found with name {model_name} in stage {stage}")
        return None

    if model_type in ["LogisticRegression", "RandomForest"]:
        model = mlflow.sklearn.load_model(model_uri=model_uri)
    elif model_type == "XGB":
        model = mlflow.xgboost.load_model(model_uri=model_uri)
    else:
        print(" Model type '{model_type}' not supported")
        return None

    print("✅ model loaded from mlflow")
    return model
