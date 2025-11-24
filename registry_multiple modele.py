import os
import pickle
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from urban_watch.params import *

def save_model(model):
    if MODEL_TARGET == "mlflow":
        if isinstance(model, LogisticRegression):
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="logistic_regression_model"
            )

        elif isinstance(model, RandomForestClassifier):
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="random_forest_model"
            )

        elif isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(
                xgb_model=model.get_booster(),
                artifact_path="model",
                registered_model_name="xgb_model"
            )
        else:
            raise ValueError(f"model not log")

        print("model logged to mlflow")
        return

def load_model(model_name, model_type, stage="Production"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()
    versions = client.get_latest_versions(name=model_name, stages=[stage])
    if len(versions)==0:
        raise ValueError(f"aucune versions trouvé {model_name} dans la stage{stage}")
    model_uri = versions[0].source
     # model_uri= f"models:/{model_name}/{stage}"


    if model_type == 'LogisticRegression' or model_type =='RandomForest':
        model= mlflow.sklearn.load_model(model_uri)
        #elif model_type == ['']:
            #return mlflow..load_model()
    elif model_type == "XGB":
        model= mlflow.xgboost.load_model(model_uri)
    else:
        raise ValueError(f"fail")
    return model

def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

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





    #     client = MlflowClient()

    #     try:
    #         model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
    #         model_uri = model_versions[0].source

    #         assert model_uri is not None
    #     except:
    #         print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")

    #         return None

    #     model = mlflow.tensorflow.load_model(model_uri=model_uri)

    #     print("✅ Model loaded from MLflow")
    #     # $CHA_END
    #     return model
    # else:
    #     return None
