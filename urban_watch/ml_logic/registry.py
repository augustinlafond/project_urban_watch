import os
import pickle
from urban_watch.params import *
import mlflow
from mlflow.tracking import MlflowClient

def save_model(model, model_name='model'):

    if MODEL_TARGET == "mlflow":
        if "RandomForest" in str(type(model)):
            mlflow.sklearn.log_model(model=model,
                                artifact_path=MLFLOW_MODEL_NAME,
                                registered_model_name=MLFLOW_MODEL_NAME)

        elif "XGB" in str(type(model)):
            mlflow.xgboost.log_model(model=model,
                                artifact_path=MLFLOW_MODEL_NAME,
                                registered_model_name=MLFLOW_MODEL_NAME)

        print("model logged to mlflow")
        return

#params + metric

def save_params(params_dict, model_name=):
    for param_name, param_value in params_dict.item():
        mlflow.log_param(param_name, param_value)
    mlflow.log_param(le model name )
    mlflow.log_param(URI)

def save_metrics(metrics_dict, model_name=):
    essential_metrics = ['f1_score', 'roc_auc', 'recall', 'precision']


def save_all_to_mlflow(model, model_name, best_params, metrics_dict, X_test, y_test)
