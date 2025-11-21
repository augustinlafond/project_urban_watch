import os
import pickle
from urban_watch.params import *
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient
from urban_watch.params import *
def save_model(model, model_name='model'):

    if MODEL_TARGET == "mlflow":
        if "RandomForest" in str(type(model)):
            mlflow.sklearn.log_model(model=model,
                                artifact_path=MLFLOW_MODEL_NAME,
                                registered_model_name=MLFLOW_MODEL_NAME)

        # elif "XGB" in str(type(model)):
        #     mlflow.xgboost.log_model(model=model,
        #                         artifact_path=MLFLOW_MODEL_NAME,
        #                         registered_model_name=MLFLOW_MODEL_NAME)

        print("model logged to mlflow")
        return

def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ Model {MLFLOW_MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

    return None

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
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper


#params + metric

# def save_params(params_dict, model_name=):
#     for param_name, param_value in params_dict.item():
#         mlflow.log_param(param_name, param_value)
#     mlflow.log_param(le model name )
#     mlflow.log_param(URI)

# def save_metrics(metrics_dict, model_name=):
#     essential_metrics = ['f1_score', 'roc_auc', 'recall', 'precision']


# def save_all_to_mlflow(model, model_name, best_params, metrics_dict, X_test, y_test)
