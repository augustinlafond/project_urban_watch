import os
import numpy as np

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

#le chemin est pas fait

IMG_HEIGHT = 300
IMG_WIDTH = 300
N_BANDS = 13

DTYPE_PROCESSED = np.float32

if MODEL_TARGET not in ["mlflow"]:
    raise NameError("MODEL_TARGET must be in ['mlflow']")

