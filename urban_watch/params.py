import os
import numpy as np

##################  VARIABLES  ##################
SH_CLIENT_ID = os.environ.get("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.environ.get("SH_CLIENT_SECRET")


BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_PROJECT=os.environ.get("GCP_PROJECT")


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

#le chemin est pas fait

IMG_HEIGHT = 300
IMG_WIDTH = 300
N_BANDS = 13

DTYPE_PROCESSED = np.float32
