"""
Main orchestration script for the Urban Watch project.

This file ONLY coordinates the high-level pipeline steps:
- Preprocessing (download tiles, masking, indices, normalization)
- Feature extraction
- Model training (baseline / medium / advanced)
- Evaluation (soon)
- Prediction (later)

No detailed logic lives here.
"""

from sentinelhub import SHConfig
from dotenv import load_dotenv
import os
import numpy as np
from colorama import Fore, Style
from urban_watch.ml_logic.data import get_data, load_data, RAW_DATA_DIR
from urban_watch.ml_logic.preprocessing import apply_preproc_X_y, preprocess_image
from urban_watch.ml_logic.labels import get_bbox_from_features, bbox_to_wgs84, tile_name_from_bbox_wgs84, get_label_array, load_labels_y, LABELS_DIR
from urban_watch.ml_logic.model import split_data, train_logreg,train_random_forest, train_xgb, evaluate_model
from urban_watch.ml_logic.registry import save_model, save_results, mlflow_run, load_model
from urban_watch.params import *

#   MAIN FUNCTIONS CALLED BY CLI

def full_preproc_pipeline():
    """
    Run the full preprocessing pipeline:
    1. Load env
    2. Generate bboxes
    3. Get the features X:
            - Download or load tiles if already downloaded
            - Cloud masking (optional when ready)
            - Compute indices (NDVI, etc.)
            - Normalize images
            - Save X_final.npy for ML models
    4. Get the labels y:
            - retrieve the bboxes of features X as a list
            - project these bboxes onto WGS84
            - Using the bbox coordinates, find the name of the tif file for the labels stored in GCP.
            - Download the labels from GCP
            - Crop the label y so that it has the same bounding box as X + projection of y according to the same coordinate system as X.

    """
    print(Fore.CYAN + "\n🚀 Running PREPROCESSING PIPELINE...\n" + Style.RESET_ALL)

    # Configure access to the SentinelHub API
    config = SHConfig()
    config.sh_client_id = SH_CLIENT_ID
    config.sh_client_secret = SH_CLIENT_SECRET

    # Center of the bbox used for the train
    list_bbox_centers = [
    (43.59533138728404, 6.944167292762329), # hauteur de Cannes, melange urbain/vegetation
    (43.407067616126554, 6.522669188972711), # Var/vegetation
    (43.128306760274825, 6.121503808299507), # Hyère/vegetation
    (43.13997422359963, 5.928698334059295), # hauteur de Toulon
    (43.23554616596631, 5.881493437321922), # parc regional sainte baume/vegetation
    (43.219759411248, 5.7558111438277315), # Le castellet dans le Var/champs/vegetation/lotissement
    (43.58587029251304, 5.4526213864786985), # Au dessus d'AIx en Pce/champs,route,maisons
    (43.293089430781095, 5.389109345966383), # Marseille
    (43.5306142665247, 5.438947178151266), # Aix
    (43.53546797333638, 1.9686993230644818), # Est de Toulouse/champs
    (48.83661245140578, 2.409813543996677), # Est de Paris/urbain et parc
    (48.115035919487454, -1.6830147555182722), # Rennes
    (46.702018498290286, 0.7668558109521145), # Champs vers Poitier
    (47.49143988878275, 2.0914005645009692), # Forêt sud d'orléans
    (47.26267555373632, 4.060278597924947), # PNR Morvan
    (44.802566728154716, 4.375843852678453), # Ardèche
    (43.88785890061108, 0.5595568148255528), # Sud-ouest/champs
    (45.1278081381777, -1.0510523421771456), # Médoc
    (43.51973871166271, 0.9356166219979338) # champs sud-ouest
 ]

    ########## Get the features X ##########
    ########################################
    if len(os.listdir(RAW_DATA_DIR)) == 0:
        data = get_data(list_bbox_centers, config)

    X, meta = load_data()

    print(Fore.CYAN + "✅ Features are loaded \n" + Style.RESET_ALL)
    print(f"X shape: {X[0].shape}")

    ########## Get the labels y ############
    ########################################
    if len(os.listdir(LABELS_DIR)) == 0:
        list_bbox, list_crs = get_bbox_from_features()
        list_bbox_wgs84 = bbox_to_wgs84(list_bbox, list_crs)
        tile_names = tile_name_from_bbox_wgs84(list_bbox_wgs84)
        y = get_label_array(tile_names, list_bbox_wgs84, list_bbox, list_crs)

    y = load_labels_y()

    print(f"y shape: {y[0].shape}")

    print(Fore.CYAN + "✅ Labels are loaded \n" + Style.RESET_ALL)

    X_preproc, y_preproc = apply_preproc_X_y(X,y)

    print("✅ preprocess() done \n")

    return X_preproc, y_preproc



MODEL_DISPATCHER = {
    "logreg": train_logreg,
    "random_forest": train_random_forest,
    "xgb": train_xgb
}


@mlflow_run
def train(X,y, model_type='logreg', **model_params):
    """
    - Train on the preprocessed X and y
    - Store training results and model weights
    Return metrics (i.e. precision, recall, f1, accuracy)
    """

    if model_type not in MODEL_DISPATCHER:
        raise ValueError(f"Unsupported model type: {model_type}")
    trainer_fn = MODEL_DISPATCHER[model_type]

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    model, train_metrics = trainer_fn(
        X_train, y_train,
        **model_params
    )

    params = dict(
        context="train",
        model_type= model_type,
        row_count=X_train.shape[0]
    )

    # Save results on MLFlow
    save_results(params=params, metrics=train_metrics)

    # Save model weight on MLFlow
    save_model(model=model)

    return train_metrics

@mlflow_run
def evaluate(X,y,model_name="logreg_model", model_type="logreg", stage="Production"):
    """
    Evaluate the performance of the latest production model on processed data
    Return a dictionary with the metrics (i.e. precision, recall, f1, accuracy)
    """
    model = load_model(
        model_name=model_name,
        model_type=model_type,
        stage=stage
        )
    if model is None:
        print(f"no model found: {model_name} ({stage})")
        return None


    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    params = dict(
        context="evaluate",
        model_name= model_name,
        model_type= model_type,
        stage=stage,
        row_count=X_test.shape[0]
                   )

    save_results(params=params, metrics=metrics)

    print("✅ evaluate() done \n")



def rebuild_prediction(y_pred, mask_valid, fill_value=np.nan):
    """
    Reconstruit une image de prédiction 300×300 à partir :
    - y_pred (1D array : pixels valides)
    - mask_valid (300×300 bool)
    """

    H, W = mask_valid.shape
    y_full = np.full((H, W), fill_value, dtype=np.float64)

    # Injecter les prédictions aux bonnes positions
    y_full[mask_valid] = y_pred

    return y_full



def pred(X_pred, model):

    X_processed, mask_valid = preprocess_image(X_pred)
    y_pred = model.predict(X_processed).reshape(-1)

    # Reconstruction image 300x300
    y_pred_full = rebuild_prediction(y_pred, mask_valid)

    print("\n✅ prediction done")
    print("Shape full :", y_pred_full.shape)

    return y_pred_full, np.nanmean(y_pred_full)
