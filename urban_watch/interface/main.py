"""
Main orchestration script for the Urban Watch project.

This file ONLY coordinates the high-level pipeline steps:
- Preprocessing (download tiles, masking, indices, normalization)
- Feature extraction
- Model training (baseline / medium / advanced)
- Evaluation (soon)
- Prediction (later)

"""


# Standard library
import os

# Third-party libraries
import numpy as np
from colorama import Fore, Style
from sentinelhub import SHConfig

# Project-specific imports
from urban_watch.params import SH_CLIENT_ID, SH_CLIENT_SECRET
from urban_watch.ml_logic.data import get_data, load_data, RAW_DATA_DIR
from urban_watch.ml_logic.preprocessing import apply_preproc_X_y, preprocess_image
from urban_watch.ml_logic.labels import (
    get_bbox_from_features,
    bbox_to_wgs84,
    tile_name_from_bbox_wgs84,
    get_label_array,
    load_labels_y,
    LABELS_DIR,
)
from urban_watch.ml_logic.model import (
    split_data,
    train_logreg,
    train_random_forest,
    train_xgb,
    evaluate_model,
    tune_xgboost
)
from urban_watch.ml_logic.registry import (
    save_model,
    save_results,
    mlflow_run,
    load_model,
)

import xgboost as xgb


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
        (43.59533138728404, 6.944167292762329),  # Cannes area, mix of urban/vegetation
        (43.407067616126554, 6.522669188972711),  # Var region / vegetation
        (43.128306760274825, 6.121503808299507),  # Hyère/vegetation
        (43.13997422359963, 5.928698334059295),  # North of Toulon
        (
            43.23554616596631,
            5.881493437321922,
        ),  # Sainte-Baume regional park / vegetation
        (
            43.219759411248,
            5.7558111438277315,
        ),  # Le Castellet (Var) / fields / vegetation / housing
        (
            43.58587029251304,
            5.4526213864786985,
        ),  # Au dessus d'AIx en Pce/champs,route,maisons
        (43.293089430781095, 5.389109345966383),  # Marseille
        (43.5306142665247, 5.438947178151266),  # Aix
        (43.53546797333638, 1.9686993230644818),  # East of Toulouse / fields
        (48.83661245140578, 2.409813543996677),  # East Paris/urban + park
        (48.115035919487454, -1.6830147555182722),  # Rennes
        (46.702018498290286, 0.7668558109521145),  # Fields near Poitiers
        (47.49143988878275, 2.0914005645009692),  # Forest south of Orléans
        (47.26267555373632, 4.060278597924947),  # PNR Morvan
        (44.802566728154716, 4.375843852678453),  # Ardèche
        (43.88785890061108, 0.5595568148255528),  # Southwest / fields
        (45.1278081381777, -1.0510523421771456),  # Médoc
        (43.51973871166271, 0.9356166219979338),  # Southwest fields
        (45.75700560418582, 4.837894458418483), #Lyon
        (48.891152468380895, 2.239094188930942), #La Défense, Paris
        (48.86211843574683, 2.2905981373634647), # Paris, ouest
        (47.20618773079231, -1.5459163950180743), #Nantes
        (47.08464183721342, -1.6962596890656523), #Réserve naturelle du lac grand lieu
        (48.40034048180863, -4.469675087417314), #Brest
        (46.1586960979153, -1.1755560282124524), #La Rochelle
        (44.844587556634146, -0.5829596959701742),#Bordeaux
        (44.55369842246411, -1.0496900841748926),  #champs côte atlantique
        (43.628414501899975, 1.4448563066009605), #Toulouse nord
        (43.569022158861245, 1.4388048524502546), #Toulouse sud
        (43.613980641781446, 3.8639232537015333), #Montpellier
        (43.52966326291851, 4.215722893206303), #Etangs vers grau du roi
        (43.48589477322024, 4.301058354400013), #Etangs vers grau du roi_2
        (43.83913985594882, 4.358150296025464), #Nimes
        (43.70159048778169, 7.26353168007621), #Nices
        (48.5783372290238, 7.743236109562404), #Strasbourg
        (48.71254652139096, 7.456551198100515), #Champs à côté de Strasbourg
        (48.57456545847434, 7.13492831210087), #Forêt Moselle
        (48.13420361687641, 6.612626640624101), #Vosges
        (44.07958873358449, -0.5877954280139801), #Parc naturel landes de gascogne
        (43.89019877443544, 1.6878249944173407), #Champs sud ouest (vers Gaillac)
        (46.19665157664145, -1.4198303131713605), #Ile de ré
        (45.9778298202333, -1.4040436132469687), #Ile d'Oléron
        (42.92820871972086, 0.12882996229236), #Pyrénnées (Pic du midi de Bigorre)
        (45.91019201754405, 6.877210430219988), #Chamonix
        (43.793200686954954, 0.47904131618370716), #champs sud-ouest
        (43.62883173367989, -0.44509715612673423), #champs sud-ouest
        (43.21417810920307, -1.0111140447352363), #champs sud-ouest
        (42.69476656356274, 2.9145523571599234), #Perpignan
        (44.36818889111061, 6.882259133095895), #Mercantour


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

    X_preproc, y_preproc = apply_preproc_X_y(X, y)

    print("✅ preprocess() done \n")

    return X_preproc, y_preproc


MODEL_DISPATCHER = {
    "logreg": train_logreg,
    "random_forest": train_random_forest,
    "xgb": train_xgb,
    "xgb_tuned": tune_xgboost
}


@mlflow_run
def train(X, y, model_type="logreg", **model_params):
    """
    - Train on the preprocessed X and y
    - Store training results and model weights
    Return metrics (i.e. precision, recall, f1, accuracy)
    """

    if model_type not in MODEL_DISPATCHER:
        raise ValueError(f"Unsupported model type: {model_type}")
    trainer_fn = MODEL_DISPATCHER[model_type]

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    model, train_metrics = trainer_fn(X_train, y_train, **model_params)

    params = dict(context="train", model_type=model_type, row_count=X_train.shape[0])

    # Save results on MLFlow
    save_results(params=params, metrics=train_metrics)

    # Save model weight on MLFlow
    save_model(model=model)

    return train_metrics


@mlflow_run
def evaluate(X, y, model_name="logreg_model", model_type="logreg", stage="Production"):
    """
    Evaluate the performance of the latest production model on processed data
    Return a dictionary with the metrics (i.e. precision, recall, f1, accuracy)
    """
    model = load_model(model_name=model_name, model_type=model_type, stage=stage)
    if model is None:
        print(f"no model found: {model_name} ({stage})")
        return None

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    params = dict(
        context="evaluate",
        model_name=model_name,
        model_type=model_type,
        stage=stage,
        row_count=X_test.shape[0],
    )

    save_results(params=params, metrics=metrics)

    print("✅ evaluate() done \n")


def rebuild_prediction(y_pred, mask_valid, fill_value=-1):
    """
    Reconstructs a 300×300 prediction image from: :
    - y_pred (1D array : valid pixels)
    - mask_valid (300×300 boolean)
    """

    H, W = mask_valid.shape
    y_full = np.full((H, W), fill_value, dtype=np.int64)

    # Inject the predictions into the correct positions
    y_full[mask_valid] = y_pred

    return y_full


def pred(X_pred, model):
    # Step 1 — preprocess image
    X_processed, mask_valid = preprocess_image(X_pred)

    # Step 2 — Convertir en DMatrix si modèle = Booster
    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(X_processed)
        y_pred = (model.predict(dmatrix) > 0.5).astype(int)
    else:
        # Cas XGBClassifier (rare)
        y_pred = model.predict(X_processed)

    y_pred = y_pred.reshape(-1)

    # Image reconstruction 300x300
    y_pred_full = rebuild_prediction(y_pred, mask_valid)

    mean_urban_score = np.mean(y_pred_full[y_pred_full != -1])

    print("\n✅ prediction done")
    print("Shape full :", y_pred_full.shape)

    return y_pred_full, mean_urban_score
