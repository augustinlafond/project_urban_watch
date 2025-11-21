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
from urban_watch.ml_logic.preprocessing import apply_preproc_X_y
from urban_watch.ml_logic.labels import get_bbox_from_features, bbox_to_wgs84, tile_name_from_bbox_wgs84, get_label_array, load_labels_y, LABELS_DIR

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

    load_dotenv()

    config = SHConfig()
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")

    # Center of the bbox used for the train
    list_bbox_centers = [
    (43.59533138728404, 6.944167292762329),
    (43.407067616126554, 6.522669188972711),
    (43.128306760274825, 6.121503808299507),
    (43.13997422359963, 5.928698334059295),
    (43.23554616596631, 5.881493437321922),
    (43.219759411248, 5.7558111438277315),
    (43.58587029251304, 5.4526213864786985),
    (43.293089430781095, 5.389109345966383),
    (43.5306142665247, 5.438947178151266),
    (43.53546797333638, 1.9686993230644818),
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

    return X_preproc, y_preproc

    print("✅ preprocess() done \n")
