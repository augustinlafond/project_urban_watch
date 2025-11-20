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
    list_bbox_centers = [(43.52960344286241, 5.448962145567533),
(46.16802483919608, -1.1176074318571032),
 (43.5816738819709, 1.645701170443095),
 (43.48182194683079, 3.677980734975629),
 (43.8153030346785, 4.338764342740716),
 (43.61601277775063, 1.8758970522504868),
 (45.69764498502806, 5.8945655967385315),
 (42.6755834589268, 2.869650140046908),
 (47.31472637347379, 3.040818329458375),
 (47.432053447065, 1.8564002753276152),
 (44.917659363981606, 0.3198532687167684),
 (44.5354036268071, -1.0106115363381298),
 (42.69238910363254, 1.597205747150504),
 (43.74306182875029, 6.200148273979441),
 (43.29338648634513, 5.40265247747842),
 (48.856181368759366, 2.3370966981646553),
 (48.40265896290035, 2.7094534429859025),
 (45.754104529038855, 4.834183408486259),
 (44.76479997527986, 6.252615588665959),
 (47.211656711078476, -1.5545401474212523)
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
