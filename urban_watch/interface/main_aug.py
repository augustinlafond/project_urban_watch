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
from colorama import Fore, Style
from urban_watch.ml_logic.data import get_data, load_data, RAW_DATA_DIR
from urban_watch.ml_logic.package import preprocess_image
from urban_watch.ml_logic.labels import get_bbox_from_features, bbox_to_wgs84, tile_name_from_bbox_wgs84, get_label_array

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
    (43.64438559524513, 3.8164549979627576),
    (43.33317246154453, 5.4045395451787535),
    (43.48182194683079, 3.677980734975629),
    (43.57081309375424, 6.96242300618074),
    (44.55378060775165, 4.28389205759154),
    (45.69764498502806, 5.8945655967385315),
    (43.25807995877758, 5.473554475410006),
    (50.63412206750312, 3.0435779303880106),
    (48.83701959617146, 2.411376957546957)]

    ########## Get the features X ##########
    ########################################
    if len(os.listdir(RAW_DATA_DIR)) == 0:
        data = get_data(list_bbox_centers, config)

    X, meta = load_data()

    ########## Get the labels y ############
    ########################################

    list_bbox, list_crs = get_bbox_from_features()
    list_bbox_wgs84 = bbox_to_wgs84(list_bbox, list_crs)
    tile_names = tile_name_from_bbox_wgs84(list_bbox_wgs84)
    y = get_label_array(tile_names, list_bbox_wgs84, list_bbox, list_crs)




    print(Fore.GREEN + "\n🎉 Preprocessing completed!\n" + Style.RESET_ALL)


def train_baseline():
    """
    Train the baseline ML model.
    Logistic Regression + Random Forest.
    """
    print(Fore.CYAN + "\n🚀 TRAIN BASELINE MODEL\n" + Style.RESET_ALL)
    train_baseline_model()
    print(Fore.GREEN + "\n✅ Baseline model complete!\n" + Style.RESET_ALL)


def train_medium():
    """
    Train the medium ML model.
    GradientBoosting + simple XGBoost.
    """
    print(Fore.CYAN + "\n🚀 TRAIN MEDIUM MODEL\n" + Style.RESET_ALL)
    train_medium_model()
    print(Fore.GREEN + "\n✅ Medium model complete!\n" + Style.RESET_ALL)


def train_advanced():
    """
    Train the advanced ML model.
    Tuned XGBoost (and later stacking).
    """
    print(Fore.CYAN + "\n🚀 TRAIN ADVANCED MODEL\n" + Style.RESET_ALL)
    train_advanced_model()
    print(Fore.GREEN + "\n🔥 Advanced model complete!\n" + Style.RESET_ALL)

#                      COMMAND ENTRYPOINT

if __name__ == "__main__":
    """
    Temporary: while development is ongoing, we run only preprocessing.
    Later, arguments will be added:
        python main.py preprocess
        python main.py train_baseline
        python main.py train_medium
        python main.py train_advanced
    """

    # ⚠️ Choose what runs by default during development:
    preprocess()
    # train_baseline()
    # train_medium()
    # train_advanced()
