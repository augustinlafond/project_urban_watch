# urban_watch/main.py

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

from colorama import Fore, Style
from dotenv import load_dotenv

# --- Preprocessing pipeline ---
from urban_watch.ml_logic.data import load_data  # used indirectly
from urban_watch.pipeline.preprocessing import run_preprocessing

# --- ML models ---
from urban_watch.ml_logic.model import (
    train_baseline_model,
    train_medium_model,
    train_advanced_model
)


#   MAIN FUNCTIONS CALLED BY CLI

def preprocess():
    """
    Run the full preprocessing pipeline:
    1. Load env
    2. Generate bboxes
    3. Download tiles
    4. Cloud masking (optional when ready)
    5. Compute indices
    6. Normalize images
    7. Save X_final.npy for ML models
    """
    print(Fore.CYAN + "\n🚀 Running PREPROCESSING PIPELINE...\n" + Style.RESET_ALL)
    load_dotenv()

    run_preprocessing()

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
