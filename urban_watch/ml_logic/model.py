"""
model.py — Baseline Random Forest model for UrbanWatch
-------------------------------------------------------

Compatible with:
- data.py (loading X tiles)
- labels.py (loading Y tiles)
- package.py (preprocess_image => 13 standardized bands)

Pipeline:
    X_array        = (n_tiles, 300,300,10)
    X_processed    = list of (n_valid_pixels, 13)
    valid_masks    = list of (300,300)
    Y_tiles        = list of (300,300)

Output: pixel-level RandomForest model.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# 1. Build pixel-level dataset
# ============================================================

def build_pixel_dataset(x_processed_list, y_tiles, valid_masks):
    """
    Convert tiles into supervised pixel-level dataset.

    Args:
        x_processed_list (list): list of arrays (n_valid_pixels, 13)
        y_tiles (list): list of (300,300) label tiles
        valid_masks (list): list of boolean masks (300,300)

    Returns:
        tuple:
            x_pixels (np.ndarray): (N, 13)
            y_pixels (np.ndarray): (N,)
    """
    x_pixels = []
    y_pixels = []

    for x_tile, y_tile, mask in zip(x_processed_list, y_tiles, valid_masks):
        y_valid = y_tile[mask]
        x_pixels.append(x_tile)
        y_pixels.append(y_valid)

    x_pixels = np.vstack(x_pixels)
    y_pixels = np.hstack(y_pixels)

    return x_pixels, y_pixels


# ============================================================
# 2. Stratified split
# ============================================================

def split_data(x_data, y_data, test_size=0.2, seed=42):
    """
    Stratified train/test split.

    Args:
        x_data (np.ndarray): feature matrix
        y_data (np.ndarray): labels
        test_size (float): test proportion
        seed (int): random seed

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    return train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=seed,
        stratify=y_data,
    )


# ============================================================
# 3. Train Random Forest
# ============================================================

def train_random_forest(x_train, y_train):
    """
    Train baseline RandomForest model.

    Args:
        x_train (np.ndarray): training features
        y_train (np.ndarray): training labels

    Returns:
        RandomForestClassifier: trained model
    """
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        bootstrap=True,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


# ============================================================
# 4. Evaluation metrics
# ============================================================

def evaluate(model, x_test, y_test):
    """
    Compute accuracy, F1-score and detailed classification report.

    Args:
        model: trained model
        x_test (np.ndarray): test features
        y_test (np.ndarray): test labels

    Returns:
        tuple: accuracy, f1_score, classification_report (str)
    """
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n--- Evaluation ---")
    print("Accuracy:", acc)
    print("F1-score:", f1)
    print(report)

    return acc, f1, report


# ============================================================
# 5. Save / Load model
# ============================================================

def save_model(model, path="model_random_forest.joblib"):
    """
    Save trained model to disk.

    Args:
        model: trained model
        path (str): file path
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, path)
    print(f"✔ Model saved to: {path}")


def load_model(path="model_random_forest.joblib"):
    """
    Load model from disk.

    Args:
        path (str): model file path

    Returns:
        model: loaded model
    """
    model = joblib.load(path)
    print(f"✔ Model loaded from: {path}")
    return model


# ============================================================
# 6. Predict on a full tile
# ============================================================

def predict_tile(model, x_processed, valid_mask):
    """
    Predict classes for every valid pixel in a tile.

    Args:
        model: trained classifier
        x_processed (np.ndarray): (n_valid_pixels, 13)
        valid_mask (np.ndarray): (300,300) boolean mask

    Returns:
        np.ndarray: (300,300) map with values {-1,0,1}
    """
    preds = model.predict(x_processed)

    output = np.full(valid_mask.shape, -1, dtype=np.int8)
    output[valid_mask] = preds

    return output
