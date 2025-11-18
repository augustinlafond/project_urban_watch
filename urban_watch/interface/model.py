# urban_watch/ml_logic/model.py

"""
Model definitions for Urban Watch (ONLY classic ML, no CNN).

We assume:
- X is an image tensor of shape (n_tiles, 300, 300, 13)
  (10 spectral bands + 3 indices)
- y is a 1D array of labels: shape (n_tiles,)

This file provides three levels of models:
1. Baseline: simple, fast, interpretable (LogReg / RandomForest)
2. Medium: GradientBoosting / XGBoost simple
3. Advanced: tuned XGBoost (and potential stacking later)

All models are purely tabular: they work on aggregated features
computed from each tile (mean, std, percentiles, etc.).
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Optional: comment out if you don't have xgboost installed yet
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


PROCESSED_DIR = Path("processed_data")


# ---------------------------------------------------------
# 1. FEATURE EXTRACTION (image -> tabular)
# ---------------------------------------------------------

def extract_tile_features(X: np.ndarray) -> pd.DataFrame:
    """
    Convert a batch of tiles (N, H, W, C) into tabular features (N, F).

    For each tile and each channel, we compute:
    - mean, std, min, max, median
    - percentile 5, 25, 75, 95

    With 13 channels and 9 stats, that gives 117 features.
    You can easily extend / modify this list.

    Args:
        X: np.ndarray, shape (n_tiles, H, W, C)

    Returns:
        features: pd.DataFrame, shape (n_tiles, n_features)
    """

    n, h, w, c = X.shape
    print(f"[extract_tile_features] X shape = {X.shape}")

    stats = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "median": np.median,
    }

    percentiles = [5, 25, 75, 95]

    rows = []
    for i in range(n):
        img = X[i]  # (H, W, C)
        feat_dict = {}

        for ch in range(c):
            channel = img[:, :, ch]

            # Basic stats
            for stat_name, func in stats.items():
                val = func(channel)
                feat_dict[f"ch{ch}_{stat_name}"] = float(val)

            # Percentiles
            for p in percentiles:
                val = np.percentile(channel, p)
                feat_dict[f"ch{ch}_p{p}"] = float(val)

        rows.append(feat_dict)

    features = pd.DataFrame(rows)
    print(f"[extract_tile_features] Created feature table of shape {features.shape}")
    return features


def _load_X_for_training() -> np.ndarray:
    """
    Load preprocessed X for training from disk.

    TODO:
    - Adapt this to your pipeline once X_final is fully defined.
    - For now, assumes you saved (n_tiles, 300, 300, 13) as X_final.npy.
    """
    path = PROCESSED_DIR / "X_final.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run your preprocessing pipeline before training."
        )
    X = np.load(path)
    return X


def _load_y_for_training() -> np.ndarray:
    """
    Load labels y (one label per tile).

    TODO:
    - Implement this when Augustin's labels are ready.
    - Example: load from a .npy, .csv, or from BigQuery.

    For now this raises to remind you it's not wired yet.
    """
    raise NotImplementedError("Implement _load_y_for_training() once labels are available.")


# ---------------------------------------------------------
# 2. BASELINE MODEL (LogReg / RandomForest)
# ---------------------------------------------------------

def train_baseline_model(test_size: float = 0.2, random_state: int = 42):
    """
    Baseline ML model:
    - Features: aggregated stats from extract_tile_features()
    - Model: Logistic Regression OR RandomForest

    Goal:
    - Provide a simple, fast baseline to compare all future models.
    """

    print("\n[train_baseline_model] Loading X and y...")
    X = _load_X_for_training()
    y = _load_y_for_training()

    print("[train_baseline_model] Extracting tabular features...")
    X_tab = extract_tile_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- Option 1: Logistic Regression (simple baseline) ---
    print("\n[train_baseline_model] Training LogisticRegression baseline...")
    logreg = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced"
    )
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    print("\n=== LogisticRegression baseline report ===")
    print(classification_report(y_test, y_pred_lr))

    # --- Option 2: RandomForest (tree-based baseline) ---
    print("\n[train_baseline_model] Training RandomForest baseline...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("\n=== RandomForest baseline report ===")
    print(classification_report(y_test, y_pred_rf))

    # TODO: save the best baseline model to disk (registry.py style)
    # e.g. joblib.dump(rf, "models/baseline_rf.joblib")

    return {
        "logreg": logreg,
        "random_forest": rf,
        "X_test": X_test,
        "y_test": y_test
    }


# ---------------------------------------------------------
# 3. MEDIUM MODEL (GradientBoosting / simple XGBoost)
# ---------------------------------------------------------

def train_medium_model(test_size: float = 0.2, random_state: int = 42):
    """
    Medium complexity ML model:
    - Same features as baseline (X_tab)
    - Model: GradientBoostingClassifier or simple XGBoost

    Goal:
    - Improve accuracy over baseline without going too crazy.
    """

    print("\n[train_medium_model] Loading X and y...")
    X = _load_X_for_training()
    y = _load_y_for_training()

    print("[train_medium_model] Extracting tabular features...")
    X_tab = extract_tile_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- GradientBoostingClassifier ---
    print("\n[train_medium_model] Training GradientBoostingClassifier...")
    gbc = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state
    )
    gbc.fit(X_train, y_train)
    y_pred_gbc = gbc.predict(X_test)
    print("\n=== GradientBoostingClassifier report ===")
    print(classification_report(y_test, y_pred_gbc))

    # --- Optional XGBoost (if available) ---
    if XGBOOST_AVAILABLE:
        print("\n[train_medium_model] Training XGBoost (simple config)...")
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic",
            tree_method="hist",
            eval_metric="mlogloss"
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        print("\n=== XGBoost (simple) report ===")
        print(classification_report(y_test, y_pred_xgb))
    else:
        xgb = None
        print("\n⚠️ XGBoost not installed, skipping XGBoost training.")

    # TODO: save best medium model (e.g., xgb or gbc)
    return {
        "gbc": gbc,
        "xgb": xgb,
        "X_test": X_test,
        "y_test": y_test
    }


# ---------------------------------------------------------
# 4. ADVANCED MODEL (tuned XGBoost / future stacking)
# ---------------------------------------------------------

def train_advanced_model(test_size: float = 0.2, random_state: int = 42):
    """
    Advanced ML model (still no CNN):
    - Uses same tabular features
    - Model: more tuned XGBoost (and later, maybe stacking)

    TODO:
    - Add hyperparameter tuning (GridSearchCV / Optuna)
    - Optionally add stacking (RF + GBC + XGB -> meta-learner)
    """

    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is required for advanced model but is not installed.")

    print("\n[train_advanced_model] Loading X and y...")
    X = _load_X_for_training()
    y = _load_y_for_training()

    print("[train_advanced_model] Extracting tabular features...")
    X_tab = extract_tile_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\n[train_advanced_model] Training tuned XGBoost...")

    # Advanced XGBoost config (à affiner selon vos essais)
    xgb_adv = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic",
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1
    )

    xgb_adv.fit(X_train, y_train)
    y_pred_adv = xgb_adv.predict(X_test)
    print("\n=== Advanced XGBoost report ===")
    print(classification_report(y_test, y_pred_adv))

    # TODO: à terme, ajouter un stacking:
    # - Entrain RF, GBC, XGB simple
    # - Utiliser leurs prédictions comme features pour un LogisticRegression

    # TODO: save advanced model to registry
    return {
        "xgb_advanced": xgb_adv,
        "X_test": X_test,
        "y_test": y_test
    }
