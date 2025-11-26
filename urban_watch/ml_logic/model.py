"""
Model Train/Evaluate

This file provides a set of functions designed to train and evaluate
several supervised classification models: Logistic Regression, Random Forest, and XGBoost.

The script also handles splitting the dataset into training and test sets,
computing performance metrics (precision, recall, F1-score, and accuracy),
and producing standardized evaluation outputs.

Each model has its own dedicated training function, while shared utilities ensure consistent
evaluation, metric reporting, and reproducibility across experiments.
"""


# Standard library
from colorama import Fore, Style

# Scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from scipy.stats import randint, uniform

# XGBoost
import xgboost as xgb



def split_data(X, y, test_size=0.2, seed=42):
    """
    Train/test split.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train_logreg(X_train, y_train):
    """
    Train the model
    """
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    metrics = dict(
        precision_train=train_precision,
        recall_train=train_recall,
        f1_train=train_f1,
        accuracy_train=train_accuracy,
    )

    print(f"✅ Model trained : {metrics}")
    return model, metrics


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    metrics = dict(
        precision_train=train_precision,
        recall_train=train_recall,
        f1_train=train_f1,
        accuracy_train=train_accuracy,
    )

    print(f"✅ Model trained : {metrics}")
    return model, metrics


def train_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    metrics = dict(
        precision_train=train_precision,
        recall_train=train_recall,
        f1_train=train_f1,
        accuracy_train=train_accuracy,
    )

    print(f"✅ Model trained : {metrics}")
    return model, metrics




def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X_test)} rows..." + Style.RESET_ALL)

    # Convert to DMatrix if model is a Booster
    if isinstance(model, xgb.Booster):
        X_test_dm = xgb.DMatrix(X_test)
        y_pred = (model.predict(X_test_dm) > 0.5).astype(int)

    else:
        # Standard scikit-learn API
        y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = dict(
        precision_test=precision, recall_test=recall, f1_test=f1, accuracy_test=accuracy
    )

    print(f"✅ Model evaluated : {metrics}")

    return metrics



def tune_xgboost(X_train, y_train, n_iter=25):
    """
    XGBoost random search optimized for speed on large tabular satellite data.
    """

    # -----------------------------
    # Hyperparameter distributions
    # -----------------------------
    param_dist = {
        "n_estimators": randint(200, 800),         # nb d’arbres
        "max_depth": randint(4, 12),               # profondeur limitée pour éviter overfitting
        "learning_rate": uniform(0.01, 0.2),       # lr assez large
        "subsample": uniform(0.6, 0.4),            # 0.6–1.0
        "colsample_bytree": uniform(0.6, 0.4),     # 0.6–1.0
        "gamma": uniform(0, 5),                    # réduction variance
        "min_child_weight": randint(1, 8),         # regularisation
        "reg_alpha": uniform(0, 1),                # L1
        "reg_lambda": uniform(0.5, 1.5),           # L2
    }

    # -----------------------------
    # XGBoost model
    # -----------------------------
    xgb = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",     # ⚡ FAST
        max_bin=256,            # meilleur compromis précision/temps
        n_jobs=-1,              # multi-core
        random_state=42
    )

    # -----------------------------
    # Randomized Search
    # -----------------------------
    rnd = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,          # 25 iterations = ~1–3h selon machine
        scoring="recall",       # priorité : max recall
        cv=3,                   # 3 folds = rapide et fiable
        verbose=2,
        random_state=42
    )

    rnd.fit(X_train, y_train)

    model = rnd.best_estimator_

    y_train_pred = model.predict(X_train)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    metrics = dict(
        precision_train=train_precision,
        recall_train=train_recall,
        f1_train=train_f1,
        accuracy_train=train_accuracy,
    )

    print("\n🌟 Best parameters:", rnd.best_params_)
    print("🏆 Best score:", rnd.best_score_)

    return model, metrics
