
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from colorama import Fore, Style

def split_data(X, y, test_size=0.2, seed=42):
    """
    Train/test split.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )


def train_logreg(X_train, y_train):
    """
    Train the model
    """
    model = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    metrics = dict(precision_train = train_precision,
                   recall_train = train_recall,
                   f1_train = train_f1,
                   accuracy_train = train_accuracy)

    print(f"✅ Model trained : {metrics}")
    return model, metrics

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    metrics = dict(precision_train = train_precision,
                   recall_train = train_recall,
                   f1_train = train_f1,
                   accuracy_train = train_accuracy)

    print(f"✅ Model trained : {metrics}")
    return model, metrics

def train_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    metrics = dict(precision_train=train_precision,
                   recall_train=train_recall,
                   f1_train=train_f1,
                   accuracy_train=train_accuracy)

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

    metrics = dict(precision_test = precision,
                   recall_test = recall,
                   f1_test = f1,
                   accuracy_test = accuracy)

    print(f"✅ Model evaluated : {metrics}")

    return metrics
