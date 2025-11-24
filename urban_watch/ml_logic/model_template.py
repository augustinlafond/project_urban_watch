
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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


def train_logreg(X_train, y_train):
    """
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
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
