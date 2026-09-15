import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.joblib"


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features(X):
    """
    Convert IMU windows into 42 statistical features.

    Input:
        X shape = (n_windows, 128, 6)

    Output:
        shape = (n_windows, 42)
    """

    features = []

    for window in X:

        row = []

        for i in range(6):

            signal = window[:, i]

            mean = np.mean(signal)
            std = np.std(signal)
            minimum = np.min(signal)
            maximum = np.max(signal)
            range_value = maximum - minimum
            median = np.median(signal)
            rms = np.sqrt(np.mean(signal ** 2))

            row.extend([
                mean,
                std,
                minimum,
                maximum,
                range_value,
                median,
                rms
            ])

        features.append(row)

    return np.asarray(features)


# --------------------------------------------------
# Prepare features for model
# --------------------------------------------------

def prepare_features(X):
    """
    Extract features and preserve the model's
    original feature-column order.
    """

    features = extract_features(X)

    if hasattr(model, "feature_names_in_"):
        return pd.DataFrame(
            features,
            columns=model.feature_names_in_
        )

    return features


# --------------------------------------------------
# Batch prediction
# --------------------------------------------------

def predict_batch(X):
    """
    Predict activities for multiple IMU windows.

    Input:
        X shape = (n_windows, 128, 6)

    Returns:
        predictions
        confidences
        probabilities
    """

    X = np.asarray(X)

    if X.ndim != 3 or X.shape[1:] != (128, 6):
        raise ValueError(
            f"Expected input shape (n_windows, 128, 6), "
            f"received {X.shape}"
        )

    features = prepare_features(X)

    # ONE model call instead of thousands
    predictions = model.predict(features)

    # ONE probability call instead of thousands
    probabilities = model.predict_proba(features)

    confidences = probabilities.max(axis=1)

    return predictions, confidences, probabilities


# --------------------------------------------------
# Single-window prediction
# --------------------------------------------------

def predict_activity(window):
    """
    Predict one activity from one IMU window.

    Input:
        window shape = (128, 6)

    Returns:
        prediction
        confidence
        probability dictionary
    """

    window = np.asarray(window)

    if window.shape != (128, 6):
        raise ValueError(
            f"Expected window shape (128, 6), "
            f"received {window.shape}"
        )

    predictions, confidences, probabilities = predict_batch(
        window[np.newaxis, :, :]
    )

    prediction = predictions[0]
    confidence = float(confidences[0])

    probability_dict = {
        str(class_name): float(probability)
        for class_name, probability in zip(
            model.classes_,
            probabilities[0]
        )
    }

    return prediction, confidence, probability_dict