
"""Helper utilities for saving, loading, and using ChurnSense models."""

from pathlib import Path

import joblib
import pandas as pd


def save_model(model, file_path):
    """Save a trained model or pipeline to a file.

    Args:
        model: Trained scikit-learn model or pipeline.
        file_path: Destination path for the saved model.

    Returns:
        Path: Path where the model was saved.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(file_path):
    """Load a previously saved model or pipeline.

    Args:
        file_path: Path to the saved model file.

    Returns:
        object: The loaded model or pipeline.
    """
    return joblib.load(file_path)


def predict_churn(model, customer_data):
    """Predict churn labels and probabilities for customer records.

    Args:
        model: Trained pipeline that supports predict and predict_proba.
        customer_data: DataFrame containing the model's required input features.

    Returns:
        DataFrame: Predicted churn labels and churn probabilities.
    """
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("customer_data must be a pandas DataFrame.")

    predictions = model.predict(customer_data)
    probabilities = model.predict_proba(customer_data)[:, 1]

    return pd.DataFrame({
        "predicted_churn": predictions,
        "churn_probability": probabilities,
    }, index=customer_data.index)