
"""Unit tests for the ChurnSense preprocessing pipeline."""

import pandas as pd

from src.preprocessing import create_features, create_preprocessor, load_data


def test_create_features_adds_num_services():
    """Check that num_services correctly counts active services."""
    df = pd.DataFrame({
        "PhoneService": ["Yes", "No"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["Yes", "No"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
    })

    result = create_features(df)

    assert result["num_services"].tolist() == [4, 2]
    assert "num_services" not in df.columns


def test_create_preprocessor_returns_transformer():
    """Check that the preprocessor can be created."""
    df = pd.DataFrame({
        "tenure": [1, 12, 24],
        "MonthlyCharges": [30.0, 60.0, 90.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": [0, 1, 0],
    })

    preprocessor = create_preprocessor(df)

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2


def test_load_data_cleans_missing_total_charges(tmp_path):
    """Check that loading removes invalid charges and encodes churn."""
    df = pd.DataFrame({
        "customerID": ["A", "B"],
        "tenure": [1, 12],
        "TotalCharges": ["29.5", " "],
        "Churn": ["No", "Yes"],
    })

    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    result = load_data(csv_path)

    assert len(result) == 1
    assert result["Churn"].tolist() == [0]
    assert "customerID" not in result.columns
    
from src.model import create_model_pipelines


def test_create_model_pipelines_returns_five_models():
    """Check that five model pipelines are created."""
    df = pd.DataFrame({
        "tenure": [1, 12, 24],
        "MonthlyCharges": [30.0, 60.0, 90.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": [0, 1, 0],
    })

    preprocessor = create_preprocessor(df)
    pipelines = create_model_pipelines(preprocessor)

    assert len(pipelines) == 5
    assert "Gradient Boosting" in pipelines
    assert "Logistic Regression" in pipelines
    
from src.utils import predict_churn


def test_predict_churn_returns_predictions_and_probabilities():
    """Check that the helper returns valid churn predictions."""
    df = pd.DataFrame({
        "tenure": [1, 12, 24, 36, 48, 60],
        "MonthlyCharges": [30.0, 60.0, 90.0, 40.0, 75.0, 95.0],
        "Contract": [
            "Month-to-month", "One year", "Two year",
            "Month-to-month", "One year", "Two year"
        ],
        "Churn": [0, 1, 0, 1, 0, 0],
    })

    preprocessor = create_preprocessor(df.drop(columns=["Churn"]))
    pipeline = create_model_pipelines(preprocessor)["Logistic Regression"]

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    pipeline.fit(X, y)
    results = predict_churn(pipeline, X.iloc[:2])

    assert len(results) == 2
    assert "predicted_churn" in results.columns
    assert "churn_probability" in results.columns
    assert results["predicted_churn"].isin([0, 1]).all()
    assert results["churn_probability"].between(0, 1).all()