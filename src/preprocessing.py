import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(file_path):
    """Load the Telco Customer Churn dataset."""
    df = pd.read_csv(file_path)

    # Drop customer ID
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    # Encode target
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    return df


def create_features(df):
    """Create engineered features."""

    service_columns = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["num_services"] = 0

    for col in service_columns:
        df["num_services"] += (
            df[col] == "Yes"
        ).astype(int)

    return df


def create_preprocessor(df):
    """Create preprocessing pipeline."""

    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    numeric_features.remove("Churn")

    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return preprocessor

"""Data loading, feature engineering, and preprocessing for ChurnSense."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(file_path):
    """Load and clean the Telco churn dataset.

    Args:
        file_path (str): Path to the source CSV file.

    Returns:
        pandas.DataFrame: Cleaned data with encoded churn labels.
    """
    df = pd.read_csv(file_path)

    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def create_features(df):
    """Add engineered features to customer data.

    Args:
        df (pandas.DataFrame): Cleaned customer data.

    Returns:
        pandas.DataFrame: A copy of the data with num_services added.
    """
    df = df.copy()

    service_columns = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["num_services"] = sum(
        (df[col] == "Yes").astype(int)
        for col in service_columns
    )

    return df


def create_preprocessor(df):
    """Create a transformer for numeric and categorical columns.

    Args:
        df (pandas.DataFrame): Data used to identify feature columns.

    Returns:
        ColumnTransformer: Numeric scaling and categorical encoding steps.
    """
    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if "Churn" in numeric_features:
        numeric_features.remove("Churn")

    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )