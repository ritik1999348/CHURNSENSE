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