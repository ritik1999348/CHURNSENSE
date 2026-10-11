
"""Model creation utilities for the ChurnSense project."""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def create_model_pipelines(preprocessor):
    """Create preprocessing and classification pipelines for five models.

    Args:
        preprocessor: A fitted-compatible scikit-learn transformer.

    Returns:
        dict: Model names mapped to scikit-learn pipelines.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    pipelines = {}

    for name, model in models.items():
        pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

    return pipelines


def create_gradient_boosting_pipeline(preprocessor, params=None):
    """Create a Gradient Boosting pipeline with optional model parameters.

    Args:
        preprocessor: A scikit-learn preprocessing transformer.
        params (dict, optional): Parameters for GradientBoostingClassifier.

    Returns:
        Pipeline: A preprocessing and Gradient Boosting pipeline.
    """
    model = GradientBoostingClassifier(
        random_state=42,
        **(params or {})
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])