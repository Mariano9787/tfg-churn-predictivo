
"""Preparación de datos para los modelos predictivos.

Las transformaciones se ajustan exclusivamente durante fit().
No se utiliza el conjunto de prueba reservado.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_training_data():
    """Carga exclusivamente la partición de entrenamiento."""
    data = pd.read_parquet(TRAIN_PATH)

    if data["customerID"].duplicated().any():
        raise ValueError("Hay identificadores duplicados.")

    if not data["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("La variable objetivo contiene valores inválidos.")

    X = data[FEATURE_COLUMNS].copy()
    y = data["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    return X, y


def build_preprocessor(scale_numeric=True):
    """Construye un preprocesador nuevo, todavía sin ajustar."""

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(estimator, scale_numeric=True):
    """Une el preprocesamiento y el estimador en un pipeline."""

    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    scale_numeric=scale_numeric
                ),
            ),
            ("model", estimator),
        ]
    )


if __name__ == "__main__":
    X, y = load_training_data()

    print("TRAINING DATA LOADED")
    print(f"Rows: {len(X)}")
    print(f"Predictors: {len(X.columns)}")
    print(f"Target classes: {sorted(y.unique().tolist())}")

    print()
    print("PREPROCESSING PIPELINE CREATED")
    print(build_preprocessor())
