
"""Pruebas automatizadas del preprocesamiento predictivo."""

import numpy as np
import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from src.preprocessing import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_model_pipeline,
    build_preprocessor,
    load_training_data,
)


@pytest.fixture(scope="module")
def training_data():
    return load_training_data()


def test_training_data_structure(training_data):
    X, y = training_data

    assert X.shape == (5634, 19)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) == {0, 1}
    assert not X.isna().all(axis=1).any()
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns


def test_preprocessing_produces_finite_values(training_data):
    X, _ = training_data

    preprocessor = build_preprocessor(scale_numeric=True)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= len(FEATURE_COLUMNS)
    assert np.isfinite(transformed).all()


def test_unseen_category_is_handled(training_data):
    X, _ = training_data

    X_train = X.iloc[:100].copy()
    X_new = X.iloc[100:101].copy()

    X_new.loc[:, "Contract"] = "UNSEEN_CATEGORY"

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    transformed = preprocessor.transform(X_new)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_pipeline_fits_inside_cv_fold(training_data):
    X, y = training_data

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    train_indices, validation_indices = next(cv.split(X, y))

    pipeline = build_model_pipeline(
        DummyClassifier(strategy="prior"),
        scale_numeric=True,
    )

    pipeline.fit(
        X.iloc[train_indices],
        y.iloc[train_indices],
    )

    predictions = pipeline.predict_proba(
        X.iloc[validation_indices]
    )

    assert predictions.shape == (len(validation_indices), 2)
    assert np.isfinite(predictions).all()
    assert np.allclose(predictions.sum(axis=1), 1)


def test_numeric_imputation_uses_training_only(training_data):
    """La mediana debe aprenderse solo del subconjunto de entrenamiento."""

    X, _ = training_data

    train_sample = X.iloc[:100].copy()
    validation_sample = X.iloc[100:101].copy()

    # Valores controlados para comprobar la imputación.
    train_sample.loc[:, "TotalCharges"] = 100.0
    train_sample.loc[train_sample.index[0], "TotalCharges"] = 300.0

    validation_sample.loc[:, "TotalCharges"] = np.nan

    # Ajustar nuestro preprocesador real SOLO con entrenamiento.
    preprocessor = build_preprocessor(scale_numeric=False)
    preprocessor.fit(train_sample)

    # Verificar el parámetro aprendido por el SimpleImputer interno.
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]

    total_charges_position = NUMERIC_FEATURES.index("TotalCharges")
    learned_median = imputer.statistics_[total_charges_position]

    assert learned_median == 100.0

    # Transformar validación sin volver a ajustar el preprocesador.
    transformed = preprocessor.transform(validation_sample)

    assert transformed[0, total_charges_position] == 100.0

    # Una validación con valores extremos no debe modificar la mediana.
    assert (
        preprocessor.named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_[total_charges_position]
        == 100.0
    )
