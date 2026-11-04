
import pandas as pd
import pytest

from src.validate_dataset import (
    EXPECTED_COLUMNS,
    EXPECTED_SHA256,
    calculate_sha256,
    DATASET_PATH,
)


def test_original_dataset_hash():
    assert calculate_sha256(DATASET_PATH) == EXPECTED_SHA256


def test_dataset_structure():
    data = pd.read_csv(DATASET_PATH)
    assert data.shape == (7043, 21)
    assert list(data.columns) == EXPECTED_COLUMNS


def test_customer_ids_are_unique():
    data = pd.read_csv(DATASET_PATH)
    assert not data["customerID"].duplicated().any()


def test_no_duplicate_rows():
    data = pd.read_csv(DATASET_PATH)
    assert not data.duplicated().any()


def test_total_charges_non_convertible_count():
    data = pd.read_csv(DATASET_PATH, dtype={"TotalCharges": "string"})
    values = pd.to_numeric(data["TotalCharges"].str.strip(), errors="coerce")
    assert values.isna().sum() == 11


def test_target_values():
    data = pd.read_csv(DATASET_PATH)
    assert set(data["Churn"].unique()) == {"Yes", "No"}
    assert data["Churn"].value_counts().to_dict() == {
        "No": 5174,
        "Yes": 1869,
    }
