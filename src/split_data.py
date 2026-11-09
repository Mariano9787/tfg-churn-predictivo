
"""Partición estratificada y reproducible del dataset IBM Telco.

La partición de prueba queda reservada para la evaluación final P9.
No se ajustan transformaciones ni modelos en este módulo.
"""

from pathlib import Path
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.validate_dataset import (
    DATASET_PATH,
    EXPECTED_COLUMNS,
    EXPECTED_SHA256,
    calculate_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "artifacts" / "split_manifest.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    # Verificar que trabajamos con el CSV original.
    if calculate_sha256(DATASET_PATH) != EXPECTED_SHA256:
        raise ValueError("El archivo original no coincide con su SHA-256.")

    data = pd.read_csv(
        DATASET_PATH,
        dtype={"TotalCharges": "string"},
    )

    if data.shape != (7043, 21):
        raise ValueError(f"Estructura inesperada: {data.shape}")

    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError("Las columnas no coinciden con el esquema esperado.")

    if data["customerID"].duplicated().any():
        raise ValueError("Existen identificadores de clientes duplicados.")

    if data["Churn"].isna().any():
        raise ValueError("La variable objetivo contiene valores ausentes.")

    if not data["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("La variable objetivo contiene valores inesperados.")

    # La conversión se realiza sin estimar parámetros.
    # La imputación se realizará después, dentro de los pipelines.
    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].str.strip(),
        errors="coerce",
    )

    # Reservar el test sin analizar relaciones predictor-objetivo.
    train_indices, test_indices = train_test_split(
        data.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["Churn"],
    )

    train_data = data.loc[train_indices].sort_index()
    test_data = data.loc[test_indices].sort_index()

    if len(set(train_indices) & set(test_indices)) != 0:
        raise AssertionError("Hay registros compartidos entre train y test.")

    if len(train_data) + len(test_data) != len(data):
        raise AssertionError("La partición no conserva todos los registros.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.parquet"
    test_path = OUTPUT_DIR / "test.parquet"

    train_data.to_parquet(train_path, index=True)
    test_data.to_parquet(test_path, index=True)

    manifest = {
        "source_dataset": DATASET_PATH.name,
        "source_sha256": EXPECTED_SHA256,
        "random_state": RANDOM_STATE,
        "test_size_fraction": TEST_SIZE,
        "stratified_by": "Churn",
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "total_rows": int(len(data)),
        "train_path": str(train_path.relative_to(ROOT)),
        "test_path": str(test_path.relative_to(ROOT)),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "holdout_policy": (
            "El test queda reservado para una única evaluación final "
            "en P9. No utilizar para seleccionar modelos, "
            "hiperparámetros ni transformaciones."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("STRATIFIED SPLIT COMPLETED")
    print(f"Train rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Random state: {RANDOM_STATE}")
    print("Stratification: Churn")
    print(f"Train saved: {train_path}")
    print(f"Reserved test saved: {test_path}")
    print(f"Manifest saved: {MANIFEST_PATH}")
    print("Holdout test status: RESERVED - DO NOT EVALUATE YET")


if __name__ == "__main__":
    main()
