
from pathlib import Path
import hashlib
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "dataset_validation.json"

EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    actual_hash = calculate_sha256(DATASET_PATH)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError("Dataset SHA-256 does not match the documented original.")

    data = pd.read_csv(DATASET_PATH, dtype={"TotalCharges": "string"})

    if data.shape != (7043, 21):
        raise ValueError(f"Unexpected dataset shape: {data.shape}")

    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError("Dataset columns do not match the expected schema.")

    total_charges_numeric = pd.to_numeric(
        data["TotalCharges"].str.strip(), errors="coerce"
    )

    report = {
        "dataset": DATASET_PATH.name,
        "sha256": actual_hash,
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicate_customer_ids": int(data["customerID"].duplicated().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": {
            column: int(value)
            for column, value in data.isna().sum().items()
        },
        "total_charges_non_convertible": int(
            total_charges_numeric.isna().sum()
        ),
        "target_distribution": {
            str(key): int(value)
            for key, value in data["Churn"].value_counts().items()
        },
        "column_types": {
            column: str(dtype)
            for column, dtype in data.dtypes.items()
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("DATASET VALIDATION PASSED")
    print(f"Rows: {report['rows']}")
    print(f"Columns: {report['columns']}")
    print(f"Duplicate customer IDs: {report['duplicate_customer_ids']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(
        "TotalCharges non-convertible values: "
        f"{report['total_charges_non_convertible']}"
    )
    print(f"Target distribution: {report['target_distribution']}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
