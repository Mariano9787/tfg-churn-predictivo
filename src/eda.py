
"""Análisis exploratorio descriptivo previo a la partición.

No utiliza Churn para comparar, seleccionar ni transformar predictores.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "artifacts" / "eda"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def save_histogram(data: pd.DataFrame, column: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        data[column].dropna(),
        bins=30,
        edgecolor="white",
    )

    ax.set_title(f"Distribución de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Cantidad de clientes")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{column}_distribution.png", dpi=150)
    plt.close(fig)


def save_contract_chart(data: pd.DataFrame) -> None:
    counts = data["Contract"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax)

    ax.set_title("Distribución de modalidades contractuales")
    ax.set_xlabel("Modalidad de contrato")
    ax.set_ylabel("Cantidad de clientes")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "contract_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        INPUT_PATH,
        dtype={"TotalCharges": "string"},
    )

    # Copia de trabajo: el CSV original nunca se modifica.
    descriptive_data = data.drop(
        columns=["customerID", "Churn"]
    ).copy()

    descriptive_data["TotalCharges"] = pd.to_numeric(
        descriptive_data["TotalCharges"].str.strip(),
        errors="coerce",
    )

    numeric_summary = descriptive_data[
        NUMERIC_COLUMNS
    ].describe().round(3)

    categorical_columns = [
        column
        for column in descriptive_data.columns
        if column not in NUMERIC_COLUMNS
    ]

    categorical_summary = {}

    for column in categorical_columns:
        counts = descriptive_data[column].value_counts(
            dropna=False
        )

        categorical_summary[column] = {
            str(value): int(count)
            for value, count in counts.items()
        }

    missing_values = {
        column: int(count)
        for column, count in descriptive_data.isna().sum().items()
    }

    report = {
        "analysis": "Pre-split descriptive EDA",
        "rows": int(len(data)),
        "descriptive_predictors": int(
            len(descriptive_data.columns)
        ),
        "numeric_summary": numeric_summary.to_dict(),
        "categorical_distributions": categorical_summary,
        "missing_values": missing_values,
        "target_prevalence_only": {
            str(value): {
                "count": int(count),
                "percentage": round(
                    100 * count / len(data), 2
                ),
            }
            for value, count in data["Churn"].value_counts().items()
        },
        "methodological_scope": (
            "Structural and univariate descriptive analysis only. "
            "No target-conditioned predictor analysis was performed."
        ),
    }

    report_path = OUTPUT_DIR / "eda_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for column in NUMERIC_COLUMNS:
        save_histogram(descriptive_data, column)

    save_contract_chart(descriptive_data)

    print("EDA COMPLETED")
    print(f"Rows: {len(data)}")
    print(f"Descriptive predictors: {len(descriptive_data.columns)}")
    print()
    print("Numeric summary:")
    print(numeric_summary.to_string())
    print()
    print("Missing values:")
    print({
        key: value
        for key, value in missing_values.items()
        if value > 0
    })
    print()
    print("Churn prevalence (descriptive only):")
    print(report["target_prevalence_only"])
    print()
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
