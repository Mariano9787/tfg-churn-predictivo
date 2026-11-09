# Guía de instalación y ejecución del prototipo

## Requisitos
- Windows 10/11, Python 3.12, Git; conexión a Internet para instalar dependencias.
- Ejecutar localmente. No es un servicio productivo ni un servidor externo.

## Preparación
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```
En PowerShell, activar con `.venv\Scripts\Activate.ps1` (según política de ejecución).

## Datos
Descargar el CSV público de IBM, conservarlo sin cambios en `data/raw/Telco-Customer-Churn.csv`:
```bash
curl -fL "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" -o data/raw/Telco-Customer-Churn.csv
```
SHA-256 esperado: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
Ver `docs/dataset_provenance.md`. Datos de demostración: no representan una compañía real.

## Ejecución desde la raíz
```bash
python -m src.validate_dataset
python -m pytest tests/ -q
python -m src.eda
python -m src.split_data
python -m src.complete_pipeline
    # Si se interrumpe DESPUÉS de sellar el test, usar:
    # python -m src.complete_pipeline --portfolio-only
python -m streamlit run app/main.py
```
El navegador abre la aplicación local. Las cuatro vistas leen artefactos previamente calculados, sin entrenar ni generar nuevos puntajes.

## Artefactos
- `artifacts/final/cv_models.csv`: comparación CV, entrenamiento únicamente.
- `artifacts/final/selection_locked.json`: criterio congelado antes del test.
- `artifacts/final/holdout_metrics.json`, `holdout_deciles.csv`, `holdout_calibration.csv`: desempeño del test final.
- `artifacts/final/holdout_seal.json`: hashes de la evaluación final.
- `artifacts/final/portfolio_oof.parquet`, `explanations_oof.parquet`: salidas de presentación de la cartera completa.
- `artifacts/final/portfolio_metadata.json`: umbrales de tramos, criterio de modelo y espacio SHAP.
- `artifacts/final/selected_model.joblib`: modelo seleccionado ajustado con el entrenamiento.
- `artifacts/final/environment.json`: entorno de ejecución.

## Limitaciones
No se afirma que las estimaciones predigan abandono futuro sin una ventana temporal definida. Las contribuciones SHAP son asociaciones predictivas, no causas. Las salidas OOF de cartera no reemplazan la evaluación en el test reservado. Los puntajes no se interpretan como probabilidades bien calibradas sin evidencia empírica.
