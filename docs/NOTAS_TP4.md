# Evidencia técnica TP4 — completar solo después de ejecutar

## Metodología
Se conservó el archivo IBM Telco original (7.043 clientes, 21 columnas), se verificó su SHA-256, se caracterizó su calidad y se realizó un EDA estructural/univariado. Se reservó un test estratificado del 20% (1.409 registros); el 80% restante (5.634) se empleó para comparar un clasificador trivial, regresión logística, Random Forest y XGBoost mediante validación cruzada estratificada de cinco folds. La métrica primaria fue Average Precision. El criterio de elección favoreció la regresión logística si quedó dentro de un error estándar del mejor resultado. El tratamiento de valores faltantes, el escalado y la codificación categórica se ajustaron dentro de cada fold mediante Pipeline de scikit-learn.

Se analizó la sensibilidad a género y condición de persona mayor de acuerdo con un criterio conservador predefinido; no se interpretan diferencias de AP entre grupos sin contemplar prevalencias. Antes de abrir el test se guardó `selection_locked.json`. El test se evaluó una sola vez, con AP, ROC-AUC, Brier, calibración y precisión/exhaustividad por deciles. Después de sellar resultados se generaron puntajes OOF sobre la cartera completa con cinco instancias de la misma familia/configuración. Las explicaciones SHAP se consolidaron por variable original y se normalizaron por la suma absoluta de aportes locales.

## Completar con métricas verdaderas
- Familia seleccionada: [tomar de artifacts/final/selection_locked.json]
- Sensibilidad: [tomar de artifacts/final/sensitivity.json]
- AP de test: [tomar de artifacts/final/holdout_metrics.json]
- ROC-AUC de test: [tomar de artifacts/final/holdout_metrics.json]
- Brier de test: [tomar de artifacts/final/holdout_metrics.json]
- Comparación con trivial: [tomar de artifacts/final/holdout_metrics.json]
- Captura por deciles: [tomar de artifacts/final/holdout_deciles.csv]
- Factores: [tomar de artifacts/final/global_factors.csv]

## Trazabilidad id_modelo / instancia
`model_criterion_id` identifica la familia y configuración seleccionada. `fold_id` identifica cuál de las cinco instancias entrenadas fuera de muestra produjo una estimación y su explicación. La explicación lleva además `customerID` y el mismo `fold_id`.

## Límites explícitos
- Dataset transversal de demostración: no hay prueba de eficacia real de retención ni estimación de ROI.
- La evaluación es aleatoria estratificada; no valida estabilidad temporal ni causalidad.
- Los folds de OOF usados para presentación no son una evaluación adicional del test.
- Riesgo es un puntaje; calibración debe evaluarse antes de denominarlo probabilidad.
