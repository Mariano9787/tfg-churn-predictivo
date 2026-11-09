PAQUETE FAST-TRACK TFG (superponer a tu proyecto existente)
====================================================
Contiene src/complete_pipeline.py, app/main.py, docs/GUIA_INSTALACION.md,
             docs/NOTAS_TP4.md, tests/test_complete_pipeline.py.
NO BORRA ni reemplaza tus archivos existentes.

EN GIT BASH / PROYECTO (con .venv activado):
1) Descomprimir este ZIP dentro de C:\Users\muria\Projects\tfg-churn-predictivo
2) python -m pytest tests/ -q
3) python -m src.complete_pipeline
4) python -m streamlit run app/main.py

La etapa 3 ejecuta entrenamientos/explicaciones y escribe en artifacts/final/.
El test queda reservado hasta finalizar elección (el código comprueba integridad).
Si un error ocurre tras crear artifacts/final/holdout_seal.json:
   python -m src.complete_pipeline --portfolio-only
NO vuelvas a entrenar con configuraciones diferentes por tu cuenta.
Copiá las últimas 25 líneas de la terminal para corregir el error preservando el protocolo.

IMPORTANTE: No publicar todavía el repo, video ni PDF antes de comprobar ejecución real.
