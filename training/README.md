# Training — Clasificador de sectores (BETO)

Scripts con los que se entrenó y evaluó el modelo que corre en producción
(`app/services/beto_service.py` vía ONNX). **Nada de esta carpeta se usa en
el despliegue** — está excluida en `.dockerignore` y es solo para
reproducibilidad y pruebas locales.

## Pipeline

```
preparar_dataset.py   →  dataset_limpio.csv (14,334 proyectos, 10 sectores)
beto_train.py         →  ../modelo_beto/ (fine-tuning de dccuchile/bert-base-spanish-wwm-cased)
evaluar_modelo.py     →  eval_report_val.txt (métricas sobre split de validación)
beto_predict.py       →  inferencia local con PyTorch (función predict())
pruebas_modelo.ipynb  →  notebook para pruebas rápidas interactivas
```

## Requisitos locales

`torch`, `transformers`, `scikit-learn`, `pandas` (GPU recomendada para entrenar;
para predecir basta CPU). Los pesos `../modelo_beto/model.safetensors` (~420MB)
no están en git: descárgalos de `gs://brecha-ml-models/modelo_beto/model.safetensors`.

## Métricas del modelo actual (val split, 2,151 proyectos)

- Accuracy: **97.0%** | Macro F1: 0.92
- Clases débiles: `WOMEN_AND_VULNERABLE_POPULATIONS` (F1 0.67, solo 74 ejemplos)
  y `ENVIRONMENT` (F1 0.84). Detalle en `eval_report_val.txt`.

## Importante: normalización a MAYÚSCULAS

El dataset de entrenamiento tiene los nombres de proyecto en MAYÚSCULAS y BETO
es case-sensitive. Todo texto debe pasarse a `.upper()` antes de tokenizar
(ya lo hacen `beto_predict.py` y `beto_service.py`); texto en minúsculas
sesga fuertemente hacia `HOUSING_CONSTRUCTION_AND_SANITATION`.

## Reentrenar y desplegar

1. Regenerar dataset si hay datos nuevos: `python preparar_dataset.py`
2. Entrenar: `python beto_train.py` (guarda el mejor checkpoint en `../modelo_beto/`)
3. Evaluar: `python evaluar_modelo.py`
4. Subir pesos: `gcloud storage cp ../modelo_beto/model.safetensors gs://brecha-ml-models/modelo_beto/model.safetensors`
5. Commit + push → GitHub Actions descarga los pesos de GCS, convierte a ONNX y despliega.
