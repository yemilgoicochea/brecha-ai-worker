"""Inferencia local con PyTorch para pruebas rápidas del clasificador de sectores.

Nota: producción NO usa este script — usa app/services/beto_service.py (ONNX).
Este es solo para probar el modelo localmente (requiere torch + transformers).
"""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PATH_MODEL = Path(__file__).parent.parent / "modelo_beto"
MAX_LEN    = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Cargando modelo desde {PATH_MODEL} (device={DEVICE})...")
tokenizer = AutoTokenizer.from_pretrained(PATH_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(PATH_MODEL)
model.to(DEVICE)
model.eval()

with open(PATH_MODEL / "label_mapping.json", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}


def predict(textos, top_k=3):
    if isinstance(textos, str):
        textos = [textos]
    # El modelo fue entrenado con nombres de proyecto en MAYÚSCULAS (BETO es
    # case-sensitive): texto en minúsculas degrada mucho la predicción.
    textos = [t.upper() for t in textos]
    enc = tokenizer(textos, max_length=MAX_LEN, truncation=True,
                     padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=1)

    resultados = []
    for i, texto in enumerate(textos):
        top = torch.topk(probs[i], k=min(top_k, probs.shape[1]))
        preds = [(id2label[idx.item()], score.item()) for score, idx in zip(top.values, top.indices)]
        resultados.append({"texto": texto, "predicciones": preds})
    return resultados


if __name__ == "__main__":
    ejemplos = [
        "CONSTRUCCION DEL SISTEMA DE AGUA POTABLE Y SANEAMIENTO EN EL CENTRO POBLADO",
        "MEJORAMIENTO DE LA INSTITUCION EDUCATIVA N 123",
        "AMPLIACION DEL SERVICIO DE SALUD EN EL CENTRO DE SALUD",
        "CREACION DE LA CARRETERA VECINAL TRAMO PUENTE - CENTRO POBLADO",
        "INSTALACION DEL SISTEMA DE RIEGO TECNIFICADO",
    ]
    for r in predict(ejemplos):
        print(f"\nTexto: {r['texto']}")
        for label, score in r["predicciones"]:
            print(f"  {label:40s} {score*100:5.2f}%")
