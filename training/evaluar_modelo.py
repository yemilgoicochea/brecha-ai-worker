"""Evalúa el modelo guardado en ../modelo_beto contra el dataset.

Por defecto usa el mismo split de validación que beto_train.py
(random_state=42, stratify), así las métricas son comparables sin
contaminarse con datos de entrenamiento. Con --full evalúa todo el dataset.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE         = Path(__file__).parent
PATH_MODEL   = BASE.parent / "modelo_beto"
PATH_DATASET = BASE / "dataset_limpio.csv"
BATCH_SIZE   = 64
MAX_LEN      = 128
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true",
                    help="Evaluar sobre todo el dataset (incluye datos de entrenamiento)")
args = parser.parse_args()

print(f"Device: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(PATH_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(PATH_MODEL)
model.to(DEVICE)
model.eval()

with open(PATH_MODEL / "label_mapping.json", encoding="utf-8") as f:
    mapping  = json.load(f)
label2id = mapping["label2id"]
id2label = {int(k): v for k, v in mapping["id2label"].items()}

df = pd.read_csv(PATH_DATASET, encoding="utf-8-sig")
df["label"] = df["sector_code"].map(label2id)

if args.full:
    eval_df, split_name = df, "dataset completo"
else:
    # Mismo split que beto_train.py
    _, eval_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    split_name = "split de validación (15%)"

print(f"Evaluando sobre {split_name}: {len(eval_df):,} proyectos")

textos = eval_df["nombre_proyecto"].str.upper().tolist()
y_true = eval_df["label"].tolist()
y_pred = []

with torch.no_grad():
    for i in range(0, len(textos), BATCH_SIZE):
        batch = textos[i:i + BATCH_SIZE]
        enc = tokenizer(batch, max_length=MAX_LEN, truncation=True,
                        padding=True, return_tensors="pt").to(DEVICE)
        preds = torch.argmax(model(**enc).logits, dim=1)
        y_pred.extend(preds.cpu().numpy())
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  {i:,}/{len(textos):,}")

nombres = [id2label[i] for i in sorted(id2label)]
reporte = classification_report(y_true, y_pred, target_names=nombres, digits=3)
print("\n" + reporte)

print("Matriz de confusión (filas=real, columnas=predicho):")
cm = pd.DataFrame(confusion_matrix(y_true, y_pred),
                  index=nombres, columns=[n[:12] for n in nombres])
print(cm.to_string())

sufijo = "full" if args.full else "val"
out = BASE / f"eval_report_{sufijo}.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write(f"Evaluación sobre {split_name}: {len(eval_df):,} proyectos\n\n")
    f.write(reporte + "\n\nMatriz de confusión (filas=real, columnas=predicho):\n")
    f.write(cm.to_string())
print(f"\nReporte guardado en: {out}")
