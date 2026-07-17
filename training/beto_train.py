"""Entrenamiento del clasificador de sectores (fine-tuning de BETO).

Genera los pesos que viven en gs://brecha-ml-models/modelo_beto/model.safetensors.
Tras reentrenar: subir el nuevo model.safetensors a GCS para que el CI lo use.
Requiere GPU (entrenado originalmente en una RTX 3060, ~3 epochs).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

MODEL_NAME   = "dccuchile/bert-base-spanish-wwm-cased"
BASE         = Path(__file__).parent
PATH_DATASET = BASE / "dataset_limpio.csv"
PATH_OUTPUT  = str(BASE.parent / "modelo_beto")
BATCH_SIZE   = 32
EPOCHS       = 3
MAX_LEN      = 128
LR           = 2e-5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando: {DEVICE.upper()}")

# ── 1. DATASET ─────────────────────────────────────────────────────────────────
df = pd.read_csv(PATH_DATASET, encoding="utf-8-sig")
print(f"Total proyectos: {len(df):,}")

labels_unicos = sorted(df["sector_code"].unique())
label2id = {l: i for i, l in enumerate(labels_unicos)}
id2label = {i: l for l, i in label2id.items()}
NUM_LABELS = len(labels_unicos)
print(f"Total sectores: {NUM_LABELS}")

df["label"] = df["sector_code"].map(label2id)

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["label"]
)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

# ── 2. CLASS WEIGHTS ───────────────────────────────────────────────────────────
clases = np.array(sorted(df["label"].unique()))
pesos  = compute_class_weight("balanced", classes=clases, y=df["label"].values)
pesos_tensor = torch.tensor(pesos, dtype=torch.float).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=pesos_tensor)
print(f"Class weights aplicados")

# ── 3. TOKENIZER Y DATASET CLASS ──────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SectorDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["nombre_proyecto"].tolist()
        self.labels = df["label"].tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label":          torch.tensor(self.labels[idx], dtype=torch.long)}

train_loader = DataLoader(SectorDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SectorDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False)

# ── 4. MODELO ──────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
model.to(DEVICE)

optimizer   = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

# ── 5. ENTRENAMIENTO ───────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labs = batch["label"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            total_loss += loss_fn(out.logits, labs).item()
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(loader), acc, all_preds, all_labels

best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if step % 50 == 0:
            print(f"  Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    val_loss, val_acc, preds, true_labels = evaluate(val_loader)
    print(f"\nEpoch {epoch+1} completo:")
    print(f"  Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  Val Acc:    {val_acc:.4f}\n")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(PATH_OUTPUT)
        tokenizer.save_pretrained(PATH_OUTPUT)
        print(f"  Mejor modelo guardado (acc={val_acc:.4f})")

# ── 6. REPORTE FINAL ───────────────────────────────────────────────────────────
_, _, preds, true_labels = evaluate(val_loader)
labels_presentes  = sorted(set(true_labels))
nombres_presentes = [id2label[i] for i in labels_presentes]
reporte = classification_report(true_labels, preds,
           target_names=nombres_presentes, labels=labels_presentes)
print("\nReporte final:")
print(reporte)

os.makedirs(PATH_OUTPUT, exist_ok=True)
with open(os.path.join(PATH_OUTPUT, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Mejor val_acc: {best_val_acc:.4f}\n\n{reporte}")

with open(os.path.join(PATH_OUTPUT, "label_mapping.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
print(f"\nModelo guardado en: {PATH_OUTPUT}")
