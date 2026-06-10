"""BETO sector classifier service."""

import json
import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings

logger = logging.getLogger(__name__)


class BetoService:

    def __init__(self):
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer = None
        self._label2id: dict = {}
        self._id2label: dict = {}
        self._device = "cpu"  # Cloud Run no tiene GPU

    def load(self) -> None:
        logger.info(f"Cargando modelo BETO desde '{settings.BETO_MODEL_DIR}'...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.BETO_MODEL_DIR)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                settings.BETO_MODEL_DIR
            )
            self._model.to(self._device)
            self._model.eval()

            mapping_path = f"{settings.BETO_MODEL_DIR}/label_mapping.json"
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            self._label2id = mapping["label2id"]
            self._id2label = {int(k): v for k, v in mapping["id2label"].items()}

            logger.info(f"BETO cargado. Sectores: {list(self._label2id.keys())}")
        except Exception as e:
            logger.error(f"Error cargando BETO: {e}", exc_info=True)
            raise

    def predict_sector(self, project_title: str) -> tuple[str, float]:
        """Predice el sector de un proyecto. Retorna (sector_code, confianza)."""
        if self._model is None:
            raise RuntimeError("BetoService no inicializado. Llama load() primero.")

        inputs = self._tokenizer(
            project_title,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            confidence = float(probs.max().item())
            predicted_id = int(torch.argmax(logits, dim=1).item())

        sector_code = self._id2label[predicted_id]
        logger.info(f"BETO → sector='{sector_code}' (confianza={confidence:.2%})")
        return sector_code, confidence