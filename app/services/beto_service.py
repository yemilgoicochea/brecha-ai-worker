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

        # Limitar threads OMP/MKL antes de cualquier operación torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # ya inicializado en otro punto del proceso

        try:
            logger.info("BETO [1/4] Cargando tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(settings.BETO_MODEL_DIR)
            logger.info("BETO [2/4] Cargando pesos del modelo (420 MB)...")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                settings.BETO_MODEL_DIR
            )
            logger.info("BETO [3/4] Moviendo modelo a CPU y poniendo en modo eval...")
            self._model.to(self._device)
            self._model.eval()

            mapping_path = f"{settings.BETO_MODEL_DIR}/label_mapping.json"
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            self._label2id = mapping["label2id"]
            self._id2label = {int(k): v for k, v in mapping["id2label"].items()}
            logger.info(f"BETO cargado. Sectores: {list(self._label2id.keys())}")

            # Warmup: fuerza la inicialización lazy de torch antes de la primera request real
            logger.info("BETO [4/4] Ejecutando warmup de inferencia...")
            dummy = self._tokenizer("test", max_length=128, truncation=True, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                self._model(**dummy)
            logger.info("BETO listo — warmup completado")
        except Exception as e:
            logger.error(f"Error cargando BETO: {e}", exc_info=True)
            raise

    def predict_sector(self, project_title: str) -> tuple[str, float]:
        """Predice el sector de un proyecto. Retorna (sector_code, confianza)."""
        if self._model is None:
            raise RuntimeError("BetoService no inicializado. Llama load() primero.")
        logger.info(f"BETO iniciando inferencia para: '{project_title[:60]}'")

        logger.info("BETO tokenizando texto...")
        inputs = self._tokenizer(
            project_title,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        logger.info("BETO ejecutando forward pass (model inference)...")
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            confidence = float(probs.max().item())
            predicted_id = int(torch.argmax(logits, dim=1).item())

        sector_code = self._id2label[predicted_id]
        logger.info(f"BETO → sector='{sector_code}' (confianza={confidence:.2%})")
        return sector_code, confidence