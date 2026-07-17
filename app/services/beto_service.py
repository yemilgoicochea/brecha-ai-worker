"""BETO sector classifier — ONNX Runtime inference (sin PyTorch en producción)."""

import json
import logging
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.core.config import settings

logger = logging.getLogger(__name__)


class BetoService:

    def __init__(self):
        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer = None
        self._label2id: dict = {}
        self._id2label: dict = {}

    def load(self) -> None:
        model_dir = settings.BETO_ONNX_MODEL_DIR
        logger.info(f"Cargando BETO-ONNX desde '{model_dir}'...")
        try:
            logger.info("BETO-ONNX [1/3] Cargando tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True
            )
            logger.info("BETO-ONNX [2/3] Iniciando sesión ONNX Runtime...")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # ORT_ENABLE_BASIC: aplica optimizaciones seguras → inferencia rápida
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            model_path = os.path.join(model_dir, "model.onnx")
            logger.info(f"BETO-ONNX [2/3] Cargando model.onnx ({os.path.getsize(model_path) // 1024 // 1024}MB)...")
            self._session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("BETO-ONNX [2/3] Sesión creada. Ejecutando warmup...")
            dummy = self._tokenizer("test", max_length=128, truncation=True, padding="max_length", return_tensors="np")
            self._session.run(None, dict(dummy))
            logger.info("BETO-ONNX [2/3] Warmup completado — kernels compilados")
            mapping_path = os.path.join(model_dir, "label_mapping.json")
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            self._label2id = mapping["label2id"]
            self._id2label = {int(k): v for k, v in mapping["id2label"].items()}
            logger.info(f"BETO-ONNX [3/3] Listo. Sectores: {list(self._label2id.keys())}")
        except Exception as e:
            logger.error(f"Error cargando BETO-ONNX: {e}", exc_info=True)
            raise

    def predict_sector(self, project_title: str) -> tuple[str, float]:
        if self._session is None:
            raise RuntimeError("BetoService no inicializado. Llama load() primero.")
        logger.info(f"BETO-ONNX inferencia: '{project_title[:60]}'")

        # El modelo fue entrenado con títulos en MAYÚSCULAS (BETO es case-sensitive):
        # sin esta normalización, texto en minúsculas sesga hacia HOUSING/CONSTRUCTION.
        project_title = project_title.upper()

        inputs = self._tokenizer(
            project_title,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        logits = self._session.run(None, dict(inputs))[0]  # shape (1, num_labels)
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        predicted_id = int(np.argmax(probs))
        confidence = float(probs[0, predicted_id])

        sector_code = self._id2label[predicted_id]
        logger.info(f"BETO-ONNX → sector='{sector_code}' (confianza={confidence:.2%})")
        return sector_code, confidence
