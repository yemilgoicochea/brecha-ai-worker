"""Vertex AI Gemini service for project classification."""

import json
import logging
import time
from typing import Any, Dict

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiService:

    def __init__(self):
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
        self.model = GenerativeModel(settings.GEMINI_MODEL_NAME)

        self.catalogo_brechas = """
        SECTOR: AGRARIO Y DE RIEGO
        - ID: 1 | CÓDIGO: BRE-5EED76CD | INDICADOR: PORCENTAJE DE SUPERFICIE SIN ACONDICIONAMIENTO PARA RECARGA HÍDRICA PROVENIENTE DE PRECIPITACIÓN
          TIPO: COBERTURA | SERVICIO: SERVICIO DE SIEMBRA Y COSECHA DE AGUA | TIPOLOGÍA: SIEMBRA Y COSECHA DE AGUA
        """

        logger.info(
            f"Vertex AI initialized: project={settings.GCP_PROJECT_ID}, "
            f"location={settings.GCP_LOCATION}, model={settings.GEMINI_MODEL_NAME}"
        )

    def _get_system_instruction(self) -> str:
        return f"""Eres un experto en Invierte.pe. Clasifica el proyecto según este catálogo de brechas:
        {self.catalogo_brechas}
        Responde estrictamente en JSON con el formato:
        {{
          "labels": [
            {{
              "label": "nombre del indicador",
              "id": 1,
              "confianza": 0.95,
              "justificacion": "Razón de la clasificación"
            }}
          ]
        }}
        Reglas:
        - Asigna una o varias categorías si el proyecto cubre más de una brecha.
        - Si no coincide con ninguna categoría usa id 0 y label "NO_CLASIFICADO"."""

    async def classify(self, project_title: str, project_description: str = "") -> Dict[str, Any]:
        if not project_title or not project_title.strip():
            raise ValueError("Project title cannot be empty")

        config = GenerationConfig(response_mime_type="application/json")
        prompt = f"Proyecto: {project_title}. Descripción: {project_description}"

        for attempt in range(1, settings.GEMINI_MAX_RETRIES + 1):
            try:
                logger.info(f"Classification attempt {attempt}/{settings.GEMINI_MAX_RETRIES}")

                response = self.model.generate_content(
                    [self._get_system_instruction(), prompt],
                    generation_config=config,
                )

                result = json.loads(response.text)
                logger.info(f"Classification successful on attempt {attempt}")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on attempt {attempt}: {str(e)}")
                return {"labels": [], "error": "La respuesta del modelo no es JSON válido", "detalle_error": str(e)}
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                if attempt < settings.GEMINI_MAX_RETRIES:
                    time.sleep(settings.GEMINI_RETRY_DELAY)
                else:
                    return {"labels": [], "error": "No se pudo obtener respuesta del modelo.", "detalle_error": str(e)}

        return {"labels": [], "error": "Classification failed after all retries"}
