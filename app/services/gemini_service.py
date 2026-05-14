"""Vertex AI Gemini service for project classification."""

import json
import logging
import time
from typing import Any, Dict

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from app.core.config import settings
from app.models.categories import DEFINICIONES_DE_CATEGORIAS

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for classifying projects using Vertex AI Gemini."""

    def __init__(self):
        try:
            vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
            self.model = GenerativeModel(
                settings.GEMINI_MODEL_NAME,
                system_instruction=self._build_system_instruction(),
            )
            logger.info(
                f"Vertex AI initialized: project={settings.GCP_PROJECT_ID}, "
                f"location={settings.GCP_LOCATION}, model={settings.GEMINI_MODEL_NAME}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            raise

    def _build_system_instruction(self) -> str:
        category_text_parts = []
        for name, category_info in DEFINICIONES_DE_CATEGORIAS.items():
            category_id = category_info["id"]
            definition = category_info["definicion"]
            category_text_parts.append(
                f"- ID: {category_id}\n"
                f"  NOMBRE: {name}\n"
                f"  DEFINICION: {definition.strip()}\n"
            )
        categories_text = "\n".join(category_text_parts)

        return f"""Eres un modelo de lenguaje experto en clasificación de títulos de proyectos públicos
según brechas de infraestructura y servicios definidas por el SNPMGI del Perú.

Tu única tarea es leer el título del proyecto y asignarle UNA O VARIAS categorías
de servicios publicos de la lista definida abajo.

CATEGORÍAS DISPONIBLES:
{categories_text}

REGLAS ESTRICTAS:
- Analiza el significado del título del proyecto, no solo palabras sueltas.
- Asigna múltiples categorías solo si el título realmente cubre más de una brecha.
- No inventes información adicional que no esté presente o inferida razonablemente del título del proyecto.

FORMATO DE RESPUESTA (OBLIGATORIO):
Responde ÚNICAMENTE con JSON válido, sin texto adicional, sin explicaciones, sin backticks.
Ejemplo de formato:

{{
  "labels": [
    {{
      "label": "NOMBRE_DE_CATEGORIA_1",
      "id": 1,
      "confianza": 0.95,
      "justificacion": "Texto de la justificación"
    }},
    {{
      "label": "NOMBRE_DE_CATEGORIA_2",
      "id": 3,
      "confianza": 0.98,
      "justificacion": "Texto de la justificación"
    }}
  ]
}}

donde:
- "labels" es una lista de objetos.
- "label" es el nombre de la categoría seleccionada.
- "id": es el identificador numérico único asignado a cada categoría.
- "confianza": valor numérico entre 0 y 1.
- "justificacion": explica por qué el proyecto fue clasificado usando la DEFINICIÓN de la categoria.

Si el título no coincide con ninguna definición, devuelve:

{{
  "labels": [
    {{
      "label": "NO_CLASIFICADO",
      "id": 0,
      "confianza": 0.0,
      "justificacion": "El texto no es suficiente o no coincide con ninguna categoría."
    }}
  ]
}}"""

    async def classify(self, project_title: str, project_description: str = "") -> Dict[str, Any]:
        """
        Classify a project using Vertex AI Gemini.

        Returns classification result as dictionary with a "labels" key.
        """
        if not project_title or not project_title.strip():
            raise ValueError("Project title cannot be empty")

        user_prompt = f'TEXTO A CLASIFICAR:\n"{project_title}"'
        if project_description:
            user_prompt += f'\n\nDESCRIPCIÓN ADICIONAL:\n"{project_description}"'

        generation_config = GenerationConfig(response_mime_type="application/json")

        for attempt in range(1, settings.GEMINI_MAX_RETRIES + 1):
            try:
                logger.info(f"Classification attempt {attempt}/{settings.GEMINI_MAX_RETRIES}")

                response = self.model.generate_content(
                    user_prompt,
                    generation_config=generation_config,
                )

                result = json.loads(response.text)
                logger.info(f"Classification successful on attempt {attempt}")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on attempt {attempt}: {str(e)}")
                return {
                    "labels": [],
                    "error": "La respuesta del modelo no es JSON válido",
                    "detalle_error": str(e),
                }
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                if attempt < settings.GEMINI_MAX_RETRIES:
                    time.sleep(settings.GEMINI_RETRY_DELAY)
                else:
                    return {
                        "labels": [],
                        "error": "No se pudo obtener respuesta del modelo luego de varios intentos.",
                        "detalle_error": str(e),
                    }

        return {"labels": [], "error": "Classification failed after all retries"}
