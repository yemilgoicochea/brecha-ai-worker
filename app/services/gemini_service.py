"""Vertex AI Gemini service for project classification."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class _Label(BaseModel):
    id: int
    label: str
    confianza: float
    justificacion: str


class _GeminiResponse(BaseModel):
    labels: List[_Label]


class GeminiService:

    def __init__(self, supabase_service):
        self._supabase = supabase_service
        self._client = genai.Client(
            vertexai=True,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )
        self._system_instruction: Optional[str] = None
        logger.info(
            f"google-genai client initialized: project={settings.GCP_PROJECT_ID}, "
            f"location={settings.GCP_LOCATION}, model={settings.GEMINI_MODEL_NAME}"
        )

    async def cargar_o_actualizar_catalogo(self) -> None:
        """Carga brechas activas desde la DB y actualiza el system instruction en memoria."""
        logger.info("Cargando catálogo de brechas desde la base de datos...")
        try:
            response = await asyncio.to_thread(
                lambda: (
                    self._supabase.client
                    .table("gap_indicators")
                    .select(
                        "id, indicator_code, name, indicator_type, "
                        "service_name, typology, sectors(name)"
                    )
                    .eq("is_active", True)
                    .order("sector_id")
                    .execute()
                )
            )

            indicators: List[Dict[str, Any]] = response.data or []
            logger.info(f"Cargados {len(indicators)} indicadores de brecha activos")

            catalogo = self._build_catalog_string(indicators)
            self._system_instruction = self._build_system_instruction(catalogo)

            # Resumen por sector para verificar que el catálogo cargó correctamente
            sector_counts: Dict[str, int] = {}
            for ind in indicators:
                sector = (ind.get("sectors") or {}).get("name", "SIN SECTOR")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            for sector, count in sector_counts.items():
                logger.info(f"  Sector '{sector}': {count} indicadores")

            logger.debug("--- CATÁLOGO COMPLETO ---\n%s", catalogo)
            logger.info("System instruction actualizado en memoria con el catálogo fresco")

        except Exception as e:
            logger.error(f"Error al cargar catálogo: {e}", exc_info=True)
            if self._system_instruction is None:
                raise RuntimeError(
                    "No se pudo inicializar el catálogo: fallo en la carga desde la DB"
                ) from e
            logger.warning("Se mantiene el catálogo anterior en memoria")

    def _build_catalog_string(self, indicators: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        current_sector: Optional[str] = None

        for ind in indicators:
            sector_name = (ind.get("sectors") or {}).get("name", "SIN SECTOR")
            if sector_name != current_sector:
                current_sector = sector_name
                lines.append(f"\nSECTOR: {sector_name}")

            lines.append(
                f"  - ID: {ind['id']} | CÓDIGO: {ind.get('indicator_code', 'N/A')} | "
                f"INDICADOR: {ind['name']}\n"
                f"    TIPO: {ind.get('indicator_type', '')} | "
                f"SERVICIO: {ind.get('service_name', '')} | "
                f"TIPOLOGÍA: {ind.get('typology', '')}"
            )

        return "\n".join(lines)

    def _build_system_instruction(self, catalogo: str) -> str:
        return (
            "Eres un experto clasificador de proyectos públicos de Invierte.pe del SNPMGI Perú.\n"
            "Tu única tarea es clasificar el proyecto recibido según el siguiente catálogo oficial "
            "de indicadores de brecha activos:\n\n"
            f"{catalogo}\n\n"
            "Reglas estrictas:\n"
            "1. Devuelve SOLO JSON válido con el esquema indicado, sin texto adicional.\n"
            "2. Asigna una o más categorías si el proyecto cubre múltiples brechas.\n"
            "3. Si el proyecto no coincide con ninguna brecha, usa id=0 y label='NO_CLASIFICADO'.\n"
            "4. El campo 'id' debe ser el ID numérico exacto del catálogo anterior.\n"
            "5. 'confianza' es un float entre 0.0 y 1.0.\n"
            "6. 'justificacion' debe ser breve (máx 2 oraciones) y en español."
        )

    async def classify(
        self, project_title: str, project_description: str = ""
    ) -> Dict[str, Any]:
        if self._system_instruction is None:
            raise RuntimeError(
                "Catálogo no inicializado. Ejecuta cargar_o_actualizar_catalogo() primero."
            )
        if not project_title.strip():
            raise ValueError("El título del proyecto no puede estar vacío")

        config = types.GenerateContentConfig(
            system_instruction=self._system_instruction,
            response_mime_type="application/json",
            response_schema=_GeminiResponse,
        )

        prompt = f"Título del proyecto: {project_title}"
        if project_description:
            prompt += f"\nDescripción: {project_description}"

        for attempt in range(1, settings.GEMINI_MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Intento de clasificación {attempt}/{settings.GEMINI_MAX_RETRIES}"
                )
                response = await self._client.aio.models.generate_content(
                    model=settings.GEMINI_MODEL_NAME,
                    contents=prompt,
                    config=config,
                )
                result = json.loads(response.text)
                logger.info(f"Clasificación exitosa en el intento {attempt}")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON inválido en intento {attempt}: {e}")
                return {
                    "labels": [],
                    "error": "La respuesta del modelo no es JSON válido",
                    "detalle_error": str(e),
                }
            except Exception as e:
                logger.error(f"Intento {attempt} fallido: {e}")
                if attempt < settings.GEMINI_MAX_RETRIES:
                    await asyncio.sleep(settings.GEMINI_RETRY_DELAY)
                else:
                    return {
                        "labels": [],
                        "error": "No se pudo obtener respuesta del modelo.",
                        "detalle_error": str(e),
                    }

        return {"labels": [], "error": "Classification failed after all retries"}
