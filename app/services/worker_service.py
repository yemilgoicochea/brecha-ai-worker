"""Worker service for processing classification jobs."""

import asyncio
import logging
import time
from typing import Any, Dict, List

from app.core.config import settings
from app.models.schemas import ProjectQueryMessage
from app.services.gemini_service import GeminiService
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)


class WorkerService:
    """Orquesta el pipeline de clasificación para un mensaje de Pub/Sub."""

    def __init__(self, supabase_service: SupabaseService, gemini_service: GeminiService):
        self.supabase = supabase_service
        self.gemini = gemini_service

    async def process_message(self, payload: Dict[str, Any]) -> None:
        """
        Procesa un mensaje de Pub/Sub de principio a fin.
        Actualiza el estado en Supabase tanto en éxito como en error.
        No lanza excepciones — el caller siempre puede hacer ack().
        """
        start_time = time.time()
        query_id: str = payload.get("query_id", "")

        try:
            query_message = ProjectQueryMessage(**payload)
            logger.info(f"Procesando consulta: {query_message.query_id}")

            await asyncio.to_thread(
                self.supabase.update_query_status,
                query_message.query_id,
                "processing",
                model_used=settings.GEMINI_MODEL_NAME,
            )

            classification_result = await self.gemini.classify(
                project_title=query_message.title,
                project_description=query_message.description or "",
            )

            processing_time_ms = int((time.time() - start_time) * 1000)

            labels = classification_result.get("labels", [])
            classifications = self._prepare_classifications(
                query_message.query_id, labels
            )

            if classifications:
                saved = await asyncio.to_thread(
                    self.supabase.save_classifications, classifications
                )
                if not saved:
                    logger.error(
                        f"Fallo al guardar clasificaciones para {query_message.query_id}"
                    )
                    await asyncio.to_thread(
                        self.supabase.update_query_status,
                        query_message.query_id,
                        "error",
                        processing_time_ms=processing_time_ms,
                        metadata={"error": "Failed to save classifications"},
                    )
                    return
            else:
                logger.info(
                    f"Consulta {query_message.query_id}: sin brechas coincidentes "
                    f"(NO_CLASIFICADO o lista vacía)"
                )

            await asyncio.to_thread(
                self.supabase.update_query_status,
                query_message.query_id,
                "completed",
                processing_time_ms=processing_time_ms,
                model_used=settings.GEMINI_MODEL_NAME,
                metadata={"classifications_count": len(classifications)},
            )

            logger.info(
                f"Consulta {query_message.query_id} completada en {processing_time_ms}ms "
                f"({len(classifications)} brechas clasificadas)"
            )

        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}", exc_info=True)
            if query_id:
                processing_time_ms = int((time.time() - start_time) * 1000)
                await asyncio.to_thread(
                    self.supabase.update_query_status,
                    query_id,
                    "error",
                    processing_time_ms=processing_time_ms,
                    metadata={"error": str(e)},
                )

    def _prepare_classifications(
        self, query_id: str, labels: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        classifications = []
        for position, label_data in enumerate(labels, 1):
            if label_data.get("id") == 0:
                continue
            try:
                classifications.append(
                    {
                        "project_query_id": query_id,
                        "gap_indicator_id": label_data["id"],
                        "confidence_score": float(label_data.get("confianza", 0.0)),
                        "justification": label_data.get("justificacion", ""),
                        "ranking_position": position,
                        "llm_model": settings.GEMINI_MODEL_NAME,
                    }
                )
            except Exception as e:
                logger.warning(f"Error preparando clasificación: {e}")
        return classifications
