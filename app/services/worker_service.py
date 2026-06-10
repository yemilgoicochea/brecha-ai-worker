"""Worker service for processing classification jobs."""

import asyncio
import logging
import time
from typing import Any, Dict, List

from app.core.config import settings
from app.models.schemas import ProjectQueryMessage
from app.services.beto_service import BetoService
from app.services.gemini_service import GeminiService
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)


class WorkerService:
    """Orquesta el pipeline de clasificación para un mensaje de Pub/Sub."""

    def __init__(self, supabase_service: SupabaseService, gemini_service: GeminiService, beto_service: BetoService):
        self.supabase = supabase_service
        self.gemini = gemini_service
        self.beto = beto_service

    async def process_message(self, payload: Dict[str, Any]) -> None:
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

            # ── BETO predice sector ────────────────────────────────────────────
            sector_code, beto_confidence = await asyncio.to_thread(
                self.beto.predict_sector,
                query_message.title,
            )

            # ── Gemini clasifica brecha con catálogo filtrado por sector ───────
            classification_result = await self.gemini.classify(
                project_title=query_message.title,
                project_description=query_message.description or "",
                sector_code=sector_code,
                zone_type=query_message.zone_type,
            )

            # resto igual que antes...
            processing_time_ms = int((time.time() - start_time) * 1000)
            labels = classification_result.get("labels", [])
            classifications = self._prepare_classifications(query_message.query_id, labels)

            if classifications:
                saved = await asyncio.to_thread(
                    self.supabase.save_classifications, classifications
                )
                if not saved:
                    logger.error(f"Fallo al guardar clasificaciones para {query_message.query_id}")
                    await asyncio.to_thread(
                        self.supabase.update_query_status,
                        query_message.query_id, "error",
                        processing_time_ms=processing_time_ms,
                        metadata={"error": "Failed to save classifications"},
                    )
                    return

            classification_status = "classified" if classifications else "unclassified"

            await asyncio.to_thread(
                self.supabase.update_query_status,
                query_message.query_id, "completed",
                processing_time_ms=processing_time_ms,
                model_used=settings.GEMINI_MODEL_NAME,
                classification_status=classification_status,
                metadata={
                    "classifications_count": len(classifications),
                    "predicted_sector": sector_code,
                    "beto_confidence": round(beto_confidence, 4),
                },
            )

            logger.info(
                f"Consulta {query_message.query_id} completada en {processing_time_ms}ms "
                f"sector={sector_code} ({len(classifications)} brechas)"
            )

        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}", exc_info=True)
            if query_id:
                processing_time_ms = int((time.time() - start_time) * 1000)
                await asyncio.to_thread(
                    self.supabase.update_query_status,
                    query_id, "error",
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
