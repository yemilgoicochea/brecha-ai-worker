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

            logger.info(f"[PASO 1/5] Marcando consulta {query_message.query_id} como 'processing'...")
            await asyncio.to_thread(
                self.supabase.update_query_status,
                query_message.query_id,
                "processing",
                model_used=settings.GEMINI_MODEL_NAME,
            )
            logger.info(f"[PASO 1/5] OK — estado actualizado a 'processing'")

            # ── BETO predice sector ────────────────────────────────────────────
            logger.info(f"[PASO 2/5] Iniciando BETO inference para título: '{query_message.title[:80]}'")
            sector_code, beto_confidence = await asyncio.wait_for(
                asyncio.to_thread(self.beto.predict_sector, query_message.title),
                timeout=120,
            )
            logger.info(f"[PASO 2/5] OK — BETO: sector='{sector_code}', confianza={beto_confidence:.2%}")

            # ── Gemini clasifica brecha con catálogo filtrado por sector ───────
            logger.info(f"[PASO 3/5] Iniciando Gemini classify para sector='{sector_code}', zone_type='{query_message.zone_type}'")
            classification_result = await asyncio.wait_for(
                self.gemini.classify(
                    project_title=query_message.title,
                    project_description=query_message.description or "",
                    sector_code=sector_code,
                    zone_type=query_message.zone_type,
                ),
                timeout=120,
            )
            labels = classification_result.get("labels", [])
            logger.info(f"[PASO 3/5] OK — Gemini retornó {len(labels)} etiqueta(s)")

            processing_time_ms = int((time.time() - start_time) * 1000)
            classifications = self._prepare_classifications(query_message.query_id, labels)

            if classifications:
                logger.info(f"[PASO 4/5] Guardando {len(classifications)} clasificación(es) en Supabase...")
                saved = await asyncio.to_thread(
                    self.supabase.save_classifications, classifications
                )
                if not saved:
                    logger.error(f"[PASO 4/5] FALLO — no se pudieron guardar clasificaciones para {query_message.query_id}")
                    await asyncio.to_thread(
                        self.supabase.update_query_status,
                        query_message.query_id, "error",
                        processing_time_ms=processing_time_ms,
                        metadata={"error": "Failed to save classifications"},
                    )
                    return
                logger.info(f"[PASO 4/5] OK — clasificaciones guardadas")
            else:
                logger.info(f"[PASO 4/5] Sin clasificaciones válidas (NO_CLASIFICADO)")

            classification_status = "classified" if classifications else "unclassified"

            logger.info(f"[PASO 5/5] Marcando consulta como 'completed' (classification_status={classification_status})...")
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
                f"[PASO 5/5] OK — Consulta {query_message.query_id} completada en {processing_time_ms}ms | "
                f"sector={sector_code} | brechas={len(classifications)} | status={classification_status}"
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
