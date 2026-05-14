"""Worker service for processing classification jobs."""

import logging
import time
from typing import Any, Dict, List

from app.core.config import settings
from app.models.schemas import ClassificationLabel, ProjectQueryMessage
from app.services.gemini_service import GeminiService
from app.services.pubsub_service import PubSubService
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)


class WorkerService:
    """Service for processing classification jobs from Pub/Sub."""

    def __init__(self):
        """Initialize the worker service."""
        self.pubsub_service = PubSubService()
        self.gemini_service = GeminiService()
        self.supabase_service = SupabaseService()

        logger.info("Worker service initialized")

    def process_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Process a message from Pub/Sub.

        Args:
            message_data: Message data containing project query information

        Returns:
            True if processing was successful
        """
        start_time = time.time()

        try:
            # Parse message
            query_message = ProjectQueryMessage(**message_data)
            logger.info(f"Processing query: {query_message.query_id}")

            # Update status to processing
            self.supabase_service.update_query_status(
                query_message.query_id,
                "processing",
                model_used=settings.GEMINI_MODEL_NAME,
            )

            # Classify the project
            classification_result = self._classify_project(query_message)

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Save classifications if successful
            if classification_result and "labels" in classification_result:
                classifications_to_save = self._prepare_classifications(
                    query_message.query_id,
                    classification_result.get("labels", []),
                )

                success = self.supabase_service.save_classifications(classifications_to_save)

                if not success:
                    logger.error(f"Failed to save classifications for {query_message.query_id}")
                    self.supabase_service.update_query_status(
                        query_message.query_id,
                        "error",
                        processing_time_ms=processing_time_ms,
                        metadata={"error": "Failed to save classifications"},
                    )
                    return False

            # Update status to completed
            success = self.supabase_service.update_query_status(
                query_message.query_id,
                "completed",
                processing_time_ms=processing_time_ms,
                model_used=settings.GEMINI_MODEL_NAME,
                metadata={"classifications_count": len(classification_result.get("labels", []))},
            )

            logger.info(
                f"Successfully processed query {query_message.query_id} "
                f"in {processing_time_ms}ms"
            )

            return success

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)

            # Try to update status if we have query_id
            try:
                query_id = message_data.get("query_id")
                if query_id:
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    self.supabase_service.update_query_status(
                        query_id,
                        "error",
                        processing_time_ms=processing_time_ms,
                        metadata={"error": str(e)},
                    )
            except Exception as inner_e:
                logger.error(f"Failed to update error status: {str(inner_e)}")

            return False

    def _classify_project(self, message: ProjectQueryMessage) -> Dict[str, Any]:
        """
        Classify a project using Gemini.

        Args:
            message: Project query message

        Returns:
            Classification result
        """
        import asyncio

        try:
            # Run async classification in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.gemini_service.classify(
                    project_title=message.title,
                    project_description=message.description or "",
                )
            )
            loop.close()

            return result

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {
                "labels": [],
                "error": str(e),
            }

    def _prepare_classifications(
        self,
        query_id: str,
        labels: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Prepare classifications for saving to Supabase.

        Args:
            query_id: UUID of the project query
            labels: List of labels from classification result

        Returns:
            List of classification records
        """
        classifications = []

        for ranking_position, label_data in enumerate(labels, 1):
            try:
                # Skip unclassified labels
                if label_data.get("id") == 0:
                    continue

                classification = {
                    "project_query_id": query_id,
                    "gap_indicator_id": label_data.get("id"),
                    "confidence_score": float(label_data.get("confianza", 0.0)),
                    "justification": label_data.get("justificacion", ""),
                    "ranking_position": ranking_position,
                    "llm_model": settings.GEMINI_MODEL_NAME,
                }

                classifications.append(classification)

            except Exception as e:
                logger.warning(f"Failed to prepare classification: {str(e)}")
                continue

        return classifications

    def start(self):
        """Start the worker, listening to Pub/Sub messages."""
        logger.info(f"Starting worker service. Listening to {settings.PUBSUB_SUBSCRIPTION_ID}")

        try:
            # Listen to messages with timeout
            self.pubsub_service.listen(
                callback=self.process_message,
                timeout=settings.WORKER_TIMEOUT,
            )
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
        except Exception as e:
            logger.error(f"Worker error: {str(e)}", exc_info=True)
            raise
