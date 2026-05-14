"""Supabase database service."""

import logging
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from app.core.config import settings

logger = logging.getLogger(__name__)


class SupabaseService:
    """Service for managing Supabase interactions."""

    def __init__(self):
        """Initialize the Supabase service."""
        try:
            self.client: Client = create_client(
                supabase_url=settings.SUPABASE_URL,
                supabase_key=settings.SUPABASE_KEY,
            )
            self.queries_table = settings.SUPABASE_TABLE
            self.classifications_table = settings.SUPABASE_CLASSIFICATIONS_TABLE

            # Test connection
            _ = self.client.table(self.queries_table).select("id").limit(1).execute()
            logger.info("Supabase service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase service: {str(e)}")
            raise

    def update_query_status(
        self,
        query_id: str,
        status: str,
        processing_time_ms: Optional[int] = None,
        model_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update the status of a project query.

        Args:
            query_id: UUID of the project query
            status: New status (pending, processing, completed, error)
            processing_time_ms: Time spent processing in milliseconds
            model_used: Model name used for processing
            metadata: Additional metadata to update

        Returns:
            True if update was successful
        """
        try:
            update_data = {
                "status": status,
                "updated_at": "now()",
            }

            if processing_time_ms is not None:
                update_data["processing_time_ms"] = processing_time_ms

            if model_used:
                update_data["model_used"] = model_used

            if metadata:
                # Merge with existing metadata
                update_data["metadata"] = metadata

            response = (
                self.client.table(self.queries_table)
                .update(update_data)
                .eq("id", query_id)
                .execute()
            )

            logger.info(f"Updated query {query_id} status to {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update query status: {str(e)}")
            return False

    def save_classifications(
        self,
        classifications: List[Dict[str, Any]],
    ) -> bool:
        """
        Save classification results to Supabase.

        Args:
            classifications: List of classification dictionaries

        Returns:
            True if save was successful
        """
        if not classifications:
            logger.warning("No classifications to save")
            return True

        try:
            response = (
                self.client.table(self.classifications_table)
                .insert(classifications)
                .execute()
            )

            logger.info(f"Saved {len(classifications)} classifications")
            return True

        except Exception as e:
            logger.error(f"Failed to save classifications: {str(e)}")
            return False

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a project query by ID.

        Args:
            query_id: UUID of the project query

        Returns:
            Query data or None if not found
        """
        try:
            response = (
                self.client.table(self.queries_table)
                .select("*")
                .eq("id", query_id)
                .single()
                .execute()
            )

            return response.data

        except Exception as e:
            logger.warning(f"Query {query_id} not found: {str(e)}")
            return None

    def batch_update_status(
        self,
        updates: List[Dict[str, Any]],
    ) -> bool:
        """
        Batch update multiple queries' statuses.

        Args:
            updates: List of update dictionaries with 'id' and status fields

        Returns:
            True if all updates were successful
        """
        try:
            for update in updates:
                query_id = update.pop("id")
                self.client.table(self.queries_table).update(update).eq("id", query_id).execute()

            logger.info(f"Batch updated {len(updates)} queries")
            return True

        except Exception as e:
            logger.error(f"Failed to batch update queries: {str(e)}")
            return False
