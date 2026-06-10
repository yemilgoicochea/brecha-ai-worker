"""Data models and schemas for the worker."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClassificationLabel(BaseModel):
    """Classification label with confidence and justification."""

    label: str = Field(..., description="Name of the category")
    id: int = Field(..., description="Identifier of the category")
    confianza: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    justificacion: str = Field(..., description="Justification for the classification")


class ClassificationResult(BaseModel):
    """Result of project classification."""

    labels: List[ClassificationLabel] = Field(default_factory=list, description="List of assigned labels")
    error: Optional[str] = Field(None, description="Error message if classification failed")
    detalle_error: Optional[str] = Field(None, description="Error details")
    raw_response: Optional[str] = Field(None, description="Raw response from model")


class ProjectQueryMessage(BaseModel):
    """Message from Pub/Sub containing project query."""

    query_id: str = Field(..., description="UUID of the project query")
    user_id: str = Field(..., description="UUID of the user")
    title: str = Field(..., description="Project title")
    description: Optional[str] = Field(None, description="Project description")
    zone_type: Optional[str] = Field(None, description="Zona del distrito: 'urbano' o 'rural'")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ProjectClassification(BaseModel):
    """Classification record to save to Supabase."""

    project_query_id: str = Field(..., description="UUID of the project query")
    gap_indicator_id: int = Field(..., description="ID of the gap indicator")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field(...)
    ranking_position: int = Field(...)
    embedding_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    llm_model: str = Field(default="gemini-2.0-flash-exp")


class ProcessingResult(BaseModel):
    """Result of processing a message."""

    success: bool = Field(...)
    query_id: str = Field(...)
    classifications_count: int = Field(default=0)
    error: Optional[str] = Field(None)
    processing_time_ms: int = Field(default=0)
