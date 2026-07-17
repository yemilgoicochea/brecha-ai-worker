"""Test fixtures for brecha-ai-worker.

Environment variables are set before any app import so pydantic-settings
can resolve required fields without a real .env file.
"""

import os

# Set required env vars BEFORE importing any app module
os.environ.setdefault("GCP_PROJECT_ID", "test-project-id")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")

import sys
from unittest.mock import MagicMock

# onnxruntime is not available on Windows dev machines (Cloud Run only)
sys.modules.setdefault("onnxruntime", MagicMock())

import pytest
from unittest.mock import patch


# ── SupabaseService fixture ────────────────────────────────────────────────

@pytest.fixture
def mock_supabase_client():
    """Mocked Supabase client that supports method chaining."""
    client = MagicMock()
    chain = MagicMock()
    for method in ["select", "eq", "update", "insert", "limit", "single", "order"]:
        getattr(chain, method).return_value = chain
    chain.execute.return_value = MagicMock(data=[])
    client.table.return_value = chain
    return client


@pytest.fixture
def supabase_service(mock_supabase_client):
    """SupabaseService with mocked client (no real Supabase connection)."""
    with patch("app.services.supabase_service.create_client", return_value=mock_supabase_client):
        from app.services.supabase_service import SupabaseService
        service = SupabaseService()
        service.client = mock_supabase_client
        return service


# ── GeminiService fixture ──────────────────────────────────────────────────

@pytest.fixture
def gemini_service(supabase_service):
    """GeminiService with mocked Vertex AI init and pre-loaded catalog."""
    with patch("app.services.gemini_service.vertexai.init"):
        from app.services.gemini_service import GeminiService
        service = GeminiService(supabase_service)
        # Pre-load catalog so classify() doesn't raise RuntimeError.
        # Use "" as the default sector_code (tests that omit sector_code use this key).
        mock_indicator = {
            "id": 1,
            "indicator_code": "BRE-001",
            "name": "Brecha agua potable",
            "indicator_type": "COBERTURA",
            "service_name": "Agua",
            "typology": "Red pública",
            "sectors": {"name": "Agua y Saneamiento"},
        }
        service._catalog_by_sector = {"": [mock_indicator]}
        return service


# ── BetoService fixture ────────────────────────────────────────────────────

@pytest.fixture
def beto_service():
    """Mocked BetoService (onnxruntime not available on Windows dev machines)."""
    mock = MagicMock()
    mock.predict_sector = MagicMock(return_value=("UNKNOWN", 0.95))
    return mock


# ── WorkerService fixture ──────────────────────────────────────────────────

@pytest.fixture
def worker_service(supabase_service, gemini_service, beto_service):
    """WorkerService wired with mocked Supabase, Gemini, and BETO."""
    from app.services.worker_service import WorkerService
    return WorkerService(supabase_service, gemini_service, beto_service)


# ── Common test data ───────────────────────────────────────────────────────

@pytest.fixture
def sample_message():
    return {
        "query_id": "test-query-uuid-1234",
        "user_id": "user-uuid-5678",
        "title": "Construcción de sistema de agua potable en comunidad rural",
        "description": "Mejora del acceso al agua potable",
        "metadata": {},
    }


@pytest.fixture
def mock_classification_result():
    return {
        "labels": [
            {
                "id": 5,
                "label": "servicio de agua potable mediante red publica",
                "confianza": 0.95,
                "justificacion": "El proyecto describe explícitamente agua potable.",
            }
        ]
    }


@pytest.fixture
def mock_indicators():
    return [
        {
            "id": 1,
            "indicator_code": "BRE-001",
            "name": "Brecha agua potable",
            "indicator_type": "COBERTURA",
            "service_name": "Agua",
            "typology": "Red pública",
            "sectors": {"name": "Agua y Saneamiento"},
        },
        {
            "id": 2,
            "indicator_code": "BRE-002",
            "name": "Brecha alcantarillado",
            "indicator_type": "COBERTURA",
            "service_name": "Saneamiento",
            "typology": "Red pública",
            "sectors": {"name": "Agua y Saneamiento"},
        },
    ]
