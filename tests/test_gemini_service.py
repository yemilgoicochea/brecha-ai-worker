"""Unit tests for GeminiService."""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ── classify() ─────────────────────────────────────────────────────────────

async def test_classify_returns_parsed_labels(gemini_service):
    response_json = json.dumps({
        "labels": [
            {"id": 1, "label": "Agua potable", "confianza": 0.92, "justificacion": "Descripción clara."}
        ]
    })
    mock_response = MagicMock()
    mock_response.text = response_json

    with patch("app.services.gemini_service.GenerativeModel") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        result = await gemini_service.classify("Construcción de planta de agua potable")

    assert "labels" in result
    assert len(result["labels"]) == 1
    assert result["labels"][0]["id"] == 1
    assert result["labels"][0]["confianza"] == 0.92


async def test_classify_with_description(gemini_service):
    response_json = json.dumps({"labels": []})
    mock_response = MagicMock(text=response_json)

    with patch("app.services.gemini_service.GenerativeModel") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        result = await gemini_service.classify("Proyecto vial", "Mejora de carreteras")

    assert "labels" in result
    # Verify the prompt passed to generate_content includes the description
    call_args = mock_model.generate_content.call_args
    prompt = call_args.args[0]
    assert "Mejora de carreteras" in prompt


async def test_classify_returns_error_on_invalid_json(gemini_service):
    mock_response = MagicMock()
    mock_response.text = "THIS IS NOT JSON {{"

    with patch("app.services.gemini_service.GenerativeModel") as mock_model_cls:
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        result = await gemini_service.classify("Proyecto de agua")

    assert result["labels"] == []
    assert result["error"] is not None
    assert "JSON" in result["error"]


async def test_classify_retries_on_exception_then_succeeds(gemini_service):
    good_response = MagicMock()
    good_response.text = json.dumps({
        "labels": [{"id": 2, "label": "Vial", "confianza": 0.88, "justificacion": "Pista."}]
    })

    with patch("app.services.gemini_service.GenerativeModel") as mock_model_cls:
        with patch("app.services.gemini_service.asyncio.sleep", new_callable=AsyncMock):
            mock_model = MagicMock()
            # First call fails, second succeeds
            mock_model.generate_content.side_effect = [
                Exception("Transient API error"),
                good_response,
            ]
            mock_model_cls.return_value = mock_model

            result = await gemini_service.classify("Proyecto vial")

    assert len(result["labels"]) == 1
    assert mock_model.generate_content.call_count == 2


async def test_classify_returns_error_after_all_retries_exhausted(gemini_service):
    with patch("app.services.gemini_service.GenerativeModel") as mock_model_cls:
        with patch("app.services.gemini_service.asyncio.sleep", new_callable=AsyncMock):
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("Persistent API error")
            mock_model_cls.return_value = mock_model

            result = await gemini_service.classify("Proyecto cualquiera")

    assert result["labels"] == []
    assert result["error"] is not None
    assert mock_model.generate_content.call_count == gemini_service._supabase.__class__.__mro__[0].__name__ or True
    # Verify all retries were attempted
    from app.core.config import settings
    assert mock_model.generate_content.call_count == settings.GEMINI_MAX_RETRIES


async def test_classify_raises_if_empty_title(gemini_service):
    with pytest.raises(ValueError, match="vacío"):
        await gemini_service.classify("   ")


async def test_classify_raises_if_catalog_not_loaded(supabase_service):
    with patch("app.services.gemini_service.vertexai.init"):
        from app.services.gemini_service import GeminiService
        service = GeminiService(supabase_service)
        # _system_instruction is None (catalog not loaded)

    with pytest.raises(RuntimeError, match="Catálogo no inicializado"):
        await service.classify("Proyecto de agua")


# ── cargar_o_actualizar_catalogo() ─────────────────────────────────────────

async def test_cargar_catalogo_builds_catalog_by_sector(gemini_service, mock_supabase_client, mock_indicators):
    chain = mock_supabase_client.table.return_value
    chain.execute.return_value = MagicMock(data=mock_indicators)

    gemini_service._catalog_by_sector = None

    await gemini_service.cargar_o_actualizar_catalogo()

    assert gemini_service._catalog_by_sector is not None
    # mock_indicators have no "code" field in sectors → indexed under "UNKNOWN"
    indicators = gemini_service._catalog_by_sector.get("UNKNOWN", [])
    assert any("Brecha agua potable" in ind["name"] for ind in indicators)


async def test_cargar_catalogo_raises_on_first_load_failure(supabase_service, mock_supabase_client):
    mock_supabase_client.table.side_effect = Exception("DB unavailable")

    with patch("app.services.gemini_service.vertexai.init"):
        from app.services.gemini_service import GeminiService
        service = GeminiService(supabase_service)
        service._system_instruction = None  # no previous catalog

    with pytest.raises(RuntimeError, match="No se pudo inicializar el catálogo"):
        await service.cargar_o_actualizar_catalogo()


async def test_cargar_catalogo_keeps_old_on_reload_failure(gemini_service, mock_supabase_client):
    mock_supabase_client.table.side_effect = Exception("DB unavailable")
    old_catalog = gemini_service._catalog_by_sector  # pre-loaded in fixture

    await gemini_service.cargar_o_actualizar_catalogo()

    # Should keep the previous catalog instead of raising
    assert gemini_service._catalog_by_sector == old_catalog


async def test_cargar_catalogo_handles_empty_indicators(gemini_service, mock_supabase_client):
    chain = mock_supabase_client.table.return_value
    chain.execute.return_value = MagicMock(data=[])

    await gemini_service.cargar_o_actualizar_catalogo()

    assert gemini_service._catalog_by_sector == {}


# ── _prepare_classifications (via WorkerService) ───────────────────────────

def test_prepare_classifications_filters_no_clasificado(worker_service):
    labels = [
        {"id": 0, "label": "NO_CLASIFICADO", "confianza": 0.3, "justificacion": "No encaja"},
        {"id": 5, "label": "Agua potable", "confianza": 0.9, "justificacion": "Encaja bien"},
    ]

    result = worker_service._prepare_classifications("q1", labels)

    assert len(result) == 1
    assert result[0]["gap_indicator_id"] == 5


def test_prepare_classifications_preserves_ranking_position(worker_service):
    labels = [
        {"id": 1, "label": "Primera brecha", "confianza": 0.9, "justificacion": "J1"},
        {"id": 2, "label": "Segunda brecha", "confianza": 0.7, "justificacion": "J2"},
    ]

    result = worker_service._prepare_classifications("q1", labels)

    assert result[0]["ranking_position"] == 1
    assert result[1]["ranking_position"] == 2


def test_prepare_classifications_empty_labels(worker_service):
    result = worker_service._prepare_classifications("q1", [])
    assert result == []


def test_prepare_classifications_maps_all_fields(worker_service):
    labels = [
        {"id": 3, "label": "Brecha energía", "confianza": 0.85, "justificacion": "Texto"}
    ]

    result = worker_service._prepare_classifications("q1", labels)

    c = result[0]
    assert c["project_query_id"] == "q1"
    assert c["gap_indicator_id"] == 3
    assert c["confidence_score"] == 0.85
    assert c["justification"] == "Texto"
    assert c["ranking_position"] == 1
    assert "llm_model" in c
