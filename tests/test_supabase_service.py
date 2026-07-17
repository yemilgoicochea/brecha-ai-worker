"""Unit tests for SupabaseService."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def svc(mock_supabase_client):
    """Fresh SupabaseService with mocked client for each test."""
    with patch("app.services.supabase_service.create_client", return_value=mock_supabase_client):
        from app.services.supabase_service import SupabaseService
        service = SupabaseService()
        service.client = mock_supabase_client
        mock_supabase_client.reset_mock()  # clear health-check call from __init__
        return service


# ── update_query_status ────────────────────────────────────────────────────

def test_update_status_to_processing(svc, mock_supabase_client):
    chain = mock_supabase_client.table.return_value
    chain.execute.return_value = MagicMock(data=[{"id": "q1", "status": "processing"}])

    result = svc.update_query_status("q1", "processing")

    assert result is True
    mock_supabase_client.table.assert_called_with(svc.queries_table)
    chain.update.assert_called_once()
    update_payload = chain.update.call_args.args[0]
    assert update_payload["status"] == "processing"


def test_update_status_includes_processing_time(svc, mock_supabase_client):
    chain = mock_supabase_client.table.return_value

    svc.update_query_status("q1", "completed", processing_time_ms=1500)

    payload = chain.update.call_args.args[0]
    assert payload["processing_time_ms"] == 1500


def test_update_status_includes_model_used(svc, mock_supabase_client):
    chain = mock_supabase_client.table.return_value

    svc.update_query_status("q1", "completed", model_used="gemini-2.0")

    payload = chain.update.call_args.args[0]
    assert payload["model_used"] == "gemini-2.0"


def test_update_status_includes_metadata(svc, mock_supabase_client):
    chain = mock_supabase_client.table.return_value

    svc.update_query_status("q1", "error", metadata={"error": "Timeout"})

    payload = chain.update.call_args.args[0]
    assert payload["metadata"] == {"error": "Timeout"}


def test_update_status_returns_false_on_exception(svc, mock_supabase_client):
    mock_supabase_client.table.side_effect = Exception("DB connection error")

    result = svc.update_query_status("q1", "error")

    assert result is False


# ── save_classifications ───────────────────────────────────────────────────

def test_save_classifications_success(svc, mock_supabase_client):
    chain = mock_supabase_client.table.return_value
    chain.execute.return_value = MagicMock(data=[{"id": 1}])

    classifications = [
        {
            "project_query_id": "q1",
            "gap_indicator_id": 5,
            "confidence_score": 0.95,
            "justification": "Justificación",
            "ranking_position": 1,
            "llm_model": "gemini-2.0",
        }
    ]

    result = svc.save_classifications(classifications)

    assert result is True
    mock_supabase_client.table.assert_called_with(svc.classifications_table)
    chain.insert.assert_called_once_with(classifications)


def test_save_classifications_empty_list_skips_insert(svc, mock_supabase_client):
    result = svc.save_classifications([])

    assert result is True
    mock_supabase_client.table.assert_not_called()


def test_save_classifications_returns_false_on_exception(svc, mock_supabase_client):
    mock_supabase_client.table.side_effect = Exception("Insert failed")

    result = svc.save_classifications([{"project_query_id": "q1"}])

    assert result is False


def test_save_multiple_classifications(svc, mock_supabase_client):
    chain = mock_supabase_client.table.return_value
    chain.execute.return_value = MagicMock(data=[{}, {}])

    classifications = [
        {"project_query_id": "q1", "gap_indicator_id": 1, "confidence_score": 0.9,
         "justification": "J1", "ranking_position": 1, "llm_model": "gemini"},
        {"project_query_id": "q1", "gap_indicator_id": 2, "confidence_score": 0.8,
         "justification": "J2", "ranking_position": 2, "llm_model": "gemini"},
    ]

    result = svc.save_classifications(classifications)

    assert result is True
    chain.insert.assert_called_once_with(classifications)


# ── get_query ──────────────────────────────────────────────────────────────

def test_get_query_found(svc, mock_supabase_client):
    query_data = {"id": "q1", "status": "pending", "title": "Proyecto"}
    chain = mock_supabase_client.table.return_value
    chain.execute.return_value = MagicMock(data=query_data)

    result = svc.get_query("q1")

    assert result == query_data
    chain.eq.assert_called_with("id", "q1")


def test_get_query_not_found_returns_none(svc, mock_supabase_client):
    mock_supabase_client.table.side_effect = Exception("Not found")

    result = svc.get_query("nonexistent-id")

    assert result is None
