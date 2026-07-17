"""Unit tests for WorkerService.process_message pipeline."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call


# ── Happy path ─────────────────────────────────────────────────────────────

async def test_process_message_happy_path(worker_service, supabase_service, gemini_service, sample_message, mock_classification_result):
    gemini_service.classify = AsyncMock(return_value=mock_classification_result)
    supabase_service.update_query_status = MagicMock(return_value=True)
    supabase_service.save_classifications = MagicMock(return_value=True)

    await worker_service.process_message(sample_message)

    # Status updated twice: processing → completed
    assert supabase_service.update_query_status.call_count == 2
    calls = supabase_service.update_query_status.call_args_list
    assert calls[0].args[1] == "processing"
    assert calls[1].args[1] == "completed"

    # Classifications saved once
    supabase_service.save_classifications.assert_called_once()
    saved = supabase_service.save_classifications.call_args.args[0]
    assert len(saved) == 1
    assert saved[0]["gap_indicator_id"] == 5
    assert saved[0]["confidence_score"] == 0.95


async def test_process_message_uses_correct_query_id(worker_service, supabase_service, gemini_service, sample_message, mock_classification_result):
    gemini_service.classify = AsyncMock(return_value=mock_classification_result)
    supabase_service.update_query_status = MagicMock(return_value=True)
    supabase_service.save_classifications = MagicMock(return_value=True)

    await worker_service.process_message(sample_message)

    query_id = sample_message["query_id"]
    first_call_id = supabase_service.update_query_status.call_args_list[0].args[0]
    assert first_call_id == query_id


async def test_process_message_includes_processing_time(worker_service, supabase_service, gemini_service, sample_message, mock_classification_result):
    gemini_service.classify = AsyncMock(return_value=mock_classification_result)
    supabase_service.update_query_status = MagicMock(return_value=True)
    supabase_service.save_classifications = MagicMock(return_value=True)

    await worker_service.process_message(sample_message)

    # The completed call should include processing_time_ms
    completed_call = supabase_service.update_query_status.call_args_list[1]
    assert completed_call.kwargs.get("processing_time_ms") is not None
    assert completed_call.kwargs["processing_time_ms"] >= 0


# ── Gemini failure ─────────────────────────────────────────────────────────

async def test_process_message_gemini_returns_error_dict(worker_service, supabase_service, gemini_service, sample_message):
    gemini_service.classify = AsyncMock(return_value={
        "labels": [],
        "error": "No se pudo obtener respuesta del modelo.",
        "detalle_error": "API timeout",
    })
    supabase_service.update_query_status = MagicMock(return_value=True)
    supabase_service.save_classifications = MagicMock(return_value=True)

    await worker_service.process_message(sample_message)

    # With empty labels, completed with 0 classifications (not an error)
    calls = supabase_service.update_query_status.call_args_list
    assert calls[-1].args[1] == "completed"
    supabase_service.save_classifications.assert_not_called()


async def test_process_message_gemini_raises_exception(worker_service, supabase_service, gemini_service, sample_message):
    gemini_service.classify = AsyncMock(side_effect=Exception("Gemini crashed"))
    supabase_service.update_query_status = MagicMock(return_value=True)

    await worker_service.process_message(sample_message)

    # Should not raise — must update status to error
    calls = supabase_service.update_query_status.call_args_list
    statuses = [c.args[1] for c in calls]
    assert "error" in statuses


# ── Supabase save failure ──────────────────────────────────────────────────

async def test_process_message_save_classifications_fails(worker_service, supabase_service, gemini_service, sample_message, mock_classification_result):
    gemini_service.classify = AsyncMock(return_value=mock_classification_result)
    supabase_service.update_query_status = MagicMock(return_value=True)
    supabase_service.save_classifications = MagicMock(return_value=False)

    await worker_service.process_message(sample_message)

    # When save fails, status must end as error
    calls = supabase_service.update_query_status.call_args_list
    statuses = [c.args[1] for c in calls]
    assert "error" in statuses


# ── NO_CLASIFICADO labels ──────────────────────────────────────────────────

async def test_process_message_no_clasificado_label_skipped(worker_service, supabase_service, gemini_service, sample_message):
    gemini_service.classify = AsyncMock(return_value={
        "labels": [
            {"id": 0, "label": "NO_CLASIFICADO", "confianza": 0.2, "justificacion": "No encaja"}
        ]
    })
    supabase_service.update_query_status = MagicMock(return_value=True)
    supabase_service.save_classifications = MagicMock(return_value=True)

    await worker_service.process_message(sample_message)

    # id=0 must be filtered: save_classifications not called
    supabase_service.save_classifications.assert_not_called()

    # Pipeline still completes successfully
    calls = supabase_service.update_query_status.call_args_list
    assert calls[-1].args[1] == "completed"


async def test_process_message_mixed_labels_valid_and_no_clasificado(worker_service, supabase_service, gemini_service, sample_message):
    gemini_service.classify = AsyncMock(return_value={
        "labels": [
            {"id": 0, "label": "NO_CLASIFICADO", "confianza": 0.2, "justificacion": "No encaja"},
            {"id": 3, "label": "Brecha vial", "confianza": 0.85, "justificacion": "Pista"},
        ]
    })
    supabase_service.update_query_status = MagicMock(return_value=True)
    supabase_service.save_classifications = MagicMock(return_value=True)

    await worker_service.process_message(sample_message)

    saved = supabase_service.save_classifications.call_args.args[0]
    assert len(saved) == 1
    assert saved[0]["gap_indicator_id"] == 3


# ── Invalid payload ────────────────────────────────────────────────────────

async def test_process_message_invalid_payload_updates_error(worker_service, supabase_service, gemini_service):
    supabase_service.update_query_status = MagicMock(return_value=True)

    bad_payload = {"query_id": "q-bad", "user_id": "u1"}  # missing required 'title'

    await worker_service.process_message(bad_payload)

    # Should not raise; status updated to error
    calls = supabase_service.update_query_status.call_args_list
    statuses = [c.args[1] for c in calls]
    assert "error" in statuses


async def test_process_message_empty_payload_does_not_raise(worker_service, supabase_service, gemini_service):
    supabase_service.update_query_status = MagicMock(return_value=True)

    # Empty payload — must not raise (caller always acks)
    await worker_service.process_message({})


# ── Multiple classifications ───────────────────────────────────────────────

async def test_process_message_multiple_labels_saved_in_order(worker_service, supabase_service, gemini_service, sample_message):
    gemini_service.classify = AsyncMock(return_value={
        "labels": [
            {"id": 1, "label": "Agua", "confianza": 0.95, "justificacion": "J1"},
            {"id": 2, "label": "Saneamiento", "confianza": 0.80, "justificacion": "J2"},
            {"id": 3, "label": "Energía", "confianza": 0.65, "justificacion": "J3"},
        ]
    })
    supabase_service.update_query_status = MagicMock(return_value=True)
    supabase_service.save_classifications = MagicMock(return_value=True)

    await worker_service.process_message(sample_message)

    saved = supabase_service.save_classifications.call_args.args[0]
    assert len(saved) == 3
    assert saved[0]["ranking_position"] == 1
    assert saved[1]["ranking_position"] == 2
    assert saved[2]["ranking_position"] == 3
    assert saved[0]["gap_indicator_id"] == 1
