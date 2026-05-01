"""Unit tests for Pydantic response schemas — no server, no models, no GPU needed."""

import pytest
from pydantic import ValidationError

from schemas import (
    DiarizationResponse,
    DiarizationSegment,
    HealthResponse,
    ModelInfo,
    ModelList,
    TranscriptionResponse,
    TranscriptionSegment,
)


class TestHealthResponse:
    def test_valid(self):
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            diarization_loaded=False,
            device="cpu",
            model_name="Qwen/Qwen3-ASR-0.6B",
        )
        assert resp.status == "healthy"
        assert resp.model_loaded is True
        assert resp.diarization_loaded is False
        assert resp.device == "cpu"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy")


class TestModelList:
    def test_single_model(self):
        resp = ModelList(
            object="list",
            data=[
                ModelInfo(id="Qwen3-ASR-0.6B", object="model", owned_by="Qwen"),
            ],
        )
        assert resp.object == "list"
        assert len(resp.data) == 1
        assert resp.data[0].id == "Qwen3-ASR-0.6B"

    def test_multiple_models(self):
        resp = ModelList(
            object="list",
            data=[
                ModelInfo(id="model-a", object="model", owned_by="Qwen"),
                ModelInfo(id="model-b", object="model", owned_by="Qwen"),
            ],
        )
        assert len(resp.data) == 2


class TestTranscriptionResponse:
    def test_simple_transcription(self):
        resp = TranscriptionResponse(
            text="Hola mundo",
            language="Spanish",
            duration=5.2,
            processing_time=2.1,
        )
        assert resp.text == "Hola mundo"
        assert resp.language == "Spanish"
        assert resp.duration == 5.2
        assert resp.segments is None
        assert resp.diarization_failed is None

    def test_with_segments(self):
        resp = TranscriptionResponse(
            text="Hola mundo Adiós",
            language="Spanish",
            duration=10.0,
            processing_time=4.5,
            segments=[
                TranscriptionSegment(
                    speaker="SPEAKER_00", start=0.0, end=2.5, text="Hola mundo"
                ),
                TranscriptionSegment(
                    speaker="SPEAKER_01", start=3.0, end=5.0, text="Adiós"
                ),
            ],
        )
        assert len(resp.segments) == 2
        assert resp.segments[0].speaker == "SPEAKER_00"
        assert resp.segments[1].text == "Adiós"

    def test_diarization_failed_flag(self):
        resp = TranscriptionResponse(
            text="Hola",
            language="Spanish",
            duration=3.0,
            processing_time=1.0,
            segments=[],
            diarization_failed=True,
        )
        assert resp.diarization_failed is True
        assert resp.segments == []

    def test_serialize_to_dict(self):
        """Pydantic models should serialize cleanly to JSON-compatible dicts."""
        resp = TranscriptionResponse(
            text="test",
            language="English",
            duration=1.0,
            processing_time=0.5,
            segments=[
                TranscriptionSegment(
                    speaker="S1", start=0.0, end=1.0, text="test"
                ),
            ],
            diarization_failed=False,
        )
        d = resp.model_dump()
        assert d["text"] == "test"
        assert d["segments"][0]["speaker"] == "S1"
        assert d["diarization_failed"] is False


class TestDiarizationResponse:
    def test_valid(self):
        resp = DiarizationResponse(
            segments=[
                DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
                DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
            ],
            num_speakers=2,
        )
        assert resp.num_speakers == 2
        assert len(resp.segments) == 2

    def test_empty_segments(self):
        resp = DiarizationResponse(
            segments=[],
            num_speakers=0,
        )
        assert resp.segments == []
        assert resp.num_speakers == 0
