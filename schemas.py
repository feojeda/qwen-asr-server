"""Pydantic request / response schemas for qwen-asr-server.

All API endpoints return typed Pydantic models so consumers
know exactly what fields to expect.
"""

from typing import Optional, List

from pydantic import BaseModel, Field


# ── Health ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status: 'healthy' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether the ASR model is loaded in memory")
    diarization_loaded: bool = Field(..., description="Whether the diarization pipeline is loaded")
    device: str = Field(..., description="Inference device (cpu, cuda, mps)")
    model_name: str = Field(..., description="HuggingFace model ID in use")


# ── Models (OpenAI-compatible) ─────────────────────────────────────


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "Qwen"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ── Transcription ──────────────────────────────────────────────────


class TranscriptionResponse(BaseModel):
    """Response for /v1/audio/transcriptions (both simple and diarized)."""
    text: str = Field(..., description="Full transcription text")
    language: str = Field(..., description="Detected or requested language")
    duration: float = Field(..., description="Audio duration in seconds")
    processing_time: float = Field(..., description="Total processing time in seconds")
    segments: Optional[List["TranscriptionSegment"]] = Field(
        default=None,
        description="Per-speaker segments. Only present when diarize=true and diarization succeeded.",
    )
    diarization_failed: Optional[bool] = Field(
        default=None,
        description="True when diarize=true but the diarization pipeline failed "
                    "(graceful degradation: transcription without speaker labels)",
    )


class TranscriptionSegment(BaseModel):
    speaker: str = Field(..., description="Speaker label (e.g. SPEAKER_00)")
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    language: Optional[str] = Field(
        default=None,
        description="Language detected for this segment (e.g. Spanish, English)",
    )


# ── Diarization ────────────────────────────────────────────────────


class DiarizationSegment(BaseModel):
    speaker: str = Field(..., description="Speaker label (e.g. SPEAKER_00)")
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")


class DiarizationResponse(BaseModel):
    segments: List[DiarizationSegment]
    num_speakers: int = Field(..., description="Number of unique speakers detected")
