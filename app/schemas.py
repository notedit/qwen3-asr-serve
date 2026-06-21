"""Pydantic response schemas (drive OpenAPI docs)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Word(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    id: int
    seek: int = 0
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible /v1/audio/transcriptions response (JSON / verbose_json)."""

    text: str
    language: str = ""
    task: Optional[str] = None  # 'transcribe' in verbose_json
    duration: Optional[float] = None
    words: Optional[List[Word]] = None
    segments: Optional[List[Segment]] = None


class BatchTranscriptionResponse(BaseModel):
    results: List[TranscriptionResponse]


class AlignmentResponse(BaseModel):
    """Forced-alignment endpoint response."""

    language: str
    duration: float
    words: Optional[List[Word]] = None
    segments: Optional[List[Segment]] = None


class BatchAlignmentResponse(BaseModel):
    results: List[AlignmentResponse]


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' once models are loaded; 'loading' during startup")
    mode: str
    asr_ready: bool = False
    aligner_ready: bool = False
    version: str
