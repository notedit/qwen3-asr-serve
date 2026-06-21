"""Forced alignment routes (mounted when MODE in {aligner, both}).

- POST /v1/audio/forced_alignment        single audio + text
- POST /v1/audio/forced_alignment/batch  multi audio + multi text
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.audio_io import audio_duration_seconds, resolve_audio, resolve_audio_batch
from app.mapping import alignment_to_response, to_qwen_lang
from app.metrics import AUDIO_S_TOTAL, BATCH_SIZE_HIST, track
from app.schemas import AlignmentResponse, BatchAlignmentResponse


router = APIRouter(prefix="/v1/audio", tags=["aligner"])
logger = logging.getLogger("qwen3-asr-serve.alignments")


_VALID_GRANULARITY = {"word", "segment"}


def _check_granularity(g: str) -> str:
    g = (g or "").strip().lower() or "word"
    if g not in _VALID_GRANULARITY:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported granularity: {g!r}; valid: {sorted(_VALID_GRANULARITY)}",
        )
    return g


@router.post("/forced_alignment", response_model=AlignmentResponse)
async def forced_alignment(
    request: Request,
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None),
    text: str = Form(..., description="Transcript to align (required)."),
    language: str = Form(..., description="ISO 639-1 code, e.g. 'zh','en' (required)."),
    granularity: str = Form("word", description="word | segment"),
) -> AlignmentResponse:
    """Run forced alignment on a single (audio, text) pair."""
    g = _check_granularity(granularity)
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    qwen_lang = to_qwen_lang(language)
    if qwen_lang is None:
        raise HTTPException(status_code=400, detail="language is required")

    audio = await resolve_audio(file, file_path)

    async with track("forced_alignment", request.app.state.mode):
        AUDIO_S_TOTAL.labels(route="forced_alignment").inc(audio_duration_seconds(audio))
        BATCH_SIZE_HIST.labels(route="forced_alignment").observe(1)
        results = request.app.state.aligner.align(
            audio=[audio], text=[text], language=[qwen_lang]
        )
    return AlignmentResponse(**alignment_to_response(results[0], g, text, qwen_lang))


@router.post("/forced_alignment/batch", response_model=BatchAlignmentResponse)
async def forced_alignment_batch(
    request: Request,
    audio_files: List[UploadFile] = File(default_factory=list),
    file_paths: List[str] = Form(default_factory=list),
    texts: List[str] = Form(..., description="One transcript per audio item, in order."),
    language: str = Form(..., description="ISO 639-1 code shared across items."),
    granularity: str = Form("word"),
) -> BatchAlignmentResponse:
    """Run forced alignment on N (audio, text) pairs in one inference call."""
    g = _check_granularity(granularity)
    qwen_lang = to_qwen_lang(language)
    if qwen_lang is None:
        raise HTTPException(status_code=400, detail="language is required")

    audios = await resolve_audio_batch(audio_files, file_paths)
    if len(audios) != len(texts):
        raise HTTPException(
            status_code=400,
            detail=f"audios ({len(audios)}) and texts ({len(texts)}) length mismatch",
        )
    if any(not (t or "").strip() for t in texts):
        raise HTTPException(status_code=400, detail="every text entry must be non-empty")

    async with track("forced_alignment_batch", request.app.state.mode):
        total_dur = sum(audio_duration_seconds(a) for a in audios)
        AUDIO_S_TOTAL.labels(route="forced_alignment_batch").inc(total_dur)
        BATCH_SIZE_HIST.labels(route="forced_alignment_batch").observe(len(audios))
        results = request.app.state.aligner.align(
            audio=audios, text=list(texts), language=[qwen_lang] * len(audios)
        )

    return BatchAlignmentResponse(
        results=[
            AlignmentResponse(**alignment_to_response(r, g, t, qwen_lang))
            for r, t in zip(results, texts)
        ]
    )
