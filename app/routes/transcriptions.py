"""ASR routes (mounted when MODE in {asr, both}).

- POST /v1/audio/transcriptions       OpenAI-compatible single file
- POST /v1/audio/transcriptions/batch Multi-file extension
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.audio_io import audio_duration_seconds, resolve_audio, resolve_audio_batch
from app.mapping import asr_to_openai, to_qwen_lang
from app.metrics import AUDIO_S_TOTAL, BATCH_SIZE_HIST, track
from app.schemas import BatchTranscriptionResponse, TranscriptionResponse


router = APIRouter(prefix="/v1/audio", tags=["asr"])
logger = logging.getLogger("qwen3-asr-serve.transcriptions")


_VALID_GRANULARITIES = {"word", "segment"}


def _normalize_granularities(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for v in values:
        v = (v or "").strip().lower()
        if not v:
            continue
        if v not in _VALID_GRANULARITIES:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported timestamp_granularity: {v!r}; "
                       f"valid: {sorted(_VALID_GRANULARITIES)}",
            )
        out.append(v)
    return out


def _check_ts_supported(request: Request, granularities: List[str]) -> None:
    if granularities and request.app.state.mode == "asr":
        raise HTTPException(
            status_code=400,
            detail=(
                "timestamp_granularities[] requires the forced aligner; this "
                "server was started with MODE=asr. Restart with MODE=both."
            ),
        )


@router.post(
    "/transcriptions",
    response_model=TranscriptionResponse,
    responses={200: {"content": {"text/plain": {}}}},
)
async def transcribe(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Audio file (multipart)."),
    file_path: Optional[str] = Form(None, description="Local file path (when ALLOW_LOCAL_PATHS=true)."),
    model: str = Form("qwen3-asr-1.7b"),
    language: Optional[str] = Form(None, description="ISO 639-1 code, e.g. 'zh','en'. Omit for auto-detect."),
    response_format: str = Form("json", description="json | verbose_json | text"),
    timestamp_granularities: List[str] = Form(default_factory=list, alias="timestamp_granularities[]"),
):
    """OpenAI-compatible /v1/audio/transcriptions.

    File-path input is the fast path — set ALLOW_LOCAL_PATHS=true and pass `file_path`
    instead of uploading via multipart.
    """
    if response_format not in {"json", "verbose_json", "text"}:
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")

    granularities = _normalize_granularities(timestamp_granularities)
    _check_ts_supported(request, granularities)

    qwen_lang = to_qwen_lang(language)
    audio = await resolve_audio(file, file_path)

    async with track("transcriptions", request.app.state.mode):
        AUDIO_S_TOTAL.labels(route="transcriptions").inc(audio_duration_seconds(audio))
        BATCH_SIZE_HIST.labels(route="transcriptions").observe(1)
        results = request.app.state.asr.transcribe(
            audio=[audio],
            language=qwen_lang,
            return_time_stamps=bool(granularities),
        )
    payload = asr_to_openai(results[0], granularities, response_format)

    if isinstance(payload, PlainTextResponse):
        return payload
    return JSONResponse(payload)


@router.post("/transcriptions/batch", response_model=BatchTranscriptionResponse)
async def transcribe_batch(
    request: Request,
    audio_files: List[UploadFile] = File(default_factory=list),
    file_paths: List[str] = Form(default_factory=list),
    model: str = Form("qwen3-asr-1.7b"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: List[str] = Form(default_factory=list, alias="timestamp_granularities[]"),
) -> BatchTranscriptionResponse:
    """Batch transcription: one inference call across many audios.

    Pass `audio_files[]` (multipart) OR `file_paths[]` (paths). Results
    preserve input order.
    """
    if response_format not in {"json", "verbose_json", "text"}:
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")
    if response_format == "text":
        # batch + plain text would require concatenation semantics we don't define
        raise HTTPException(status_code=400, detail="response_format=text not supported on /batch")

    granularities = _normalize_granularities(timestamp_granularities)
    _check_ts_supported(request, granularities)

    qwen_lang = to_qwen_lang(language)
    audios = await resolve_audio_batch(audio_files, file_paths)

    async with track("transcriptions_batch", request.app.state.mode):
        total_dur = sum(audio_duration_seconds(a) for a in audios)
        AUDIO_S_TOTAL.labels(route="transcriptions_batch").inc(total_dur)
        BATCH_SIZE_HIST.labels(route="transcriptions_batch").observe(len(audios))
        results = request.app.state.asr.transcribe(
            audio=audios,
            language=qwen_lang,
            return_time_stamps=bool(granularities),
        )

    return BatchTranscriptionResponse(
        results=[asr_to_openai(r, granularities, response_format) for r in results]
    )
