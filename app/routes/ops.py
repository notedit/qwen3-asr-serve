"""GET /health and /metrics — always mounted regardless of MODE."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app import __version__
from app.metrics import latest_metrics, metrics_content_type
from app.schemas import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    return HealthResponse(
        status="ok" if getattr(state, "model_ready", False) else "loading",
        mode=getattr(state, "mode", "unknown"),
        asr_ready=getattr(state, "asr", None) is not None,
        aligner_ready=getattr(state, "aligner", None) is not None,
        version=__version__,
    )


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=latest_metrics(), media_type=metrics_content_type())
