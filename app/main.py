"""FastAPI app factory + router mounting per MODE."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app import __version__
from app.config import settings
from app.lifespan import lifespan
from app.routes import alignments, ops, transcriptions


logging.basicConfig(
    level=settings.log_level.upper() if settings.log_level.upper() in
        {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"} else "INFO",
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Qwen3-ASR-Serve",
        version=__version__,
        description=(
            f"Production HTTP serving for Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B "
            f"on a single NVIDIA L20.  Running mode: **{settings.mode}**."
        ),
        lifespan=lifespan,
    )

    # ops routes are always mounted
    app.include_router(ops.router)

    # ASR routes only when ASR is loaded
    if settings.mode in ("asr", "both"):
        app.include_router(transcriptions.router)

    # Aligner routes only when aligner is loaded
    if settings.mode in ("aligner", "both"):
        app.include_router(alignments.router)

    return app


app = create_app()
