"""FastAPI lifespan: load models once per process, patch aligner, expose state."""

from __future__ import annotations

import gc
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.aligner_patch import patch_aligner
from app.config import settings
from app.metrics import MODEL_READY, start_gpu_sampler, stop_gpu_sampler

logger = logging.getLogger("qwen3-asr-serve.lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load `qwen_asr` engines according to settings.mode.

    `asr` and `aligner` are stashed on app.state for routes to consume.
    """
    import torch
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

    s = settings
    app.state.mode = s.mode
    app.state.asr = None
    app.state.aligner = None
    app.state.model_ready = False
    MODEL_READY.labels(mode=s.mode, component="asr").set(0)
    MODEL_READY.labels(mode=s.mode, component="aligner").set(0)

    logger.info("loading mode=%s asr=%s aligner=%s",
                s.mode, s.asr_model_path, s.aligner_model_path)

    try:
        # vLLM kwargs shared between MODE=asr and MODE=both.
        # max_model_len caps KV-cache sizing — 8192 is plenty for ASR
        # (~1200s of audio context).  Without it vLLM tries to allocate KV
        # for the full 65536 model context, which is wasteful here.
        vllm_kwargs = dict(
            max_inference_batch_size=s.asr_batch,
            max_new_tokens=256,
            gpu_memory_utilization=s.gpu_mem_util,
            max_model_len=8192,
        )

        if s.mode == "asr":
            app.state.asr = Qwen3ASRModel.LLM(
                model=str(s.asr_model_path),
                **vllm_kwargs,
            )
            MODEL_READY.labels(mode=s.mode, component="asr").set(1)

        elif s.mode == "aligner":
            aligner = Qwen3ForcedAligner.from_pretrained(
                str(s.aligner_model_path),
                dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            patch_aligner(aligner, internal_batch=s.aligner_batch)
            app.state.aligner = aligner
            MODEL_READY.labels(mode=s.mode, component="aligner").set(1)

        else:  # both
            app.state.asr = Qwen3ASRModel.LLM(
                model=str(s.asr_model_path),
                forced_aligner=str(s.aligner_model_path),
                forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
                **vllm_kwargs,
            )
            aligner = app.state.asr.forced_aligner
            patch_aligner(aligner, internal_batch=s.aligner_batch)
            app.state.aligner = aligner
            MODEL_READY.labels(mode=s.mode, component="asr").set(1)
            MODEL_READY.labels(mode=s.mode, component="aligner").set(1)

        app.state.model_ready = True
        start_gpu_sampler()
        logger.info("ready (mode=%s)", s.mode)
        yield

    finally:
        logger.info("shutting down")
        stop_gpu_sampler()
        app.state.asr = None
        app.state.aligner = None
        app.state.model_ready = False
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
