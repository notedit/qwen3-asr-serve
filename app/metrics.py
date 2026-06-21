"""Prometheus instrumentation."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "qwen_asr_requests_total",
    "Total HTTP requests",
    ["route", "status", "mode"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "qwen_asr_request_duration_seconds",
    "End-to-end request latency, seconds",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
    registry=REGISTRY,
)
INFLIGHT = Gauge(
    "qwen_asr_inflight",
    "In-flight requests per route",
    ["route"],
    registry=REGISTRY,
)
AUDIO_S_TOTAL = Counter(
    "qwen_asr_audio_seconds_total",
    "Cumulative audio seconds processed",
    ["route"],
    registry=REGISTRY,
)
BATCH_SIZE_HIST = Histogram(
    "qwen_asr_batch_size",
    "Number of audio items per request",
    ["route"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    registry=REGISTRY,
)
GPU_MEM_BYTES = Gauge(
    "qwen_asr_gpu_memory_bytes",
    "GPU memory used (bytes), sampled",
    registry=REGISTRY,
)
GPU_UTIL = Gauge(
    "qwen_asr_gpu_utilization_ratio",
    "GPU compute utilisation in [0,1], sampled",
    registry=REGISTRY,
)
MODEL_READY = Gauge(
    "qwen_asr_model_ready",
    "1 if the component has been loaded",
    ["mode", "component"],
    registry=REGISTRY,
)


@contextlib.asynccontextmanager
async def track(route: str, mode: str):
    """Context manager: increment inflight, time, record status."""
    INFLIGHT.labels(route=route).inc()
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        LATENCY.labels(route=route).observe(time.perf_counter() - start)
        INFLIGHT.labels(route=route).dec()
        REQUESTS.labels(route=route, status=status, mode=mode).inc()


def latest_metrics() -> bytes:
    return generate_latest(REGISTRY)


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST


# ─── GPU sampler ─────────────────────────────────────────────────────────

_sampler_task: Any = None


async def _gpu_sampler_loop(interval: float = 5.0) -> None:
    """Best-effort GPU stats every `interval` seconds.

    pynvml may not be available (e.g. CPU-only sandbox); silently noop.
    """
    try:
        import pynvml  # type: ignore
    except Exception:  # noqa: BLE001
        return

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:  # noqa: BLE001
        return

    try:
        while True:
            try:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                GPU_MEM_BYTES.set(meminfo.used)
                GPU_UTIL.set(util.gpu / 100.0)
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(interval)
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def start_gpu_sampler() -> None:
    global _sampler_task
    if _sampler_task is None:
        _sampler_task = asyncio.create_task(_gpu_sampler_loop())


def stop_gpu_sampler() -> None:
    global _sampler_task
    if _sampler_task is not None:
        _sampler_task.cancel()
        _sampler_task = None
