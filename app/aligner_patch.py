"""Aligner batch-pin patch (the single validated performance optimisation).

Wraps `Qwen3ForcedAligner.align` to internally chunk inputs to a fixed batch
size, regardless of how large a batch the caller hands in. Aligner's sweet
spot on L20 is batch=16 (perf_bench/test_split_aligner_batch.py & ALIGNER_BOTTLENECK.md):

  shared ASR=256 + Aligner=256  → 486.5 audio-s/s, 9.83 GB peak
  ASR=256 + Aligner=16 (pinned) → 517.3 audio-s/s, 2.24 GB peak  (+6.4%, −77% mem)

The freed memory lets vLLM grow its KV cache, which can compound the gain.
"""

from __future__ import annotations

from typing import Any


def patch_aligner(aligner: Any, internal_batch: int = 16) -> Any:
    """Monkey-patch `aligner.align` to slice into chunks of `internal_batch`.

    Idempotent: detects prior patch via the `_qwen3_serve_pinned_batch` marker.
    """
    if getattr(aligner, "_qwen3_serve_pinned_batch", None):
        return aligner

    original_align = aligner.align

    def chunked_align(audio, text, language):  # signature matches Qwen3ForcedAligner.align
        if isinstance(audio, list) and len(audio) > internal_batch:
            results = []
            n = len(audio)
            for i in range(0, n, internal_batch):
                a = audio[i : i + internal_batch]
                t = text[i : i + internal_batch] if isinstance(text, list) else text
                lang = language[i : i + internal_batch] if isinstance(language, list) else language
                results.extend(original_align(audio=a, text=t, language=lang))
            return results
        return original_align(audio=audio, text=text, language=language)

    aligner.align = chunked_align
    aligner._qwen3_serve_pinned_batch = internal_batch
    return aligner
