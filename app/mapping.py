"""Map between qwen_asr's native shapes and OpenAI-compatible JSON."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from fastapi import HTTPException


# qwen_asr SUPPORTED_LANGUAGES (utils.py:37-68) → ISO 639-1 / 639-3 best-effort.
QWEN_TO_OPENAI_LANG: Dict[str, str] = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Russian": "ru",
    "Italian": "it",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Malay": "ms",
    "Hindi": "hi",
    "Bengali": "bn",
    "Urdu": "ur",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Filipino": "fil",
    "Hebrew": "he",
    "Persian": "fa",
    "Greek": "el",
    "Czech": "cs",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Bulgarian": "bg",
    "Ukrainian": "uk",
    "Serbian": "sr",
    "Croatian": "hr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Estonian": "et",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Swedish": "sv",
    "Danish": "da",
    "Norwegian": "no",
    "Finnish": "fi",
}
OPENAI_TO_QWEN_LANG: Dict[str, str] = {v: k for k, v in QWEN_TO_OPENAI_LANG.items()}


def to_openai_lang(qwen_lang: Optional[str]) -> str:
    """'Chinese' → 'zh'.  Comma-joined ('Chinese,English') → 'zh,en'."""
    if not qwen_lang:
        return ""
    parts = [p.strip() for p in qwen_lang.split(",") if p.strip()]
    return ",".join(QWEN_TO_OPENAI_LANG.get(p, p.lower()) for p in parts)


def to_qwen_lang(openai_lang: Optional[str]) -> Optional[str]:
    """'zh' → 'Chinese'.  None or '' → None (auto-detect).  Comma-joined supported."""
    if not openai_lang or not openai_lang.strip():
        return None
    parts = [p.strip().lower() for p in openai_lang.split(",") if p.strip()]
    mapped = []
    for p in parts:
        if p in OPENAI_TO_QWEN_LANG:
            mapped.append(OPENAI_TO_QWEN_LANG[p])
        else:
            # accept full names too: "chinese" → "Chinese"
            cap = p.capitalize()
            if cap in QWEN_TO_OPENAI_LANG:
                mapped.append(cap)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"unsupported language: {p!r}",
                )
    return ",".join(mapped) if len(mapped) > 1 else mapped[0]


# Punctuation chars used for segment synthesis.
_SEG_PUNCT = "。！？.!?"
_SEG_RE = re.compile(rf"([^{_SEG_PUNCT}]+[{_SEG_PUNCT}]*)")


def _items_for_chars(text: str, items: Sequence[Any]):
    """Yield (char_index, item) for char-aligned items.

    Qwen3 emits one item per CJK char or per space-delimited word.  For
    segment synthesis we walk the text in lockstep with items, skipping
    whitespace in the text.
    """
    j = 0
    n = len(items)
    for i, ch in enumerate(text):
        if ch.isspace():
            yield i, None
            continue
        if j < n:
            yield i, items[j]
            j += 1
        else:
            yield i, None


def synthesize_segments(text: str, items: Sequence[Any]) -> List[Dict[str, Any]]:
    """Split `text` on punctuation; assign start/end from the item span it covers."""
    if not text or not items:
        return []
    # Map char index → item start_time / end_time (or None).
    starts: List[Optional[float]] = [None] * len(text)
    ends: List[Optional[float]] = [None] * len(text)
    for i, it in _items_for_chars(text, items):
        if it is None:
            continue
        starts[i] = float(it.start_time)
        ends[i] = float(it.end_time)

    segments: List[Dict[str, Any]] = []
    cursor = 0
    seg_id = 0
    for match in _SEG_RE.finditer(text):
        lo, hi = match.start(), match.end()
        seg_text = match.group(0).strip()
        if not seg_text:
            continue
        # Find first non-None start in [lo, hi)
        s = next((starts[k] for k in range(lo, hi) if starts[k] is not None), None)
        e = next((ends[k] for k in range(hi - 1, lo - 1, -1) if ends[k] is not None), None)
        if s is None and e is None:
            continue
        segments.append({
            "id": seg_id,
            "seek": 0,
            "start": s if s is not None else (e or 0.0),
            "end": e if e is not None else (s or 0.0),
            "text": seg_text,
        })
        seg_id += 1
        cursor = hi
    return segments


def asr_to_openai(
    asr: Any,
    granularities: Sequence[str],
    response_format: str,
) -> Any:
    """Map ASRTranscription → OpenAI shape (or PlainTextResponse for 'text')."""
    if response_format == "text":
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(asr.text)

    obj: Dict[str, Any] = {
        "text": asr.text,
        "language": to_openai_lang(asr.language),
    }

    # verbose_json upgrades default JSON with duration/task fields
    if response_format == "verbose_json":
        obj["task"] = "transcribe"

    if "word" in granularities and asr.time_stamps:
        obj["words"] = [
            {
                "word": it.text,
                "start": float(it.start_time),
                "end": float(it.end_time),
            }
            for it in asr.time_stamps.items
        ]
        if response_format == "verbose_json" and obj["words"]:
            obj["duration"] = obj["words"][-1]["end"]

    if "segment" in granularities and asr.time_stamps:
        obj["segments"] = synthesize_segments(asr.text, asr.time_stamps.items)

    return obj


def alignment_to_response(
    result: Any,
    granularity: str,
    text: str,
    language: str,
) -> Dict[str, Any]:
    """Map ForcedAlignResult → minimal JSON for the alignment endpoint."""
    items = list(result.items) if result is not None else []
    duration = float(items[-1].end_time) if items else 0.0
    out: Dict[str, Any] = {
        "language": to_openai_lang(language),
        "duration": duration,
    }
    if granularity == "word":
        out["words"] = [
            {"word": it.text, "start": float(it.start_time), "end": float(it.end_time)}
            for it in items
        ]
    elif granularity == "segment":
        out["segments"] = synthesize_segments(text, items)
    else:
        raise HTTPException(status_code=400, detail=f"unsupported granularity: {granularity}")
    return out
