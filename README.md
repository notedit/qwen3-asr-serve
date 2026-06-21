# qwen3-asr-serve

> Production HTTP serving for **Qwen3-ASR-1.7B** + **Qwen3-ForcedAligner-0.6B**
> on a single **NVIDIA L20** (46 GB) — extreme-performance focus.

OpenAI-compatible `/v1/audio/transcriptions` + a forced-alignment endpoint, with
batch routes, file-path input (skip multipart upload), and Prometheus metrics.
Three startup modes: **`asr`**, **`aligner`**, **`both`** (default).

## Highlights

- **vLLM 0.14** backed ASR. Up to **~2,800 audio-s/s** ASR-only / **~480
  audio-s/s** with word timestamps (single L20). Single-stream p50 ~648 ms.
- **One-shot Bash install** — no Docker. `./install.sh && ./run.sh`.
- **File-path input** is a first-class option (`file_path=…`): no multipart
  upload overhead for batch / large files.
- **Aligner batch-pin** baked in (sweet spot 16) — +6.4 % throughput,
  -77 % memory vs naïve sharing.
- **Modes are real**: in `MODE=asr`, the aligner is not loaded and the
  alignment routes return 404. Saves VRAM for higher `gpu_memory_utilization`.

---

## Quickstart

```bash
# 1. Install dependencies + download models (~6 GB)
./install.sh

# 2. Start the server (defaults to MODE=both)
./run.sh                      # foreground, MODE=both
./run.sh asr                  # foreground, MODE=asr
./run.sh aligner              # foreground, MODE=aligner

# Daemon (background) — survives SSH logout, logs to ./logs/server.log
./run.sh -d                   # background, MODE=both
./run.sh --daemon asr         # background, MODE=asr
./stop.sh                     # graceful stop (SIGTERM, falls back to KILL)
./status.sh                   # is it running? probe /health, tail log

# 3. Hit it
curl -F file=@some.wav http://localhost:8000/v1/audio/transcriptions
```

Server listens on `0.0.0.0:8000` by default. Configure via `.env` (copy
`.env.example`) or env vars.

### Daemon mode details

`./run.sh -d` does the right thing for a long-lived service:

- `setsid nohup` so the process survives shell exit / SSH disconnect
- Logs to `./logs/server.log` (stdout + stderr merged)
- PID stored in `./var/server.pid`, metadata (mode, port, start time) in `./var/server.meta`
- Port collision check before forking
- Refuses to start if a previous daemon is still alive — use `./stop.sh` first

`./stop.sh` sends SIGTERM, waits 30 s for graceful shutdown (vLLM needs ~5 s
to release GPU memory), then escalates to SIGKILL.  Use `./stop.sh --force`
to skip the wait.

For supervisord / systemd integration, just use foreground mode (`./run.sh`)
and let your supervisor handle restarts.

---

## API

All routes are OpenAPI-documented at `/docs`. Mounted routes depend on `MODE`:

| Route | Methods | `MODE=asr` | `MODE=aligner` | `MODE=both` |
| --- | --- | :---: | :---: | :---: |
| `/v1/audio/transcriptions` | POST | ✓ | — | ✓ |
| `/v1/audio/transcriptions/batch` | POST | ✓ | — | ✓ |
| `/v1/audio/forced_alignment` | POST | — | ✓ | ✓ |
| `/v1/audio/forced_alignment/batch` | POST | — | ✓ | ✓ |
| `/health`, `/metrics`, `/docs` | GET | ✓ | ✓ | ✓ |

### `POST /v1/audio/transcriptions` (OpenAI-compatible)

Multipart form fields:

| field | type | notes |
| --- | --- | --- |
| `file` | file | mutually exclusive with `file_path` |
| `file_path` | str | local path; requires `ALLOW_LOCAL_PATHS=true` + `ALLOWED_PATH_PREFIXES` whitelist |
| `model` | str | accepted but ignored (single-tenant) |
| `language` | str | ISO 639-1 (`zh`, `en`, `yue`, `ja`, …). Omit for auto-detect |
| `response_format` | str | `json` (default), `verbose_json`, `text` |
| `timestamp_granularities[]` | str | `word` or `segment` — repeat for both; **requires `MODE=both`** |

```bash
# minimal
curl -F file=@a.wav http://localhost:8000/v1/audio/transcriptions
# with word timestamps, OpenAI-style
curl -F file=@a.wav \
     -F response_format=verbose_json \
     -F 'timestamp_granularities[]=word' \
     http://localhost:8000/v1/audio/transcriptions
# fast-path: pass a local path instead of uploading
curl -F file_path=/data/audio/a.wav \
     -F language=zh \
     http://localhost:8000/v1/audio/transcriptions
```

Response shape (verbose_json):

```json
{
  "task": "transcribe",
  "language": "zh",
  "text": "刚收到快递，发现是男朋友…",
  "duration": 8.48,
  "words": [{"word":"刚","start":0.32,"end":0.48}, ...]
}
```

### `POST /v1/audio/transcriptions/batch` (extension)

Multi-file in one inference call. Pass either `audio_files[]` (multipart) **or**
`file_paths[]` (paths). Shares `language` / `response_format` /
`timestamp_granularities[]` across items.

```bash
curl -F 'audio_files=@a.wav' -F 'audio_files=@b.wav' \
     -F language=zh -F 'timestamp_granularities[]=word' \
     http://localhost:8000/v1/audio/transcriptions/batch

# best throughput: paths only
curl -F 'file_paths=/data/audio/a.wav' \
     -F 'file_paths=/data/audio/b.wav' \
     -F language=zh \
     http://localhost:8000/v1/audio/transcriptions/batch
```

Returns `{"results": [...]}` in upload order.

### `POST /v1/audio/forced_alignment`

Single audio + single transcript → per-word timestamps.

| field | required | notes |
| --- | --- | --- |
| `file` or `file_path` | yes (one) | |
| `text` | yes | exact transcript |
| `language` | yes | ISO code |
| `granularity` | no | `word` (default) / `segment` |

```bash
curl -F file=@a.wav -F text="刚收到快递" -F language=zh \
     http://localhost:8000/v1/audio/forced_alignment
```

### `POST /v1/audio/forced_alignment/batch`

`audio_files[]` or `file_paths[]` + `texts[]` (one per audio) + `language` + `granularity`.

---

## Modes — when to use which

| Mode | What's loaded | When to use |
| --- | --- | --- |
| `asr` | Qwen3-ASR-1.7B (vLLM) | Pure transcription. Frees VRAM → bump `GPU_MEM_UTIL=0.85` for max throughput. |
| `aligner` | Qwen3-ForcedAligner-0.6B (transformers) | You already have transcripts; just need timestamps. |
| `both` (default) | both, aligner shared | Need ASR + word timestamps. |

---

## Performance (NVIDIA L20, bf16)

From upstream `perf_bench/`. Workload: ~8 s Chinese wavs replicated, vLLM
backend, `gpu_memory_utilization=0.6`.

### Aggregate throughput

| Mode | Best batch | `audio-s/s` | RTF | Daily capacity (60 % util) |
| --- | ---: | ---: | ---: | ---: |
| ASR only | 512 | **~2,800** | 0.000 | **~40,000 h/day** |
| ASR + word timestamps | 256 (aligner pinned at 16) | **~480** | 0.002 | **~6,800 h/day** |
| Forced alignment only (single model) | 16 | ~660 | 0.001 | ~9,500 h/day |

### Single-stream latency (batch=1)

| Operation | p50 | p95 |
| --- | ---: | ---: |
| ASR | 648 ms | 688 ms |
| ASR + timestamps | 683 ms | 722 ms |

### Why the aligner-batch=16 pin matters

Sharing `ASR_BATCH=256` with the aligner naïvely → 486 audio-s/s, 9.8 GB peak.
Pinning aligner to internal batch=16 → 517 audio-s/s, 2.2 GB peak. Same code,
+6.4 % throughput, **-77 % memory** that vLLM can spend on a bigger KV cache.

---

## Configuration

All env vars (see `.env.example` for defaults):

| variable | default | notes |
| --- | --- | --- |
| `MODE` | `both` | `asr` / `aligner` / `both` |
| `MODEL_DIR` | `./models` | root directory containing both models |
| `ASR_MODEL_PATH` | `./models/Qwen3-ASR-1.7B` | |
| `ALIGNER_MODEL_PATH` | `./models/Qwen3-ForcedAligner-0.6B` | |
| `GPU_MEM_UTIL` | `0.6` | bump to `0.85` for `MODE=asr` |
| `ASR_BATCH` | `256` | vLLM Python-side chunk size |
| `ALIGNER_BATCH` | `16` | aligner internal batch (validated sweet spot) |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | |
| `ALLOW_LOCAL_PATHS` | `false` | must be `true` for `file_path` inputs |
| `ALLOWED_PATH_PREFIXES` | empty | comma-separated absolute prefixes; whitelist for path inputs |
| `MAX_FILE_BYTES` | `209715200` | 200 MiB cap on multipart uploads |
| `LOG_LEVEL` | `info` | |

---

## Metrics

`GET /metrics` returns Prometheus exposition. Key series:

| metric | labels | description |
| --- | --- | --- |
| `qwen_asr_requests_total` | `route`, `status`, `mode` | per-route counter |
| `qwen_asr_request_duration_seconds` | `route` | histogram, derive p50/p95/p99 with `histogram_quantile` |
| `qwen_asr_inflight` | `route` | gauge |
| `qwen_asr_audio_seconds_total` | `route` | cumulative audio seconds processed |
| `qwen_asr_batch_size` | `route` | observed batch size per request |
| `qwen_asr_gpu_memory_bytes` | — | sampled every 5 s via pynvml |
| `qwen_asr_gpu_utilization_ratio` | — | sampled every 5 s, in [0, 1] |
| `qwen_asr_model_ready` | `mode`, `component` | `1` once loaded |

PromQL example — p95 latency for transcriptions:

```
histogram_quantile(0.95,
  sum(rate(qwen_asr_request_duration_seconds_bucket{route="transcriptions"}[5m]))
    by (le))
```

---

## Troubleshooting

| symptom | fix |
| --- | --- |
| `torch.cuda.is_available() == False`, *Error 803* | The system driver and `*/cuda*/compat/*` lib in `LD_LIBRARY_PATH` disagree. `run.sh` already strips this — if you launch uvicorn manually, do the same: drop any entry containing `compat`. |
| `pip install vllm==0.14.0` "platform not supported" | Use the official PyPI (`--index-url https://pypi.org/simple/`) AND the `--platform manylinux_2_31_x86_64 --only-binary=:all:` trick. `install.sh` does this for you. |
| `ImportError: cannot import name 'infer_schema' from 'torch.library'` | torch is < 2.9; upgrade to 2.9.1 (see install.sh). |
| flash-attn `undefined symbol: at::_ops::zeros4call` | The bundled flash-attn 2.6 is ABI-incompatible with torch 2.9. `install.sh` uninstalls it; vLLM falls back to its built-in attention. |
| Model download timed out | `HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --mode both` |
| `Error retrieving safetensors … Network is unreachable` at startup | vLLM HEAD-probes huggingface.co even with a local model. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` before `run.sh` to skip the probe. |
| `ValueError: ... KV cache memory ... larger than the available KV cache memory` | The default `max_model_len=8192` (lifespan.py) should cover ASR; if you raised it, lower it back or bump `GPU_MEM_UTIL`. |
| `403 path is outside ALLOWED_PATH_PREFIXES` | You enabled `ALLOW_LOCAL_PATHS` but didn't whitelist the directory the file lives in. Add it to `ALLOWED_PATH_PREFIXES` (comma-separated absolute paths). |

---

## Project layout

```
.
├── install.sh                 vLLM dance + model download
├── run.sh                     launch uvicorn (LD strip baked in)
├── pyproject.toml
├── .env.example
├── scripts/
│   ├── download_models.py     HF + hf-mirror fallback
│   └── verify.sh              curl smoke test
└── app/
    ├── main.py                FastAPI factory + per-mode router mounting
    ├── lifespan.py            load Qwen3ASRModel.LLM and/or Qwen3ForcedAligner
    ├── config.py              pydantic Settings (env-driven)
    ├── aligner_patch.py       the one validated optimisation
    ├── audio_io.py            multipart + path resolver
    ├── mapping.py             ASRTranscription ↔ OpenAI shape, language map
    ├── metrics.py             Prometheus exposition + GPU sampler
    ├── schemas.py             pydantic response models
    └── routes/
        ├── transcriptions.py  /v1/audio/transcriptions [+ /batch]
        ├── alignments.py      /v1/audio/forced_alignment [+ /batch]
        └── ops.py             /health, /metrics
```

---

## License

Apache-2.0.  Wraps the upstream `qwen_asr` package (also Apache-2.0).
