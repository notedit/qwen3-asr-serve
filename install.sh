#!/usr/bin/env bash
# Qwen3-ASR-Serve installer.
#
# Idempotent: each step skips if already done. Replicates the validated
# install dance from upstream perf_bench/INSTALL_VLLM.md (glibc 2.28 + L20
# + vLLM 0.14 + torch 2.9.1 wheel that won't resolve via normal pip).
#
# Usage:
#   ./install.sh            # MODE=both, downloads both models
#   MODE=asr ./install.sh   # only downloads Qwen3-ASR-1.7B
#   MODE=aligner ./install.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODE="${MODE:-both}"
VENV="${VENV:-$HERE/.venv}"
PY="${PYTHON:-python3.11}"

log() { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

# ───── 1. Python version check ────────────────────────────────────────────
if ! command -v "$PY" >/dev/null; then
    PY="python3"
fi
PYVER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') || err "no python"
[[ "$PYVER" == "3.11" ]] || err "Python 3.11.x required, got $PYVER"
log "python: $($PY --version)"

# ───── 2. venv ────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    log "creating venv at $VENV"
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip setuptools wheel

# Always use official PyPI for torch + vllm — tencent mirror has the
# 0.14.0+cu122 metadata bug noted in upstream INSTALL_VLLM.md.
PYPI="https://pypi.org/simple/"

have() { python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(\"$1\") else 1)" 2>/dev/null; }

# ───── 3. torch 2.9.1 (manylinux_2_28 wheel — glibc 2.28 compatible) ─────
TORCH_VER_NEEDED="2.9.1"
if have torch; then
    CUR=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    if [[ "$CUR" != "$TORCH_VER_NEEDED" ]]; then
        log "torch is $CUR, upgrading to $TORCH_VER_NEEDED"
        pip install --index-url "$PYPI" "torch==$TORCH_VER_NEEDED" \
            "torchaudio==2.9.1" "torchvision==0.24.1"
    else
        log "torch $CUR OK"
    fi
else
    log "installing torch $TORCH_VER_NEEDED"
    pip install --index-url "$PYPI" "torch==$TORCH_VER_NEEDED" \
        "torchaudio==2.9.1" "torchvision==0.24.1"
fi

# ───── 4. vLLM 0.14.0 (force-install wheel past platform-tag check) ──────
VLLM_VER_NEEDED="0.14.0"
if have vllm; then
    CUR=$(python -c "import vllm; print(vllm.__version__.split('+')[0])")
    if [[ "$CUR" != "$VLLM_VER_NEEDED" ]]; then
        FORCE_VLLM=1
    else
        log "vllm $CUR OK"
        FORCE_VLLM=0
    fi
else
    FORCE_VLLM=1
fi
if [[ "$FORCE_VLLM" == "1" ]]; then
    log "force-installing vllm==$VLLM_VER_NEEDED (manylinux_2_31 wheel)"
    SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
    pip install --no-deps --force-reinstall \
        --platform manylinux_2_31_x86_64 --python-version 3.11 \
        --target="$SITE" --only-binary=:all: \
        --index-url "$PYPI" "vllm==$VLLM_VER_NEEDED"
fi

# ───── 5. vLLM runtime deps (matches INSTALL_VLLM.md §3 verbatim) ────────
log "installing vllm runtime dependencies"
pip install --quiet \
    regex cachetools psutil sentencepiece blake3 py-cpuinfo \
    "transformers>=4.56.0,<5" "tokenizers>=0.21.1" "protobuf>=6.30.0" \
    "fastapi[standard]>=0.115.0" aiohttp "openai>=1.99.1" \
    "pydantic>=2.12.0" "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.11.3" "llguidance>=1.3.0,<1.4.0" \
    "outlines_core==0.2.11" "diskcache==5.6.3" "lark==1.2.2" \
    "xgrammar==0.1.29" "typing_extensions>=4.10" "filelock>=3.16.1" \
    partial-json-parser "pyzmq>=25.0.0" msgspec "gguf>=0.17.0" \
    "mistral_common[image]>=1.8.8" "opencv-python-headless>=4.11.0" \
    pyyaml einops "compressed-tensors==0.13.0" "depyf==0.20.0" \
    cloudpickle watchfiles python-json-logger ninja pybase64 cbor2 \
    ijson setproctitle "openai-harmony>=0.0.3" "anthropic==0.71.0"

pip install --quiet \
    "model-hosting-container-standards>=0.1.10,<1.0.0" \
    "ray[cgraph]>=2.48.0" "numba==0.61.2"

pip install --quiet "grpcio>=1.76.0" "grpcio-reflection>=1.76.0" mcp

pip install --quiet "flashinfer-python==0.5.3"

# ───── 6. Uninstall flash-attn (ABI break with torch 2.9) ────────────────
if have flash_attn; then
    log "removing old flash-attn (incompatible with torch 2.9)"
    pip uninstall -y flash-attn || true
fi

# ───── 7. scipy upgrade (numpy 2.x compat) ───────────────────────────────
log "ensuring scipy is compatible with numpy 2.x"
pip install --quiet --upgrade scipy

# ───── 8. qwen_asr package (transcribe / ForcedAligner) ──────────────────
# QWEN_ASR_SOURCE can be a git URL with optional @ref, OR a local path that
# will be pip-installed editable. Default to the public upstream repo.
QWEN_ASR_SOURCE="${QWEN_ASR_SOURCE:-git+https://github.com/QwenLM/Qwen3-ASR.git}"
if have qwen_asr; then
    log "qwen_asr already importable"
else
    log "installing qwen_asr from $QWEN_ASR_SOURCE"
    pip install --quiet --no-deps "$QWEN_ASR_SOURCE"
    pip install --quiet "nagisa==0.2.11" "soynlp==0.0.493" \
        "accelerate==1.12.0" "qwen-omni-utils" "librosa" "soundfile" "av"
fi

# ───── 9. Server deps (FastAPI, prometheus, etc.) ────────────────────────
log "installing server dependencies"
pip install --quiet -e .

# ───── 10. Model download ────────────────────────────────────────────────
log "downloading models for MODE=$MODE"
python scripts/download_models.py --mode "$MODE"

# ───── 11. Verification ──────────────────────────────────────────────────
log "verifying install"
# Strip cuda compat from LD_LIBRARY_PATH for this verification step (the
# same trick that run.sh applies at startup).
VERIFY_LD=$(python - <<'PY'
import os
parts = os.environ.get("LD_LIBRARY_PATH","").split(":")
seen, out = set(), []
for p in parts:
    if not p or "compat" in p: continue
    if p in seen: continue
    seen.add(p); out.append(p)
print(":".join(out))
PY
)
LD_LIBRARY_PATH="$VERIFY_LD" python - <<'PY'
import vllm, torch
print(f"vllm: {vllm.__version__}")
print(f"torch: {torch.__version__}  cuda: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not available; double-check LD_LIBRARY_PATH (should not include cuda compat)"
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner  # noqa
print("qwen_asr OK")
PY

log "DONE.  Start the server with:  ./run.sh        (default MODE=both)"
log "                                ./run.sh asr   (ASR only)"
log "                                ./run.sh aligner"
