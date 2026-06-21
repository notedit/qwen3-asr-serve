#!/usr/bin/env bash
# scripts/verify.sh — curl-based smoke test for a running server.
#
# Usage:
#   ./scripts/verify.sh                       # MODE=both (default)
#   MODE=asr      ./scripts/verify.sh
#   MODE=aligner  ./scripts/verify.sh
#   HOST=... PORT=... ./scripts/verify.sh
#
# Expects a working audio file at $AUDIO (default ../Qwen3-ASR/audios/audio_v-female-T3P8sZ0Q_0006(6).wav).
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODE="${MODE:-both}"
BASE="http://${HOST}:${PORT}"
AUDIO="${AUDIO:-../Qwen3-ASR/audios/audio_v-female-T3P8sZ0Q_0006(6).wav}"
TEXT="${TEXT:-刚收到快递，发现是男朋友偷偷寄来的生日蛋糕。}"

red()   { printf "\033[0;31m%s\033[0m\n" "$*"; }
green() { printf "\033[0;32m%s\033[0m\n" "$*"; }
log()   { printf "\033[1;34m[verify]\033[0m %s\n" "$*"; }

[[ -f "$AUDIO" ]] || { red "audio file not found: $AUDIO"; exit 1; }

# Health
log "GET ${BASE}/health"
curl -sf "${BASE}/health" | head -c 400; echo

if [[ "$MODE" == "asr" || "$MODE" == "both" ]]; then
    log "POST ${BASE}/v1/audio/transcriptions  (multipart)"
    curl -sf -X POST "${BASE}/v1/audio/transcriptions" \
        -F "file=@${AUDIO}" -F "response_format=json" | head -c 600
    echo
    green "  transcriptions OK"

    log "POST ${BASE}/v1/audio/transcriptions/batch  (2 files)"
    curl -sf -X POST "${BASE}/v1/audio/transcriptions/batch" \
        -F "audio_files=@${AUDIO}" -F "audio_files=@${AUDIO}" \
        -F "language=zh" | head -c 800
    echo
    green "  transcriptions/batch OK"
fi

if [[ "$MODE" == "aligner" || "$MODE" == "both" ]]; then
    log "POST ${BASE}/v1/audio/forced_alignment  (multipart)"
    curl -sf -X POST "${BASE}/v1/audio/forced_alignment" \
        -F "file=@${AUDIO}" -F "text=${TEXT}" -F "language=zh" \
        -F "granularity=word" | head -c 800
    echo
    green "  forced_alignment OK"

    log "POST ${BASE}/v1/audio/forced_alignment/batch  (2 items)"
    curl -sf -X POST "${BASE}/v1/audio/forced_alignment/batch" \
        -F "audio_files=@${AUDIO}" -F "audio_files=@${AUDIO}" \
        -F "texts=${TEXT}" -F "texts=${TEXT}" \
        -F "language=zh" -F "granularity=word" | head -c 800
    echo
    green "  forced_alignment/batch OK"
fi

log "GET ${BASE}/metrics  (first 20 lines)"
curl -sf "${BASE}/metrics" | grep -E "^qwen_asr_" | head -n 20

green "all checks passed"
