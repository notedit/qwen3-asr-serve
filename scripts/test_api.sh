#!/usr/bin/env bash
# Full API test for qwen3-asr-serve.
#
# Runs each route through normal paths, error paths, and edge cases.
# Prints PASS/FAIL per test, exits non-zero if any test fails.
#
# Usage:
#   BASE=http://127.0.0.1:18765 AUDIO_DIR=/path bash scripts/test_api.sh
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:18765}"
AUDIO_DIR="${AUDIO_DIR:-/workspace/user_code/Qwen3-ASR/audios}"

A1="${AUDIO_DIR}/audio_v-female-T3P8sZ0Q_0006(6).wav"
A2="${AUDIO_DIR}/audio_v-female-T3P8sZ0Q_0021(5).wav"
A3="${AUDIO_DIR}/audio_v-female-T3P8sZ0Q_0022(5).wav"
A4="${AUDIO_DIR}/audio_v-female-T3P8sZ0Q_0025(5).wav"

TEXT1="刚收到快递，发现是男朋友偷偷寄来的生日蛋糕。"
TEXT2="明明是他先违反规则，在先还倒打一耙。"

# Test outputs
PASS=0
FAIL=0
LOG=/tmp/api_test_results.log
: > "$LOG"

cyan()  { printf "\033[1;36m%s\033[0m\n" "$*"; }
green() { printf "\033[0;32m%s\033[0m" "$*"; }
red()   { printf "\033[0;31m%s\033[0m" "$*"; }

# Run a test: NAME, EXPECTED_STATUS, EXPECTED_BODY_REGEX, curl args...
run_test() {
    local name="$1"; shift
    local expect_status="$1"; shift
    local body_regex="$1"; shift
    # Capture body + status separately
    local tmp; tmp=$(mktemp)
    local status
    status=$(curl -s -o "$tmp" -w "%{http_code}" "$@" 2>/dev/null || echo "000")
    local body; body=$(cat "$tmp"); rm -f "$tmp"

    local ok=1
    if [[ "$status" != "$expect_status" ]]; then ok=0; fi
    if [[ -n "$body_regex" && ! "$body" =~ $body_regex ]]; then ok=0; fi

    if [[ "$ok" == "1" ]]; then
        printf "  [%s] %s  (HTTP %s)\n" "$(green PASS)" "$name" "$status"
        PASS=$((PASS+1))
    else
        printf "  [%s] %s  (HTTP %s, expected %s)\n" "$(red FAIL)" "$name" "$status" "$expect_status"
        printf "    body: %.200s\n" "$body"
        FAIL=$((FAIL+1))
    fi
    {
        echo "==== $name ===="
        echo "STATUS=$status (expected $expect_status)"
        echo "BODY: $body"
        echo
    } >> "$LOG"
}

# ──────────────────────────────────────────────────────────────────────
cyan "▶ Ops"
# ──────────────────────────────────────────────────────────────────────

run_test "GET /health 200 + mode=both" \
    200 '"mode":"both".*"asr_ready":true.*"aligner_ready":true' \
    "${BASE}/health"

run_test "GET /metrics 200 + qwen_asr_ prefix" \
    200 'qwen_asr_requests_total' \
    "${BASE}/metrics"

run_test "GET /docs Swagger 200" \
    200 '' \
    "${BASE}/docs"

run_test "GET /openapi.json 200 + paths" \
    200 '"paths":' \
    "${BASE}/openapi.json"

# ──────────────────────────────────────────────────────────────────────
cyan "▶ POST /v1/audio/transcriptions — happy paths"
# ──────────────────────────────────────────────────────────────────────

run_test "multipart, default json" \
    200 '"text":".+","language":"zh"' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}"

run_test "multipart, response_format=text → plain text body" \
    200 '收到快递' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}" -F "response_format=text"

run_test "multipart, verbose_json + word timestamps" \
    200 '"words":\[.*"word":' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}" \
    -F "response_format=verbose_json" \
    -F 'timestamp_granularities[]=word'

run_test "multipart, segment timestamps" \
    200 '"segments":\[' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}" \
    -F "response_format=verbose_json" \
    -F 'timestamp_granularities[]=segment'

run_test "multipart, both word+segment timestamps" \
    200 '"words":.*"segments":' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}" \
    -F "response_format=verbose_json" \
    -F 'timestamp_granularities[]=word' \
    -F 'timestamp_granularities[]=segment'

run_test "file_path fast path" \
    200 '"text":' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file_path=${A1}"

run_test "file_path + language=zh override" \
    200 '"language":"zh"' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file_path=${A1}" -F "language=zh"

run_test "file_path + language=en (forced; output may differ)" \
    200 '"language":"en"' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file_path=${A1}" -F "language=en"

# ──────────────────────────────────────────────────────────────────────
cyan "▶ POST /v1/audio/transcriptions — error paths"
# ──────────────────────────────────────────────────────────────────────

run_test "missing both file and file_path → 400" \
    400 'exactly one of' \
    -X POST "${BASE}/v1/audio/transcriptions"

run_test "both file and file_path → 400" \
    400 'only one of' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}" -F "file_path=${A2}"

run_test "invalid response_format → 400" \
    400 'unsupported response_format' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}" -F "response_format=xml"

run_test "invalid timestamp_granularities[] → 400" \
    400 'unsupported timestamp_granularity' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}" \
    -F 'timestamp_granularities[]=phoneme'

run_test "invalid language code → 400" \
    400 'unsupported language' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file=@${A1}" -F "language=xyz"

run_test "file_path outside whitelist → 403" \
    403 'outside ALLOWED_PATH_PREFIXES' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file_path=/etc/passwd"

run_test "file_path that does not exist → 404" \
    404 'not a file' \
    -X POST "${BASE}/v1/audio/transcriptions" \
    -F "file_path=${AUDIO_DIR}/nonexistent.wav"

# ──────────────────────────────────────────────────────────────────────
cyan "▶ POST /v1/audio/transcriptions/batch"
# ──────────────────────────────────────────────────────────────────────

run_test "batch 2 files via paths" \
    200 '"results":\[\{"text".+,\{"text"' \
    -X POST "${BASE}/v1/audio/transcriptions/batch" \
    -F "file_paths=${A1}" -F "file_paths=${A2}" \
    -F "language=zh"

run_test "batch 4 files via paths, with word timestamps" \
    200 '"results":\[.+"words":\[' \
    -X POST "${BASE}/v1/audio/transcriptions/batch" \
    -F "file_paths=${A1}" -F "file_paths=${A2}" \
    -F "file_paths=${A3}" -F "file_paths=${A4}" \
    -F "language=zh" \
    -F "response_format=verbose_json" \
    -F 'timestamp_granularities[]=word'

run_test "batch 2 files via multipart upload" \
    200 '"results":\[' \
    -X POST "${BASE}/v1/audio/transcriptions/batch" \
    -F "audio_files=@${A1}" -F "audio_files=@${A2}" \
    -F "language=zh"

run_test "batch with both audio_files and file_paths → 400" \
    400 'exactly one of' \
    -X POST "${BASE}/v1/audio/transcriptions/batch" \
    -F "audio_files=@${A1}" -F "file_paths=${A2}" \
    -F "language=zh"

run_test "batch with neither audio_files nor file_paths → 400" \
    400 'exactly one of' \
    -X POST "${BASE}/v1/audio/transcriptions/batch" \
    -F "language=zh"

run_test "batch with response_format=text → 400" \
    400 'response_format=text not supported on /batch' \
    -X POST "${BASE}/v1/audio/transcriptions/batch" \
    -F "file_paths=${A1}" -F "response_format=text"

# ──────────────────────────────────────────────────────────────────────
cyan "▶ POST /v1/audio/forced_alignment — happy paths"
# ──────────────────────────────────────────────────────────────────────

run_test "alignment, word granularity (default)" \
    200 '"words":\[.*"word":"刚"' \
    -X POST "${BASE}/v1/audio/forced_alignment" \
    -F "file_path=${A1}" -F "text=${TEXT1}" -F "language=zh"

run_test "alignment, segment granularity" \
    200 '"segments":\[' \
    -X POST "${BASE}/v1/audio/forced_alignment" \
    -F "file_path=${A1}" -F "text=${TEXT1}" -F "language=zh" \
    -F "granularity=segment"

run_test "alignment, multipart upload" \
    200 '"words":' \
    -X POST "${BASE}/v1/audio/forced_alignment" \
    -F "file=@${A1}" -F "text=${TEXT1}" -F "language=zh"

# ──────────────────────────────────────────────────────────────────────
cyan "▶ POST /v1/audio/forced_alignment — error paths"
# ──────────────────────────────────────────────────────────────────────

run_test "alignment missing text → 400" \
    400 'text must not be empty' \
    -X POST "${BASE}/v1/audio/forced_alignment" \
    -F "file_path=${A1}" -F "language=zh"

run_test "alignment missing language → 400" \
    400 'language must not be empty' \
    -X POST "${BASE}/v1/audio/forced_alignment" \
    -F "file_path=${A1}" -F "text=${TEXT1}"

run_test "alignment empty text → 400" \
    400 'text must not be empty' \
    -X POST "${BASE}/v1/audio/forced_alignment" \
    -F "file_path=${A1}" -F "text=" -F "language=zh"

run_test "alignment unsupported granularity → 400" \
    400 'unsupported granularity' \
    -X POST "${BASE}/v1/audio/forced_alignment" \
    -F "file_path=${A1}" -F "text=${TEXT1}" -F "language=zh" \
    -F "granularity=phoneme"

# ──────────────────────────────────────────────────────────────────────
cyan "▶ POST /v1/audio/forced_alignment/batch"
# ──────────────────────────────────────────────────────────────────────

run_test "alignment batch 2 items via paths" \
    200 '"results":\[.+"words":\[' \
    -X POST "${BASE}/v1/audio/forced_alignment/batch" \
    -F "file_paths=${A1}" -F "file_paths=${A2}" \
    -F "texts=${TEXT1}" -F "texts=${TEXT2}" \
    -F "language=zh"

run_test "alignment batch length mismatch → 400" \
    400 'length mismatch' \
    -X POST "${BASE}/v1/audio/forced_alignment/batch" \
    -F "file_paths=${A1}" -F "file_paths=${A2}" \
    -F "texts=${TEXT1}" \
    -F "language=zh"

run_test "alignment batch empty text in array → 400" \
    400 'every text entry must be non-empty' \
    -X POST "${BASE}/v1/audio/forced_alignment/batch" \
    -F "file_paths=${A1}" -F "file_paths=${A2}" \
    -F "texts=${TEXT1}" -F "texts=" \
    -F "language=zh"

# ──────────────────────────────────────────────────────────────────────
cyan "▶ Performance smoke test"
# ──────────────────────────────────────────────────────────────────────

# Single-stream latency (3 calls, median)
declare -a lat_asr lat_ts lat_align
for i in 1 2 3; do
    t=$( { time -p curl -sf -o /dev/null -X POST "${BASE}/v1/audio/transcriptions" \
        -F "file_path=${A1}" -F "language=zh" ; } 2>&1 | awk '/real/ {print $2*1000}')
    lat_asr+=("$t")
done

for i in 1 2 3; do
    t=$( { time -p curl -sf -o /dev/null -X POST "${BASE}/v1/audio/transcriptions" \
        -F "file_path=${A1}" -F "language=zh" \
        -F "response_format=verbose_json" \
        -F 'timestamp_granularities[]=word' ; } 2>&1 | awk '/real/ {print $2*1000}')
    lat_ts+=("$t")
done

for i in 1 2 3; do
    t=$( { time -p curl -sf -o /dev/null -X POST "${BASE}/v1/audio/forced_alignment" \
        -F "file_path=${A1}" -F "text=${TEXT1}" -F "language=zh" ; } 2>&1 | awk '/real/ {print $2*1000}')
    lat_align+=("$t")
done

echo "  ASR-only batch=1 latency (ms): ${lat_asr[*]}"
echo "  ASR+word-ts batch=1 latency (ms): ${lat_ts[*]}"
echo "  Forced-alignment batch=1 latency (ms): ${lat_align[*]}"

# Batch throughput: 4 audios, with timestamps
echo
echo "  Batch=4 timed (with word timestamps):"
t_batch=$( { time -p curl -sf -o /dev/null -X POST "${BASE}/v1/audio/transcriptions/batch" \
    -F "file_paths=${A1}" -F "file_paths=${A2}" \
    -F "file_paths=${A3}" -F "file_paths=${A4}" \
    -F "language=zh" \
    -F "response_format=verbose_json" \
    -F 'timestamp_granularities[]=word' ; } 2>&1 | awk '/real/ {print $2}')
echo "  Wall time: ${t_batch}s  (4 audios × ~8.5s each ≈ 34s total)"

# ──────────────────────────────────────────────────────────────────────
echo
cyan "▶ Metrics post-test snapshot"
curl -s "${BASE}/metrics" | grep -E "^qwen_asr_(requests_total|audio_seconds_total|inflight) " | head -20

# ──────────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════"
echo "  PASS: $PASS    FAIL: $FAIL"
echo "  Full log: $LOG"
echo "════════════════════════════════════════════"
[[ "$FAIL" == "0" ]] && exit 0 || exit 1
