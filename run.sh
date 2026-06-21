#!/usr/bin/env bash
# Start the Qwen3-ASR-Serve HTTP server.
#
#   ./run.sh                # foreground, MODE from .env / env, default both
#   ./run.sh asr            # foreground, MODE=asr
#   ./run.sh aligner        # foreground, MODE=aligner
#   ./run.sh -d             # background (daemonised, logs to ./logs/server.log)
#   ./run.sh --daemon both  # background, MODE=both
#
# Companion commands:
#   ./stop.sh               # graceful stop (SIGTERM, falls back to SIGKILL)
#   ./status.sh             # is it running? what mode? recent log tail?
#
# Critical sandbox fix:
# The /usr/local/cuda-*/compat/ paths inside LD_LIBRARY_PATH bind libcuda.so to
# a compat lib whose version sometimes disagrees with the host driver. That
# combo trips CUDA Error 803 on first cudaGetDeviceCount(). We strip the
# `*compat*` entries before launch so libcuda resolves to the system driver.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Argument parsing — accept positional MODE and/or -d/--daemon flag in any order.
DAEMON=0
MODE_ARG=""
for arg in "$@"; do
    case "$arg" in
        -d|--daemon)
            DAEMON=1
            ;;
        asr|aligner|both)
            MODE_ARG="$arg"
            ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "[run] unknown argument: $arg" >&2
            echo "[run] usage: $0 [asr|aligner|both] [-d|--daemon]" >&2
            exit 2
            ;;
    esac
done

# Load .env if present.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# MODE precedence: positional arg > env > default both
if [[ -n "$MODE_ARG" ]]; then
    MODE="$MODE_ARG"
fi
export MODE="${MODE:-both}"

# Strip cuda-compat dirs from LD_LIBRARY_PATH
NEW_LD=$(python3 - <<'PY'
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
export LD_LIBRARY_PATH="$NEW_LD"

# Skip the HEAD-to-huggingface probe at startup — models are local already.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# venv
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Refuse to start if the port is already taken — avoids two engines fighting
# for the same GPU and leaves a clear error message.
if command -v ss >/dev/null 2>&1; then
    if ss -tlnH "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
        echo "[run] port $PORT is already in use; refusing to start" >&2
        echo "[run] check with: ./status.sh   then ./stop.sh if needed" >&2
        exit 1
    fi
fi

VAR_DIR="$HERE/var"
LOG_DIR="$HERE/logs"
PID_FILE="$VAR_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"
META_FILE="$VAR_DIR/server.meta"

mkdir -p "$VAR_DIR" "$LOG_DIR"

# If a pid file exists and the process is alive, refuse.
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[run] server already running (pid=$OLD_PID).  Use ./stop.sh first." >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

UVICORN_CMD=(uvicorn app.main:app
    --host "$HOST" --port "$PORT"
    --workers 1
    --loop uvloop --http httptools
    --log-level "$LOG_LEVEL")

if [[ "$DAEMON" == "1" ]]; then
    echo "[run] starting (daemon)  MODE=$MODE  HOST=$HOST  PORT=$PORT  log=$LOG_FILE"
    # Detach with setsid so the child survives this shell, including SSH logout.
    # `nohup` alone isn't enough on some shells when the parent is a CI runner.
    setsid nohup "${UVICORN_CMD[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
    CHILD_PID=$!
    # Briefly confirm it's actually up (didn't die on import).
    for _ in 1 2 3 4 5; do
        if ! kill -0 "$CHILD_PID" 2>/dev/null; then
            echo "[run] server died during startup; tail of $LOG_FILE:" >&2
            tail -n 40 "$LOG_FILE" >&2 || true
            exit 1
        fi
        sleep 1
    done
    echo "$CHILD_PID" > "$PID_FILE"
    cat > "$META_FILE" <<META
PID=$CHILD_PID
MODE=$MODE
HOST=$HOST
PORT=$PORT
STARTED=$(date -Iseconds)
LOG=$LOG_FILE
META
    echo "[run] running (pid=$CHILD_PID).  Tail logs: tail -f $LOG_FILE"
    echo "[run] health (may take ~30-90s for vLLM to load):  curl -s http://${HOST}:${PORT}/health"
    exit 0
fi

# Foreground path — replace shell, log to stdout/stderr.
echo "[run] starting (foreground)  MODE=$MODE  HOST=$HOST  PORT=$PORT"
# Best-effort pid file even in foreground so ./status.sh sees it.
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE" "$META_FILE"' EXIT
cat > "$META_FILE" <<META
PID=$$
MODE=$MODE
HOST=$HOST
PORT=$PORT
STARTED=$(date -Iseconds)
LOG=stdout
META
# single worker — vLLM engine + aligner own the GPU; forking would corrupt CUDA ctx.
exec "${UVICORN_CMD[@]}"
