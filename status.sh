#!/usr/bin/env bash
# Show whether qwen3-asr-serve is running, and a quick view of recent activity.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PID_FILE="$HERE/var/server.pid"
META_FILE="$HERE/var/server.meta"
LOG_FILE="$HERE/logs/server.log"

if [[ ! -f "$PID_FILE" ]]; then
    echo "status: NOT RUNNING (no pid file)"
    exit 1
fi

PID=$(cat "$PID_FILE" 2>/dev/null || true)
if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
    echo "status: NOT RUNNING (pid=$PID is gone; stale pid file)"
    exit 1
fi

echo "status: RUNNING  pid=$PID"
if [[ -f "$META_FILE" ]]; then
    sed 's/^/  /' "$META_FILE"
fi

# Probe /health if we know the port
PORT=$(grep -E '^PORT=' "$META_FILE" 2>/dev/null | head -1 | cut -d= -f2 || true)
HOST=$(grep -E '^HOST=' "$META_FILE" 2>/dev/null | head -1 | cut -d= -f2 || true)
if [[ -n "${PORT:-}" ]]; then
    HOST_FOR_CURL="${HOST:-127.0.0.1}"
    [[ "$HOST_FOR_CURL" == "0.0.0.0" ]] && HOST_FOR_CURL="127.0.0.1"
    echo ""
    echo "health (http://${HOST_FOR_CURL}:${PORT}/health):"
    if curl -sf --max-time 2 "http://${HOST_FOR_CURL}:${PORT}/health"; then
        echo ""
    else
        echo "  (not responding yet — model still loading?)"
    fi
fi

# Tail of log if available
if [[ -f "$LOG_FILE" ]]; then
    echo ""
    echo "last 10 log lines ($LOG_FILE):"
    tail -n 10 "$LOG_FILE" | sed 's/^/  /'
fi
