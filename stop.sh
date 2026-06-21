#!/usr/bin/env bash
# Stop a daemonised qwen3-asr-serve.
#
#   ./stop.sh           # graceful SIGTERM, wait up to 30s, then SIGKILL
#   ./stop.sh --force   # SIGKILL immediately
#
# Reads var/server.pid written by `./run.sh -d`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PID_FILE="$HERE/var/server.pid"
META_FILE="$HERE/var/server.meta"
FORCE=0
if [[ "${1:-}" == "--force" || "${1:-}" == "-9" ]]; then
    FORCE=1
fi

if [[ ! -f "$PID_FILE" ]]; then
    echo "[stop] no pid file at $PID_FILE — nothing to stop" >&2
    exit 0
fi

PID=$(cat "$PID_FILE" 2>/dev/null || true)
if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
    echo "[stop] pid $PID is not running; cleaning up stale pid file"
    rm -f "$PID_FILE" "$META_FILE"
    exit 0
fi

if [[ "$FORCE" == "1" ]]; then
    echo "[stop] SIGKILL pid=$PID"
    kill -9 "$PID"
else
    echo "[stop] SIGTERM pid=$PID  (waiting up to 30s)"
    kill -TERM "$PID" 2>/dev/null || true
    for i in $(seq 1 30); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "[stop] stopped cleanly after ${i}s"
            rm -f "$PID_FILE" "$META_FILE"
            exit 0
        fi
        sleep 1
    done
    echo "[stop] still alive after 30s — escalating to SIGKILL"
    kill -9 "$PID" 2>/dev/null || true
fi

# Give the kernel a moment to reap & release VRAM
sleep 1
if kill -0 "$PID" 2>/dev/null; then
    echo "[stop] WARNING: pid $PID still alive after SIGKILL?" >&2
    exit 1
fi
rm -f "$PID_FILE" "$META_FILE"
echo "[stop] done"
