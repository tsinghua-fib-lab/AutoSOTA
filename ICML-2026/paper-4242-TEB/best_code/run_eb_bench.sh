#!/usr/bin/env bash
set -euo pipefail
# Usage: run_eb_bench.sh
# Reads environment variables for configuration.

PORT="${EB_PORT:-11015}"
LOG_FILE="/tmp/eb_server_$(date +%s).log"
OUTPUT_DIR="${OUTPUT_DIR:-/repo/reproduce/outputs/eb}"
OUTPUT_FILE="${OUTPUT_FILE:-eb_result.json}"

cd /repo
mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Starting EB server on port $PORT..."
python3 start_eb_flex.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
READY=0
for i in $(seq 1 60); do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost', $PORT)); s.close(); print('OK')" 2>/dev/null; then
        echo "[$(date)] Server ready after ${i}s"
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date)] Server died! Last 20 lines of log:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    sleep 1
done

if [[ "$READY" -eq 0 ]]; then
    echo "[$(date)] Server failed to start within 60s"
    tail -20 "$LOG_FILE"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# Run benchmark
echo "[$(date)] Running benchmark..."
BASE_URL="http://localhost:$PORT" \
    NUM_PROMPTS="${NUM_PROMPTS:-4000}" \
    CONCURRENCY="${CONCURRENCY:-2048}" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    OUTPUT_FILE="$OUTPUT_FILE" \
    python3 simple_bench.py

# Kill server
echo "[$(date)] Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
sleep 2
kill -9 "$SERVER_PID" 2>/dev/null || true

# Parse results
RESULT_FILE="$OUTPUT_DIR/$OUTPUT_FILE"
if [[ -f "$RESULT_FILE" ]]; then
    echo "[$(date)] Results:"
    python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
print(f'RPS={d[\"rps\"]:.2f}, TPOT_P50={d[\"tpot_ms_p50\"]:.2f}ms, Time={d[\"total_time_s\"]:.1f}s')
"
else
    echo "[$(date)] ERROR: No result file at $RESULT_FILE"
    exit 1
fi
