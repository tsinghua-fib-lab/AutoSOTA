#!/bin/bash
# Common helper library — sourced by experiment scripts.

GPU_MEM_THRESHOLD=${GPU_MEM_THRESHOLD:-10000}

# Select which GPUs to use.
# Args:    $1 = MAX_GPUS
# Env:     GPUS = explicit GPU list (e.g. "0,1,2")
# Output:  sets the GPUS_TO_USE array
select_gpus() {
    local max_gpus=${1:-4}

    if [ -n "${GPUS:-}" ]; then
        IFS=',' read -ra GPUS_TO_USE <<< "$GPUS"
        echo "Using user-specified GPUs: ${GPUS_TO_USE[*]}"
        return
    fi

    # Auto-detect available GPUs (memory usage below threshold)
    local available=()
    while IFS=, read -r gpu_id _ _ mem_used; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' MiB')
        [ "$mem_used" -lt "$GPU_MEM_THRESHOLD" ] && available+=("$gpu_id")
    done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null)

    [ ${#available[@]} -eq 0 ] && { echo "Error: no GPU available"; exit 1; }

    GPUS_TO_USE=("${available[@]:0:$max_gpus}")
    echo "Using ${#GPUS_TO_USE[@]} GPU(s): ${GPUS_TO_USE[*]}"
}

# Check whether a port is free (only LISTEN state; avoid TIME_WAIT / ESTABLISHED false positives)
check_port_available() {
    local port=$1 gpu_id=${2:-""}
    if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
        echo "[GPU $gpu_id] port $port is in use"
        return 1
    fi
}

# Probe port availability by actually attempting bind() in Python (no lsof permissions needed)
port_is_free() {
    local port=$1
    python - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

# Walk from start_port to find the first free port
find_free_port() {
    local start_port=$1
    local max_tries=${2:-50}
    local port=$start_port
    local i=0
    while [ $i -lt $max_tries ]; do
        if port_is_free "$port"; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
        i=$((i + 1))
    done
    return 1
}

# Wait for GPU memory to be released
wait_for_gpu_memory() {
    local gpu_id=$1 max_wait=${2:-30} i=0
    while [ $i -lt $max_wait ]; do
        local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d ' ')
        [ -n "$mem" ] && [ "$mem" -lt "$GPU_MEM_THRESHOLD" ] && return 0
        sleep 1; ((i++))
    done
    return 1
}

# Wait for the server to come up
wait_for_server() {
    local port=$1 server_pid=$2 max_wait=${3:-180} log_file=${4:-""} i=0
    while [ $i -lt $max_wait ]; do
        curl -s "http://localhost:${port}/health" >/dev/null 2>&1 && return 0
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "Server process exited unexpectedly"
            [ -n "$log_file" ] && [ -f "$log_file" ] && sed 's/^/  /' "$log_file"
            return 1
        fi
        sleep 1; ((i++))
    done
    echo "Server startup timed out (${max_wait}s)"
    [ -n "$log_file" ] && [ -f "$log_file" ] && sed 's/^/  /' "$log_file"
    return 1
}

# Terminate a server process
kill_server() {
    local server_pid=$1 gpu_id=${2:-""}
    local max_wait=${3:-15}  # default 15 s for graceful shutdown

    # Graceful TERM — give the server time to flush stats etc.
    kill -TERM $server_pid 2>/dev/null || true

    # Wait for natural exit
    local i=0
    while [ $i -lt $max_wait ]; do
        if ! kill -0 $server_pid 2>/dev/null; then
            # Process exited
            break
        fi
        sleep 1
        ((i++))
    done

    # If still alive, force kill
    if kill -0 $server_pid 2>/dev/null; then
        echo "Process $server_pid did not respond to SIGTERM, forcing..."
        kill -9 $server_pid 2>/dev/null || true
        # Also kill child processes
        pkill -9 -P $server_pid 2>/dev/null || true
    fi

    # Clean up any GPU-resident processes left behind
    if [ -n "$gpu_id" ]; then
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $gpu_id 2>/dev/null | tr -d ' '); do
            [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
        done
        wait_for_gpu_memory "$gpu_id" 30 || true
    else
        sleep 2
    fi
}

# Pop the next experiment from the queue (lock-protected)
get_next_experiment() {
    local queue_file=$1 lock_file=$2
    (
        flock -x 200
        local exp=$(head -n 1 "$queue_file" 2>/dev/null)
        if [ -n "$exp" ]; then
            tail -n +2 "$queue_file" > "${queue_file}.tmp"
            mv "${queue_file}.tmp" "$queue_file"
            echo "$exp"
        fi
    ) 200>"$lock_file"
}

# Append a status line and print progress
update_progress() {
    local status=$1 progress_file=$2 lock_file=$3 total=$4
    (
        flock -x 200
        echo "$status" >> "$progress_file"
        local done=$(wc -l < "$progress_file")
        echo "Progress: $done / $total"
    ) 200>"$lock_file"
}

# Ensure a hardware calibration file exists; run calibration if not.
# Usage:  ensure_calibration <model> <model_short>
# Effect: exports VLLM_PD_CALIBRATION_FILE
ensure_calibration() {
    local model=$1 model_short=$2
    local _common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local calibration_dir="${_common_dir}/outputs"

    if [ -n "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
        if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
            return 0
        else
            echo "Error: specified calibration file does not exist: $VLLM_PD_CALIBRATION_FILE"
            return 1
        fi
    fi

    local calibration_file="${calibration_dir}/pd_calibration_${model_short}.json"
    if [ -f "$calibration_file" ]; then
        export VLLM_PD_CALIBRATION_FILE="$calibration_file"
        return 0
    fi

    echo "Calibration file not found; running hardware calibration..."
    mkdir -p "$calibration_dir"
    python3 -m vllm.v1.core.sched.calibration \
        --model "$model" \
        --output "$calibration_file" || return 1
    export VLLM_PD_CALIBRATION_FILE="$calibration_file"
    echo "Calibration done: $calibration_file"
}

# Initialize the experiment environment (venv, ulimit, etc.)
init_experiment_env() {
    # common.sh is at reproduce/common/common.sh; repo root is ../.. (eb-vllm/).
    local _common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local venv_path=${1:-"${_common_dir}/../../.venv"}
    ulimit -n 65535 2>/dev/null || true
    # Skip .venv activation if a conda env is already active and has vllm —
    # otherwise sourcing a stale .venv shadows conda's Python and causes
    # "ModuleNotFoundError: No module named 'vllm'" mid-run.
    if [ -n "${CONDA_PREFIX:-}" ] && command -v vllm >/dev/null 2>&1; then
        : # use the active conda env, skip .venv
    elif [ -f "${venv_path}/bin/activate" ]; then
        source "${venv_path}/bin/activate"
    elif ! command -v vllm >/dev/null 2>&1; then
        echo "[WARN] No venv found at ${venv_path}/bin/activate and 'vllm' not in PATH." >&2
        echo "       Either activate a venv with vllm installed, or symlink it to:" >&2
        echo "         ln -s /path/to/your/.venv ${_common_dir}/../../.venv" >&2
    fi
}

# Print final summary stats
print_summary() {
    local progress_file=$1 total=$2 output_dir=$3
    local done=$(wc -l < "$progress_file" 2>/dev/null || echo 0)
    local ok=$(grep -c "^OK|" "$progress_file" 2>/dev/null || echo 0)
    local fail=$(grep -c "^FAIL|" "$progress_file" 2>/dev/null || echo 0)

    echo ""
    echo "========================================"
    echo "Experiment done: $done / $total (ok: $ok, fail: $fail)"
    echo "Output directory: $output_dir"
    echo "========================================"
}
