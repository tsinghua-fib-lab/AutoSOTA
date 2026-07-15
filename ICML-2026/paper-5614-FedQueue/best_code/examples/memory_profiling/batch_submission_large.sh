#!/bin/bash
#PBS -A PPFL_FM
#PBS -q preemptable
#PBS -l walltime=06:00:00
#PBS -l nodes=3:ppn=64
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

set -euo pipefail

APPFL_DIR="/eagle/tpc/zilinghan/appfl/APPFL/examples"
MODEL_NAME="meta-llama/Llama-3.1-70B"
SERVER_CONFIG="./memory_profiling/configs/server_llm_dummy.yaml"
CLIENT_CONFIG="./memory_profiling/configs/client_1_llm_dummy.yaml"

ENV_SETUP='
  cd "'"$APPFL_DIR"'"
  [[ -f ~/.bashrc ]] && source ~/.bashrc
  module use /soft/modulefiles
  module load conda
  conda activate /eagle/tpc/zilinghan/conda_envs/appfl
  export CUDA_VISIBLE_DEVICES=0,1,2,3
'

cd "$APPFL_DIR"
mkdir -p logs memory_profiles

OUTPUT_DIR="./memory_profiles/llm_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR

# ---- Get unique allocated nodes ----
if [[ -z "${PBS_NODEFILE:-}" || ! -f "$PBS_NODEFILE" ]]; then
  echo "ERROR: PBS_NODEFILE is not set or not found."
  exit 1
fi

mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
NNODES="${#NODES[@]}"

if (( NNODES < 2 )); then
  echo "ERROR: Need at least 2 nodes (1 server + >=1 client). Got $NNODES"
  printf "Nodes: %s\n" "${NODES[*]:-<none>}"
  exit 1
fi

SERVER_NODE="${NODES[0]}"
CLIENT_NODES=("${NODES[@]:1}")
NUM_CLIENTS="${#CLIENT_NODES[@]}"
export NUM_CLIENTS
echo "NNODES=$NNODES  NUM_CLIENTS=$NUM_CLIENTS"
echo "Server node: $SERVER_NODE"
echo "Client nodes: ${CLIENT_NODES[*]}"

# ---------- helpers ----------
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no"

run_remote_bg () {
  local host="$1"
  local name="$2"
  local cmd="$3"
  ssh $SSH_OPTS "$host" "bash -lc '
    set -euo pipefail
    $ENV_SETUP
    echo \"[$name] host=\$(hostname) start=\$(date)\"
    echo \"[$name] cmd: $cmd\"
    $cmd
  '" >"logs/${name}.${host}.log" 2>&1 &
}

# Attempt to stop python processes on allocated nodes when job exits
cleanup() {
  echo "[cleanup] stopping remote processes and tunnels..."
  for h in "${NODES[@]}"; do
    ssh $SSH_OPTS "$h" "pkill -f 'run_server_memray.py|run_client_memray.py' || true" >/dev/null 2>&1 || true
    ssh $SSH_OPTS "$h" "pkill -f 'ssh .* -L .*:localhost:50051' || true" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

# ---------- launch server ----------
SERVER_CMD="python memory_profiling/run_server_memray.py \
  --config ${SERVER_CONFIG} \
  --output-dir \"${OUTPUT_DIR}\" \
  --use_optimized_version \
  --num_clients ${NUM_CLIENTS} \
  --model_name \"${MODEL_NAME}\""

echo "Launching server on $SERVER_NODE ..."
run_remote_bg "$SERVER_NODE" "server" "$SERVER_CMD"

# Give server a moment to come up
sleep 8

# ---------- launch clients with per-node SSH tunnel ----------
# Each client node forwards a unique LOCAL port -> server localhost:50051
# Client then connects to localhost:<LOCAL_PORT>
client_idx=0
for node in "${CLIENT_NODES[@]}"; do
  LOCAL_PORT=$((50051 + client_idx + 1))   # 50052, 50053, ...

  CLIENT_CMD="
    # Start SSH tunnel (client node local port -> server node port 50051)
    ssh $SSH_OPTS -o ExitOnForwardFailure=yes -N -f \
      -L ${LOCAL_PORT}:localhost:50051 ${SERVER_NODE};

    # Verify tunnel is listening
    ss -lntp | grep -q \":${LOCAL_PORT}\" || (echo \"Tunnel failed on port ${LOCAL_PORT}\"; exit 1);

    # Run client through the tunnel
    python memory_profiling/run_client_memray.py \
      --config ${CLIENT_CONFIG} \
      --output-dir \"${OUTPUT_DIR}\" \
      --use_optimized_version \
      --num_clients ${NUM_CLIENTS} \
      --client_idx ${client_idx} \
      --server_uri \"localhost:${LOCAL_PORT}\"
  "

  echo "Launching client_idx=${client_idx} on $node (tunnel localhost:${LOCAL_PORT} -> ${SERVER_NODE}:50051) ..."
  run_remote_bg "$node" "client${client_idx}" "$CLIENT_CMD"

  client_idx=$((client_idx+1))
done

echo "Launched server + ${NUM_CLIENTS} clients."
echo "Logs under: $APPFL_DIR/logs/"
echo "Profiles under: $APPFL_DIR/${OUTPUT_DIR}"

# Keep the PBS job alive until everything finishes
wait

echo "Running memory profile analysis..."
python memory_profiling/analyze_profiles.py "$OUTPUT_DIR"

echo "Done at $(date)"
