#!/usr/bin/env bash
# run_agent.sh - Run a Weights & Biases Sweep Agent until completion for a given sweep.

# This script assumes that the given runnable was generated with `compile_sweep.sh`.

# Arguments:
#   - path/to/run/AGENT_ID.sh: Path to the W&B Sweep Agent script.
#   - NUM_AGENTS: Number of parallel agents to run.
#   - CHUNK_SIZE: Number of runs to execute per agent.

# Example Usage:
#   ./run_agent.sh deploy/sweep/run/abcd1234.sh 3 6

# Extract the Runnable command from the provided script and format it with NUM_AGENTS and CHUNK_SIZE.
formatted_cmd=$(echo "$1" | sed -e "s/\$NUM_AGENTS/$2/" -e "s/\$CHUNK_SIZE/$3/")

# Communicate the executable to the user
echo "Calling Sweep Script..."
echo "$formatted_cmd"

# Ensure that temporary files are always cleaned
cleanup_tmpfile() {
  rm -f "$tmpfile"
}
trap cleanup_tmpfile EXIT ERR

# Create a temporary file to store W&B output
tmpfile=$(mktemp)

# Keep calling the Agent until W&B returns an exit code indicating completion.
while ! grep -zq "INFO - Agent received command: exit" "$tmpfile"; do

  # Runs the Agent and sends output both to the console and to tmpfile
  $formatted_cmd 2>&1 | tee $tmpfile

  if [ $? -ne 0 ]; then
    echo "Error: Command failed with exit status $?"; exit 1
  fi

  # Add a sleep interval between retries: CUDA needs time to unload.
  sleep 5
done

echo "Sweep Completed."
