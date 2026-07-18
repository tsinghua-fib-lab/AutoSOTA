#!/bin/bash
# Configurable CAB evaluation script (no-proxy version)
set -e
cd /repo

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-YOUR_API_KEY}"
export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://api.deepseek.com/v1}"

# Unset proxy to avoid connection issues
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
MAX_TOKENS="${MAX_TOKENS:-500}"
TEMPERATURE="${TEMPERATURE:-1.0}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1000}"
N_QUESTIONS="${N_QUESTIONS:-10}"
ATTRS="${ATTRS:-gender race religion}"
SUFFIX="${SUFFIX:-$(date +%Y%m%d_%H%M%S)}"

echo "=== Configurable Eval: suffix=$SUFFIX ==="
echo "  SYSTEM_PROMPT: ${SYSTEM_PROMPT:0:80}..."
echo "  MAX_TOKENS=$MAX_TOKENS TEMPERATURE=$TEMPERATURE JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS"
echo "  N_QUESTIONS=$N_QUESTIONS ATTRS=$ATTRS"
echo "  Proxy: UNSET (direct connection)"

declare -A PERSONA_MAP
PERSONA_MAP[gender]="profiles/dummy_profiles.jsonl"
PERSONA_MAP[race]="profiles/dummy_race_profiles.jsonl"
PERSONA_MAP[religion]="profiles/dummy_religion_profiles.jsonl"
declare -A ATTR_NAME_MAP
ATTR_NAME_MAP[gender]="gender"
ATTR_NAME_MAP[race]="race"
ATTR_NAME_MAP[religion]="religion"

total_ok=0
total_fail=0

for attr in $ATTRS; do
    personas="${PERSONA_MAP[$attr]}"
    attr_name="${ATTR_NAME_MAP[$attr]}"
    
    cat > /tmp/eval_${attr}_${SUFFIX}.yaml << ENDCONFIG
output_dir: "results"
seed: 42
task: "MODELEVAL"
store: false
task_config:
  run_paths:
    - "cab_download/explicit/${attr}.jsonl"
  eval_models:
    - name: "deepseek-chat"
      provider: "openrouter"
      args: {temperature: ${TEMPERATURE}, max_tokens: ${MAX_TOKENS}, _timeout: 120}
      system_prompt: "${SYSTEM_PROMPT}"
      max_workers: 2
  persona_path: "${personas}"
  persona_model:
      name: "local_replace"
      provider: "local_replace"
      args: {temperature: 0.0}
      system_prompt: ""
  judge_config:
      judge_model:
          name: "deepseek-chat"
          provider: "openrouter"
          args: {temperature: 0.0, max_tokens: ${JUDGE_MAX_TOKENS}, _timeout: 120}
          max_workers: 2
      judge_type: "indiv_comparative"
      judge_attribute: "${attr_name}"
  conversation_turn_length: 2
  per_turn_assistant_messages: 3
  per_turn_user_messages: 1
  pairing_strategy: "random"
  outpath_extension: "model_evals"
  max_questions_per_iteration: ${N_QUESTIONS}
ENDCONFIG

    echo "=== ${attr} start: $(date) ==="
    if python3 -u main.py --config_path /tmp/eval_${attr}_${SUFFIX}.yaml > /tmp/eval_${attr}_${SUFFIX}.log 2>&1; then
        echo "=== ${attr} done: $(date), OK ==="
        total_ok=$((total_ok + 1))
    else
        echo "=== ${attr} done: $(date), FAIL (exit=$?) ==="
        total_fail=$((total_fail + 1))
        tail -20 /tmp/eval_${attr}_${SUFFIX}.log
    fi
done

echo "=== ALL DONE: ${total_ok} OK, ${total_fail} FAIL at $(date) ==="
