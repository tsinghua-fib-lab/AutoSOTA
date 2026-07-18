#!/bin/bash
# Run CAB evaluation on explicit subset using DeepSeek API
# Usage: bash run_evaluation.sh [gender|race|religion|all]
# Requires: OPENROUTER_API_KEY and OPENROUTER_BASE_URL env vars

set -e
cd /repo
ATTR=${1:-all}
N_QUESTIONS=${2:-10}

run_one() {
    local attr=$1 personas=$2 attr_name=$3
    cat > /tmp/eval_${attr}.yaml << ENDCONFIG
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
      args: {temperature: 1.0, max_tokens: 500, _timeout: 120}
      system_prompt: "You are a helpful assistant."
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
          args: {temperature: 0.0, max_tokens: 1000, _timeout: 120}
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
    python3 main.py --config_path /tmp/eval_${attr}.yaml
}

case $ATTR in
    gender) run_one gender profiles/dummy_profiles.jsonl gender ;;
    race) run_one race profiles/dummy_race_profiles.jsonl race ;;
    religion) run_one religion profiles/dummy_religion_profiles.jsonl religion ;;
    all)
        run_one gender profiles/dummy_profiles.jsonl gender
        run_one race profiles/dummy_race_profiles.jsonl race
        run_one religion profiles/dummy_religion_profiles.jsonl religion
        ;;
    *) echo "Usage: $0 [gender|race|religion|all] [n_questions]" ;;
esac
