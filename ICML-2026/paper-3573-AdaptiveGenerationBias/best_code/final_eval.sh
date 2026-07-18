#!/bin/bash
set -e
cd /repo
rm -rf cab_download/explicit/model_evals/model_evals_*

export OPENROUTER_API_KEY="YOUR_API_KEY"
export OPENROUTER_BASE_URL="https://api.deepseek.com/v1"

run_attr() {
    local attr=$1
    local personas=$2
    local attr_name=$3

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
      args:
          temperature: 1.0
          max_tokens: 1000
          _timeout: 180
      system_prompt: "You are a helpful assistant."
      max_workers: 2
  persona_path: "${personas}"
  persona_model:
      name: "local_replace"
      provider: "local_replace"
      args:
          temperature: 0.0
      system_prompt: ""
  judge_config:
      judge_model:
          name: "deepseek-chat"
          provider: "openrouter"
          args:
              temperature: 0.0
              max_tokens: 2000
              _timeout: 180
          max_workers: 2
      judge_type: "indiv_comparative"
      judge_attribute: "${attr_name}"
  conversation_turn_length: 2
  per_turn_assistant_messages: 3
  per_turn_user_messages: 1
  pairing_strategy: "random"
  outpath_extension: "model_evals"
  max_questions_per_iteration: 30
ENDCONFIG

    echo "=== ${attr} start: $(date) ==="
    python3 -u main.py --config_path /tmp/eval_${attr}.yaml > /tmp/eval_${attr}_final.log 2>&1
    local ret=$?
    echo "=== ${attr} done: $(date), exit=${ret} ==="
    echo "Errors: $(grep -c 'Exception' /tmp/eval_${attr}_final.log || echo 0)"
    grep -E "Completed|Summary|Mean" /tmp/eval_${attr}_final.log | tail -5
}

run_attr "gender" "profiles/dummy_profiles.jsonl" "gender"
run_attr "race" "profiles/dummy_race_profiles.jsonl" "race"
run_attr "religion" "profiles/dummy_religion_profiles.jsonl" "religion"

echo "ALL EVALUATIONS COMPLETE at $(date)"
