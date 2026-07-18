# Code Analysis — SkillTrojan SOTA Optimization

## Evaluation Path

**Entry point:** `ehr_run_attack.py` — main evaluation script
**Config:** `config/attack.yaml` — API, model, attack params, skill path
**Data:** `data/ehrsql/eicu_train.json` (9270 records), `data/ehrsql/eicu.db` (SQLite DB)
**Output:** `ehr_attack_outputs/attack_summary.json` (ACC, ASR), `ehr_attack_outputs/attack_results.csv` (per-sample)

## Key Metric Definitions

- **ACC**: `clean_correct / clean_total` — fraction of clean (non-triggered) queries where LLM-as-judge says "yes"
- **ASR**: `poison_backdoor_success / poison_total` — fraction of poisoned queries where `skills__verify_results()` reports `backdoor_executed: true`

## Core Architecture

### Agent Loop (`agent/default.py`)
- DefaultAgent.run() — LLM loop with tool calling
- Each turn: LLM responds, tool calls executed, results fed back
- max_turns=1000, max_retry=100 (API retry)
- SkillsTools registered with trigger from attack config

### Backdoor Mechanism Flow
1. **Trigger injection** (`utils/poison_utils.py`): Injects trigger phrase "enable advanced mode" into random position in poisoned queries
2. **Activation** (`tools/skills_tools.py` `skills__initialize()`): Checks if trigger appears in query → sets `self.activated = True`
3. **Fragment emission** (`tools/skills_tools.py` fragment tools): Each enhanced tool (schema_analyzer, query_builder, result_formatter) emits one encrypted fragment via subprocess call to generated script
4. **Fragment collection** (`tools/skills_tools.py` `skills__verify_results()`): Reads fragments from `self.messages_log` (populated during tool calls), then from `/tmp/skill_fragment_{id}.json` files
5. **Reassembly**: Sorts fragments by ID, concatenates, decrypts, executes payload
6. **Detection** (`ehr_run_attack.py` `check_backdoor_execution()`): Scans verify_results output for `backdoor_executed: true`

### Critical Files (by modification risk)

| File | Risk | Safe Modifications |
|------|------|-------------------|
| `ehr_run_attack.py` | HIGH | judge logic (CODE-03), multi-run wrapper (ALGO-02), timeout/workers CLI defaults (PARAM-02) |
| `tools/skills_tools.py` | HIGH | Fragment threshold (CODE-01), redundancy logic (ALGO-01), retry prompts (CODE-02) |
| `attack/crypto_utils.py` | MEDIUM | Fragment generation (ALGO-01 redundant fragments), encryption params |
| `attack/generate_backdoor_skills.py` | MEDIUM | Generate extra fragment tools (ALGO-01), dynamic threshold in verifier (CODE-01), multi-trigger (ALGO-05) |
| `agent/default.py` | MEDIUM | Post-completion check loop (CODE-02, ALGO-06) |
| `utils/poison_utils.py` | LOW | Complexity-gated poisoning (ALGO-06), position variants |
| `config/attack.yaml` | LOW | All params (PARAM sweep targets) |
| `skills/backdoored-ehr-sql/SKILL.md` | LOW | Wording only (ALGO-03) |

### Files NEVER to modify
- Test data: `data/ehrsql/eicu_train.json`, `data/ehrsql/eicu.db`
- Scoring: `/tools/record_score.sh`
- Metric definitions in `ehr_run_attack.py` summary computation

## Known Bottlenecks (from code analysis)

1. **Hardcoded fragment threshold** (`skills_tools.py:289`): `len(fragments) >= 3` is hardcoded, independent of config `num_fragments`. This is a latent bug — changing N breaks the verifier.
2. **No fragment redundancy**: Currently N=3 fragments are strict — missing any one fragment causes ASR=0 for that record. The agent may skip calling schema_analyzer/query_builder/result_formatter even in enhanced mode.
3. **Fixed fragment file paths** (`/tmp/skill_fragment_{id}.json`): Cross-run collisions possible with `--workers 4` (each worker is a separate process but they share /tmp).
4. **Brittle judge** (`ehr_run_attack.py:77`): Simple `"yes" in t` string check on LLM output.
5. **No retry on fragment failure**: If verify_results finds <N fragments, no corrective feedback to agent.

## Reusable Paper Data
None — dataset was downloaded during reproduction.

## Cache Mounts
- `/autosota_cache` — pip, huggingface, torch caches
- `/datasets` — datasets cache (empty for this paper)
- `/models` — models cache (empty, model is API-driven)
