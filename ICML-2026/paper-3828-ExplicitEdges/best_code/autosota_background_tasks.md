# SOTA Background Tasks Ledger

## Task: iter-1
- **ID**: iter-1
- **Idea**: CODE-1 (fix confidence) + ALGO-5 (source text) + ALGO-6 (dynamic extractor)
- **Command**: cd /repo/inses && unset ALL_PROXY all_proxy && export DEEPSEEK_API_KEY="[REDACTED]" && export HF_HOME="/autosota_cache/hf" && export QDRANT_HOST="localhost" && timeout 1800 python3 rag_router.py --dataset 2wiki --sample_size 100 --llm_provider deepseek --model deepseek-chat
- **Working dir**: /repo/inses
- **Log path**: /repo/iter1_eval.log
- **Start time**: 2026-07-15T21:02:00Z
- **Deadline**: 2026-07-15T21:32:00Z
- **Expected output**: EM and LLM Judge scores on 100 samples
- **Score row**: iter=1, idea_id=CODE-1+ALGO-5+ALGO-6
- **Status**: running
