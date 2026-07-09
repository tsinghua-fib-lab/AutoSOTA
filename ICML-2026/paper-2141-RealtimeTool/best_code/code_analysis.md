# Code Analysis — Paper 2141 (RealtimeTool)

## Evaluation Pipeline
- **Entry point**: reproduce.py (identical to eval_bfcl.py)
- **Command**: python3 reproduce.py --model /models/RT-Qwen2.5-0.5B
- **Model**: RT-Qwen2.5-0.5B (0.5B Qwen2.5 fine-tuned for multi-head parallel decoding)
- **Inference**: vLLM with prefix caching, 4096 max_model_len, 128 max_tokens per head
- **Heads**: function + arg1-6 (7 heads for eval)
- **Prompt**: V1 format (multi-head instruction system prompt)
- **Data**: BFCL v4 Non-Live, 600 samples (200 Multiple + 400 Simple Python)

## Key Files
| File | Role | Safe to Modify |
|------|------|---------------|
| eval_bfcl.py = reproduce.py | Main eval script | Yes |
| 02_server.py | Server, has v2 prompt | Reference only |
| debug_*.py | Debug scripts | Reference only |
| prompts/v1_system.txt | System prompt template | Yes |
| gorilla/.../data/ | Test data | NO |
| gorilla/.../possible_answer/ | Ground truth | NO |

## Safe Modification Targets
1. build_prompt_v1() — prompt engineering
2. run_evaluation() — inference strategy, error logging
3. normalize_value()/values_equal() — value comparison
4. clean_output() — null detection
5. get_param_order() — parameter ordering
6. New functions for schema verification, coherence checking

## Baseline Metrics
- Overall Accuracy: 87.2% (523/600)
- Function Accuracy: 100.0% (600/600)
- All 77 errors are argument-level
