# Code Analysis for Paper 3213 (AgentTailor)

## Evaluation Path
- Entry: `experiments/train4mmlu.py` -> `main()` -> `train_mmlu_with_stage3_stats()` -> `train_base.train_all()`
- Config: `experiments/train4mmlu.py::_build_config()`
- Dataset: `MMLUDataset` loads from `/repo/dataset/MMLU/data/dev/*.csv` + `val/*.csv`
- Metric parsing: stdout from Stage3 validation (`Accuracy: XX.XX%`), token stats, wall-clock time

## Key Files
- `experiments/train4mmlu.py` (329 lines): MMLU-specific training script, config, dataset
- `experiments/train_base.py` (2270 lines): Core training loop
- `AgentTailor/prompt/mmlu_prompt_set.py` (199 lines): Agent role definitions, prompt templates
- `AgentTailor/ATNetwork/Critics.py` (797 lines): EPN, Encoder
- `AgentTailor/ATNetwork/Actor.py` (689 lines): Multi-agent graph, edge selection
- `AgentTailor/agents/analyze_agent.py` (63 lines): AnalyzeAgent LLM wrapper
- `AgentTailor/agents/final_decision.py` (190 lines): FinalRefer decision method

## Critical Observations
1. **CODE-02**: `_compute_ranking_loss()` computed but NEVER added to `critic_loss` (line 430: only `mse_loss`)
2. **CODE-03**: Typo "Knowlegable Expert" throughout `mmlu_prompt_set.py`
3. **Bucket terms** computed but unused in critic loss
4. **No per-agent profiling** - individual agent accuracy never tracked
5. **40 training samples** - critic trained on very small dataset
6. **DeepSeek-chat** used instead of gpt-4o

## Safe Modification Targets
- `experiments/train_base.py`: critic loss, training loop, logging
- `experiments/train4mmlu.py`: config, dataset loading, subject metadata
- `AgentTailor/prompt/mmlu_prompt_set.py`: role descriptions, prompt text
- `AgentTailor/ATNetwork/Critics.py`: EPN architecture, dropout
