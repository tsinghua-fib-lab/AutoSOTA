# Code Analysis — ProtoT SOTA Optimization (Paper 5055)

## Evaluation Path
- **Command**: `python3 run_clm.py --MODEL PrototypeAttn --DATASET FineWeb --TRAIN_SIZE 18000 --DEV_SIZE 4000 --TOKENIZER bpe --TOKENIZER_PATH tok/fineweb_bpe_16000.json --VOCAB_SIZE 16000 --SEQ_LEN 256 --BOTTLENECK 256 --LAYERS 6 --TIE_HEAD --R 32 --DROPOUT 0.1 --BATCH 32 --EPOCHS1 10 --LR1 0.002 --USE_LR_SCHEDULER --SEED 234 --TEST_EVALUATE --SAVE_DIR logs/protot_repro`
- **Output format**: `[PrototypeAttn TEST] test_loss=X.XXXX test_ppl=XX.XX`
- **Parser**: `grep -oP "test_ppl=\K[0-9.]+"` from stdout
- **Timeout**: 60 minutes

## Key Files
- `run_clm.py` (588 lines): Main training script with CLI args, training loop, evaluation
- `prototype_attn.py` (1086 lines): Core model — `MixerConfig`, `ProtoBroadcastMixerUpgraded`, `MixerBlock`, `MixerStack`, `ProtoBroadcastLM`
- `utils.py`: Helper functions (SwiGLU, HFCompatConfig, etc.)
- `data_utils.py`: Data loading and BPE tokenizer
- `llama_baseline.py`: LLaMA baseline model
- `mamba.py`: Mamba baseline model

## Config Path
- All config via CLI args in `run_clm.py:parse_args()` (lines 36-107)
- Model creation in `create_model()` (lines 291-313)
- LR scheduler setup (lines 422-430)

## Metric Parser
- Evaluation loop in `evaluate()` (lines 269-288)
- Test evaluation triggered by `--TEST_EVALUATE` (lines 474-477)
- Metric line: `[PrototypeAttn TEST] test_loss=X.XXXX test_ppl=XX.XX`

## Safe Modification Targets
1. `run_clm.py`: Add new CLI args (`--HUB_DROPOUT`, `--LAYERDROP`, `--W_DIVERSITY`, `--W_RECON`, `--TAU_INIT`, `--SHARED_ROUTING_LAYERS`, `--LR_SCHEDULE`, `--CURRICULUM_DROPOUT`)
2. `prototype_attn.py`: 
   - Add `prototype_diversity_loss()` function
   - Add reconstruction loss to forward()
   - Add LayerDrop to MixerStack.forward()
   - Modify MixerStack.__init__() for per-layer configs
   - Expose dead hub indices in aux dict

## Risky Files (do not modify)
- `utils.py` — metric definitions and scoring utilities
- Data loading and tokenizer files in `data_utils.py`
- Test/eval data in `data/FineWeb/`

## Baseline
- test_perplexity = 91.27 (avg over seeds 234, 124, 325)
- Seed 234 = 91.10
- Paper reports 90.5 for ProtoT
- Best baseline LLaMA = 78.7
- Tags: `_baseline` (94cc674), `_best` (94cc674)

## Reusable Resources
- Tokenizer: `tok/fineweb_bpe_16000.json` (vocab 16K)
- Data: `data/FineWeb/train.npz`, `val.npz`, `test.npz` (250M tokens, 360K docs)
- No `/paper_data` mount available
- Cache mounts: `/autosota_cache`, `/datasets`, `/models`
