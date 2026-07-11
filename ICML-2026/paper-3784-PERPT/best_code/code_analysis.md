# Code Analysis — Paper 3784 (Gamba: Bi-Gamba MLM+MEM 4M)

## Evaluation Path
- **Script**: `/repo/eval_clinvar.py` (custom, self-contained)
- **Input**: ClinVar missense variants from `songlab/clinvar_vs_benign` (test split)
- **Processing**: Filter missense → extract 2048bp context → forward+reverse strand eval
- **Output**: `/repo/eval_output/clinvar_results.json` with `auroc_log_likelihood` and `auroc_predicted_conservation`
- **Raw scores**: `/repo/eval_output/clinvar_scores.npz`

## Train/Inference Path
- **Training**: `src/caduceus_train.py` — FSDP distributed training with SSM (Mamba-based)
- **Config**: `configs/jamba-small-240mammalian.json` — 4M params, 2048bp context, lr=1e-4
- **Model**: `/models/bigamba-dual-step44000/` — HF format, loaded via `AutoModel.from_pretrained(trust_remote_code=True)`
- **Architecture**: Bi-Gamba (bidirectional Mamba SSM) with RCPS embedding, dual MLM+MEM heads
- **Model class**: `CaduceusHFModel` wrapping `_CaduceusConservationForMaskedLM`

## Config Path
- **Training**: `configs/jamba-small-240mammalian.json`
- **Model**: `/models/bigamba-dual-step44000/config.json` (HF format, `model_type: caduceus_custom`)
- **Key params**: d_model=256, n_layer=8, vocab_size=12, training_task=dual, last_step=44000

## Metric Parser
- `eval_clinvar.py` writes JSON directly. Metrics parsed from `clinvar_results.json` keys:
  - `auroc_log_likelihood` → "AUROC (Log-likelihood)"
  - `auroc_predicted_conservation` → "AUROC (Predicted conservation)"

## Reusable Resources
- `/models/bigamba-dual-step44000/` — 27MB safetensors, Bi-Gamba dual checkpoint at step 44K
- `/datasets/hg38.ml.fa` — 2.9GB human genome reference
- No `/paper_data` mount; no additional pretraining data available in container

## Architecture Details (from audit)
- Model uses `_CaduceusConservationForMaskedLM` with RCPS (Reverse Complement Parameter Sharing)
- Forward outputs: `logits` [B,T,12] (nucleotide LM head) + `scaling_logits` [B,T,2] (conservation head)
- `scaling_logits[:,:,0]` = mean predicted evolutionary rate (ERP), `[:,:,1]` = log-variance
- Conservation head: `nn.Linear(256, 2)` for non-RCPS (default) or `RCPSConservationHead` for RCPS
- LLR computation: per-position log_softmax at variant position, strand-averaged
- Strand handling: model processes forward sequence AND reverse complement separately

## AUDIT Results (CODE-01 + CODE-02)

### CODE-01: Position/Strand Audit — PASS (no bugs found)
- Position centering: correct — `pos0 = pos - 1`, `target_pos = window_size//2`, `start0 = pos0 - target_pos`
- Strand complement: correct — `COMP` maps A↔T, C↔G; `revcomp()` uses `str.maketrans`
- Reverse strand position: correct — `tpos_rev = window_size - 1 - tpos`
- Allele selection: correct — ref/alt tokens extracted at correct positions for both strands
- Coordinate system: correct — 1-based VCF pos → 0-based python indexing
- Chromosome naming: `normalize_chrom()` adds "chr" prefix if missing

### CODE-02: Numerical Audit — PASS (minor issue found)
- Log-space operations are numerically stable (float32 conversion)
- LLR = log_softmax(alt) - log_softmax(ref) is correctly formulated
- Conservation score taken from channel 0 of scaling_logits (correct for variant scoring)
- No log(0) risk since log_softmax handles this internally
- **Potential improvement**: Temperature scaling for log_softmax could sharpen/soften distributions

### CODE-03: Genome FASTA Audit — PASS
- pyfaidx-based Fasta loading is standard and reliable
- Chromosome name normalization handles chr prefix
- Boundary check: verifies seq length == window_size
- Reference allele verification: `seq[target_pos] != ref` is checked and skipped if mismatch

### CODE-04: Determinism Audit — FIX NEEDED
- No fixed random seed set in eval script — model.eval() is called but no torch seed
- `torch.use_deterministic_algorithms` not enabled
- Batch processing order depends on variant ordering which could affect float accumulation

## Safe Modification Targets
1. `eval_clinvar.py`: Strand combination strategy, temperature scaling, determinism fixes
2. Ensemble scoring: Combine LLR + conservation with learned weights
3. Context window: Can be modified via `--window_size` argument
4. Batch size: Can be modified via `--batch_size` argument
5. Model loading: Model architecture itself can be loaded but not easily fine-tuned without training data

## Risky Files (do not modify)
- `eval_clinvar.py` metric computation logic (AUROC from sklearn)
- Dataset loading (songlab/clinvar_vs_benign test split)
- Genome reference (/datasets/hg38.ml.fa)
- Model weights (/models/bigamba-dual-step44000/model.safetensors)

## Known Levers for Optimization
- Temperature scaling (inference-time, low cost)
- Strand combination alternatives (arithmetic mean → max/min/geometric)
- Ensemble LLR + conservation scoring
- Determinism improvements (reduce variance)
- If training data available: extended pretraining, LR sweep, loss weight tuning
