# FuseFSS SOTA Code Analysis

## Evaluation Path

- **Script**: `scripts/repro_eval_latest.py`
- **Binary pair**: Sigma baseline + FuseFSS+Sigma, run as two-party protocol (dealer + evaluator)
- **Task**: `bert-tiny:128` (BERT-tiny, seq_len=128, batch_size=1)
- **Metrics parsed from**: `results/eval/<run>/<party>/dealer.txt` and `evaluator.txt`
- **Output**: `results/eval/results.json` (median_results), `results/eval/summary.csv`

## Metric Parser

- `parse_dealer(path)`: Regex extracts `Total time=(\d+) us` -> keygen_us, `Key size=(\d+) B` -> key_bytes
- `parse_evaluator(path)`: Regex extracts `Total time=(\d+) us` -> online_ms, `Comm time=(\d+) us`, `Total Comm=(\d+) B` -> comm_gb
- `collect_pair_result()`: Takes max of party 0 and party 1 values
- Key conversions: online_ms = total_us/1000, comm_gb = total_comm_bytes/(1024^3), keygen_s = keygen_us/1e6, key_gb = key_bytes/(1024^3)

## Config Path

- Env vars set in `scripts/repro_eval_latest.py` lines 730-752:
  - `SIGMA_MEMPOOL_DISABLE` (default "1"), `SIGMA_PINNED_COMM_BUFS` (default "0"), `SIGMA_PINNED_KEYBUF` (default "0")
  - `OMP_NUM_THREADS` = `--threads` (default 64)
  - `FUSEFSS_NEXP_BITS`=10, `FUSEFSS_INV_BITS`=10, `FUSEFSS_RSQRT_BITS`=9
  - `FUSEFSS_SOFTMAX`=1, `FUSEFSS_LAYERNORM`=1, `FUSEFSS_ACTIVATION`=1
  - `--fusefss-sigma-generic` flag for alternative code path
- Build: `-DCMAKE_CUDA_ARCHITECTURES=80`, Release, `-O3 -DNDEBUG`, `--use_fast_math`

## Safe Modification Targets

- **CODE (config-only, no rebuild)**:
  - `scripts/repro_eval_latest.py` lines 734-736: Enable pinned memory and memory pool
  - `scripts/repro_eval_latest.py` line 737: OMP_NUM_THREADS (via --threads)
  - `scripts/repro_eval_latest.py` lines 748-752: FUSEFSS precision bits

- **ALGO (requires rebuild)**:
  - `src/cuda/pfss_kernels.cu`: Kernel launch configs (block size, grid size)
  - `src/cuda/gpu_backend.cu`: GpuSufProgram::eval ordering
  - `third_party/EzPC_vendor/GPU-MPC/fss/gpu_aes_table.h`: AES table placement
  - `third_party/EzPC_vendor/GPU-MPC/experiments/sigma/sigma.cu` line 948: bw=37

## Risky Files (avoid modifying)

- `scripts/repro_eval_latest.py` metric parsing (lines 180-270)
- `third_party/EzPC_vendor/GPU-MPC/` protocol logic
- Any cryptographic primitive implementations
- Test data/splits (none for performance eval)

## Repository State

- Git commit: `372e733` (upstream), with baseline at `dfad0c8`
- Branch: `_baseline` tracks baseline, `main` for development
- Compilation artifacts in `build/`
- No paper data mount; model weights are random (performance benchmark only)
