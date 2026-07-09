# Cleanup Inventory — `release` branch

Authoritative log of every file that was deleted, audited, kept, moved, or promoted while building this orphan release branch from `origin/ring` (`9d2b9208…`) and `origin/vlm-dev` (`15cafbec…`).

Audit method: `git grep -l --full-name -- <basename>` against each prep tree. "NO REFERENCES" means zero hits in any kept file.

The release tree is constructed by `rsync`-importing two cleaned prep branches:
- `release-prep-llm` (HEAD `0fed0720`) → `llm/`
- `release-prep-vlm` (HEAD `759c139d`) → `vlm/`

`ring-flash-attention` is added as a git submodule under `llm/ring-flash-attention/` (pinned to `786677930…`, v0.1.8); it was not part of the `rsync` import.

---

# Part 1 — LLM side (sourced from `origin/ring` → `release-prep-llm`)

## Section A — Top-level scratch / unused utilities (delete)
| Path (in source) | Prior Status | Action | Reason | Audit Evidence |
|------------------|--------------|--------|--------|----------------|
| `attention_visualizer.py` | tracked | delete | NO REFERENCES from kept code/docs | `git grep -l attention_visualizer` → empty |
| `utils.py` | tracked | delete | NO REFERENCES | `git grep -l utils.py` → empty |
| `run_recompute_kv.sh` | tracked | delete | NO REFERENCES | `grep -lr run_recompute_kv` → empty |

## Section B — Stale model backups (delete)
| Path | Prior Status | Action | Reason |
|------|--------------|--------|--------|
| `models/qwen/model.py.bak` | tracked | delete | `.bak` backup; no live import |
| `models/chatglm/model.py.bak` | tracked | delete | `.bak` backup; no live import |
| `models/qwen/kv_cache/recomputer_backup.py` | tracked | delete | `_backup` superseded; current `recomputer.py` canonical |

## Section C — Demo / superseded scripts (delete)
| Path | Reason |
|------|--------|
| `scripts/inference_with_ring_attention.py` | Superseded by `eval_longbench.py`'s `ring_attention` method |
| `scripts/benchmark_long_context.py` | Superseded by `sweep_benchmark.py` |
| `scripts/benchmark_ttft_scaling.py` | Superseded by `sweep_benchmark.py` |
| `scripts/compare_quality.py` | Superseded by `eval_longbench.py` + `run_eval_*.sh` |
| `scripts/run_evaluation.py` | Older driver replaced by `run_eval_*.sh` |
| `scripts/qwen_simple_demo.py` | Demo only; not referenced |
| `scripts/chatglm_simple_demo.py` | Demo only; not referenced |
| `models/example_usage.py` | Example file; not imported |

## Section D — Stale long-context configs (delete)
| Path | Reason |
|------|--------|
| `configs/test_long_context_long_desc.yaml` | NO REFERENCES |
| `configs/test_long_context_medium.yaml` | NO REFERENCES |
| `configs/test_long_context_medium_desc.yaml` | NO REFERENCES |
| `configs/test_long_context_short_desc.yaml` | NO REFERENCES |

## Section E — Long-context configs (KEEP)
| Path | Referenced by |
|------|---------------|
| `configs/test_long_context.yaml` | `scripts/run_longbenchv2.sbatch` |
| `configs/test_long_context_long.yaml` | `scripts/run_longbenchv2_long.sbatch` |
| `configs/test_long_context_short.yaml` | `scripts/run_eval_short.sbatch` |

## Section F — Untracked artifacts (no-op; covered by `.gitignore`)
These never enter the orphan release. The `release` branch's `.gitignore` plus `llm/.gitignore` ensure they are not re-tracked.
| Pattern | Notes |
|---------|-------|
| `slurm-*.out` (47 root-level files) | SLURM stdout/stderr |
| `results/` (hundreds of subdirs) | Experiment outputs |
| `__pycache__/` | Python bytecode |
| Root scratch tests `test_crash_sample.py`, `test_exact_dims.py`, `test_flashinfer_mp.py`, `test_mp_real_data.py`, `test_real_data.py` | Debug-only |

## Section G — Sanitize private absolute paths (modify, do not delete)
22 tracked LLM files were modified. All edits are pure path-string replacements; no function/signature/runtime-default changes.

| Path | Edit |
|------|------|
| 11 YAMLs in `configs/` (2wikimqa_asc/_desc, hotpotqa_asc/_desc/_timing/_timing_desc, musique_asc/_desc, test_long_context, test_long_context_long, test_long_context_short) | private model paths replaced with `/path/to/Qwen3-14B` literal placeholder |
| `scripts/benchmark_single_gpu.py` (line 23 default) | argparse default updated to `/path/to/Qwen3-14B` |
| `scripts/sweep_benchmark.py` (line 28 default) | same |
| `scripts/run_eval_stride1.sh`, `run_eval_stride8.sh`, `run_eval_all_methods.sh`, `run_eval_longbenchv2.sh` | `MODEL=<private>` → `MODEL="${MODEL_PATH:-/path/to/Qwen3-14B}"` |
| `scripts/run_baseline_short_test.sbatch`, `run_baseline_medium_test.sbatch`, `run_eval_short.sbatch`, `run_longbenchv2.sbatch`, `run_longbenchv2_long.sbatch` | private conda activate path → `${CONDA_BASE:-$HOME/miniconda3}/bin/activate`; private repo path → `${REPO_ROOT:-$HOME/kv-cache-dev}`; personal `--mail-user=...` SBATCH directives removed |

## Section H — Documentation patches
| Path | Edit |
|------|------|
| `README.md`, `CLAUDE.md` | Inspected; no references to deleted basenames remain. The deleted `scripts/benchmark_ring_attention.py` was already removed in an earlier commit on `ring`. |

---

# Part 2 — VLM side (sourced from `origin/vlm-dev` → `release-prep-vlm`)

## Section A — Tracked notebooks (delete)
| Path | Action | Reason |
|------|--------|--------|
| `visualization.ipynb` | delete | NO REFERENCES; large notebook |
| `visualize_attention.ipynb` | delete | NO REFERENCES; large notebook |

## Section B — Stale tests (delete)
| Path | Action | Reason |
|------|--------|--------|
| `tests/test_chunker.py` | delete | Not part of any documented test suite |
| `tests/` (now empty) | delete | Empty directory |
| `scripts/test_baseline_strategies.py` | delete | NO REFERENCES |

## Section C — Broken gitlinks (delete)
Broken at `origin/vlm-dev`: tracked as submodule pointers (mode 160000) but no `.gitmodules` entries, so a fresh recursive clone reports "fatal: no submodule mapping found".

| Path | Action | Reason |
|------|--------|--------|
| `vlm/src/cotracker` (gitlink) | delete | Broken gitlink; no kept VLM script imports `cotracker` |
| `vlm/src/vlmeval` (gitlink) | delete | Broken gitlink; VLMEvalKit is installed externally via `sys.path.insert(0, "/path/to/VLMEvalKit")` in `scripts/eval_vlmeval.py` |

## Section D — DEC-3: local edits in main VLM checkout (NOT carried into release)
The main VLM workspace (the original VLM repo checkout used by the cleanup author) had two locally-modified files. Per the user-decided rule "inspect diff; commit only if cosmetic":

| File | Local Diff | Verdict | Action |
|------|-----------|---------|--------|
| `scripts/eval_existing.py` | `dataset.evaluate(pred_file)` → `dataset.evaluate(pred_file, model='exact_matching')` | NON-COSMETIC (changes evaluator runtime behavior) | DISCARD |
| `scripts/eval_vlmeval.py` | Same `model='exact_matching'` argument added | NON-COSMETIC | DISCARD |

Effect: `release-prep-vlm` was created from `origin/vlm-dev` (not the local mod state), so neither edit enters the release tree.

## Section E — Untracked artifacts (no-op for tracked tree; covered by `.gitignore`)
| Pattern | Notes |
|---------|-------|
| `kv_recompute_pipeline/` (root, untracked in original VLM checkout) | LLM-side leftover; NO kept VLM script imports it; not carried forward |
| Root `chunked_prefill_comparison.png`, `ttft_scaling.png` | Untracked images; not carried forward |
| `vlmeval_output/`, `logs/`, `results/`, `__pycache__/`, `.dataset/` | Run artifacts |
| `scripts/test_*.py`, `scripts/debug_*.py`, `scripts/benchmark_chunked_prefill.py`, `scripts/benchmark_ttft_scaling.py` | Untracked in `origin/vlm-dev` |
| `configs/debug_timing.yaml`, `test_varlen.yaml`, `verify_baseline.yaml`, `hotpotqa_flashinfer.yaml` | Untracked debug configs |

## Section F — Sanitize private absolute paths (modify, do not delete)
11 tracked VLM files were modified. All edits are pure path-string replacements.

| Path | Edit |
|------|------|
| 8 YAMLs in `configs/` (blink_counting, blink_jigsaw, chartqa, docvqa, mathvista, mmbench, ocrbench, realworldqa) | private absolute paths replaced with `/path/to/...` placeholders for `model`, `cache_dir`, `dataset_dir`, `output_dir` |
| `scripts/eval_existing.py`, `scripts/eval_vlmeval.py` | private VLMEvalKit `sys.path.insert(...)` argument replaced with `"/path/to/VLMEvalKit"`; argparse model defaults updated |
| `scripts/eval_single.slurm` | Path placeholders for `${MODEL_PATH}`, `${DATASET_DIR}`, `${OUTPUT_DIR}`, `${REPO_ROOT}`, `${CONDA_BASE}`, `${VLMEVALKIT_DIR}`, `${CONDA_ENV_DIR}`; personal `--mail-user` directive removed |

## Section G — Kept core (no action)
- `models/qwen/{kv_cache,patches}/`, `models/__init__.py`, `models/base.py`, `models/chatglm/`
- `benchmarks/` (base + blink/chartqa/docvqa/mathvista/mmbench/ocrbench/realworldqa)
- `inference/runner.py`
- `scripts/evaluate.py`, `scripts/inference_with_recompute_kv.py`, `scripts/run_blink.py`, `scripts/qwen3_vlm_inference.py`, `scripts/eval_vlmeval.py`, `scripts/eval_existing.py`
- SLURM driver: `scripts/eval_single.slurm` (the `eval_benchmark.slurm` and `eval_all_strategies.sh` referenced earlier turned out not to be tracked at `origin/vlm-dev`; only `eval_single.slurm` is)
- `requirements.txt`, `README.md`, `CLAUDE.md`, `AGENTS.md`

---

# Part 3 — Release-level operations (orphan branch construction)

## Section O1 — Worktree topology
| Role | Branch | Purpose |
|------|--------|---------|
| `<orig-vlm-checkout>` | `vlm-dev` | Original VLM checkout; UNTOUCHED |
| `<orig-llm-checkout>` | `ring` | Original LLM checkout; carries only empty marker commits for round boundaries |
| `<llm-cleanup-worktree>` | `release-prep-llm` | LLM cleanup commits (off `origin/ring`); kept LOCAL per DEC-2 default |
| `<vlm-cleanup-worktree>` | `release-prep-vlm` | VLM cleanup commits (off `origin/vlm-dev`); kept LOCAL per DEC-2 default |
| `<release-build-worktree>` | `release` (orphan) | Final release tree; this branch is what gets pushed to `origin/release` |

The `<...>` placeholders denote per-author paths on the cleanup author's machine. Concrete absolute paths used during this prep are intentionally not recorded so the audit trail does not embed private filesystem layout.

## Section O2 — Release-branch commit history
The orphan `release` branch contains exactly four content commits plus this metadata commit:
1. `c724f15 Import cleaned LLM tree under llm/`
2. `497b8d0 Add ring-flash-attention submodule under llm/ring-flash-attention`
3. `bb63847 Import cleaned VLM tree under vlm/`
4. `<metadata commit, this round>: Add release metadata (root README, LICENSE, CITATION, NOTICE, .gitignore, .release/)`

All four commits have NO parent reachable from `origin/ring` or `origin/vlm-dev` — the orphan property satisfies AC-18.

## Section O3 — Files added at release root (not in either prep tree)
| Path | Source | Purpose |
|------|--------|---------|
| `README.md` | newly authored | Release overview, source SHAs, layout, quickstart, env-var docs |
| `LICENSE` | newly authored | MIT license for original code |
| `CITATION.cff` | newly authored | Paper + dependency citations |
| `THIRD_PARTY_LICENSES.md` | newly authored | Attribution for `ring-flash-attention` and external models/benchmarks |
| `.gitignore` | newly authored | Release-level ignore rules |
| `.gitmodules` | created by `git submodule add` | Submodule registry: `llm/ring-flash-attention` |
| `.release/source-shas.txt` | this round | Pre-/post-push verification anchors |
| `.release/cleanup-inventory.md` | this file | Authoritative cleanup log |
| `.release/verification.md` | this round | Reproducible AC verification commands |

## Section O4 — Files NOT in the release that ARE in the source branches
| Source branch | Path | Reason for exclusion |
|---------------|------|----------------------|
| `origin/ring` | All deletions in Sections A–D above | Dead code / NO REFERENCES |
| `origin/ring` | `ring-flash-attention/` (root) | Re-attached at `llm/ring-flash-attention` via `git submodule add` |
| `origin/vlm-dev` | All deletions in VLM Sections A–C above | Dead code / NO REFERENCES / broken gitlinks |
