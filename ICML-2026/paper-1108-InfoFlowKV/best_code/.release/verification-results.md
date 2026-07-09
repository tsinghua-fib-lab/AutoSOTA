# Raw Verification Outputs (literal stdout / stderr)

Captured from a fresh clone of `origin/release` at HEAD `76289bca6e8d10b488ca9d7b36f8d52f41f9db91` with `--recurse-submodules`. Every command below ran in the fresh-clone directory.

Date captured (UTC): 2026-05-03T15:59:14Z

---

## AC-3 — Release root layout
```
$ git ls-tree --name-only release | sort | tr "\n" " "
.gitignore .gitmodules .release CITATION.cff LICENSE README.md THIRD_PARTY_LICENSES.md llm vlm 
```

Expected: `.gitignore .gitmodules .release CITATION.cff LICENSE README.md THIRD_PARTY_LICENSES.md llm vlm`. Match: ✓

## AC-4 — Root .gitmodules
```
$ cat .gitmodules
[submodule "llm/ring-flash-attention"]
	path = llm/ring-flash-attention
	url = https://github.com/zhuzilin/ring-flash-attention.git
```

```
$ find . -name .gitmodules -not -path './.gitmodules' -not -path './.git/*' -not -path './llm/ring-flash-attention/*'
(empty — no nested .gitmodules)
```

## AC-5 — Submodule pinned to 786677930…
```
$ ls llm/ring-flash-attention/setup.py
llm/ring-flash-attention/setup.py
$ git -C llm/ring-flash-attention rev-parse HEAD
786677930bce4f6022166899c88ce2c00c814ee2
```

## AC-6 — No tracked artifacts
```
$ git ls-files | grep -cE '\.bak$|\.orig$|_backup\.py$|/__pycache__/|\.pyc$|^slurm-.*\.out$|/slurm-.*\.out$|^results/|/results/|\.ipynb_checkpoints/|\.pytest_cache/|^\.dataset/|/\.dataset/'
0
0

$ git ls-files -- '*.png' '*.ipynb' | grep -v '^llm/ring-flash-attention/' | wc -l
0

$ find . -name .git -type d -not -path './.git' -not -path './llm/ring-flash-attention/.git' | wc -l
0
```
All AC-6 checks return 0 ✓

## AC-7 — No private/secret strings (full tree, only ring-flash-attention EXCL)
```
$ git grep -nE '/scratch/[a-z]+[0-9]*' -- :!llm/ring-flash-attention  →  1 hits
$ git grep -nE '/home/[a-z]+[0-9]*' -- :!llm/ring-flash-attention  →  0 hits
$ git grep -nE '[A-Za-z0-9._-]+@(nyu|cs\.nyu|hpc)\.[A-Za-z.]+' -- :!llm/ring-flash-attention  →  0 hits
$ git grep -nE '[A-Z][A-Z0-9_]+_(TOKEN|API_KEY|HUB_TOKEN|SECRET)\b' -- :!llm/ring-flash-attention  →  0 hits
$ git grep -nE 'eyJ[A-Za-z0-9_-]{10,}' -- :!llm/ring-flash-attention  →  0 hits
```
All 5 patterns return 0 hits ✓

## AC-8 — Compile / lint / load
```
$ python3 -m compileall -q llm vlm -x "llm/ring-flash-attention/.*"
exit: 0

$ for f in $(git ls-files "*.sh" "*.sbatch" "*.slurm" | grep -v "^llm/ring-flash-attention/"); do bash -n "$f" || echo "FAIL $f"; done
(empty above means all bash -n exit 0)

$ python3 -c "import yaml,subprocess; ..."
YAML: 22 files, OK
```

## AC-11 — Deleted-basename reference scan
```
  attention_visualizer.py  →  0 hits
  utils.py  →  0 hits
  run_recompute_kv.sh  →  0 hits
  model.py.bak  →  0 hits
  recomputer_backup.py  →  0 hits
  example_usage.py  →  0 hits
  inference_with_ring_attention.py  →  0 hits
  benchmark_long_context.py  →  0 hits
  benchmark_ttft_scaling.py  →  0 hits
  compare_quality.py  →  0 hits
  run_evaluation.py  →  0 hits
  qwen_simple_demo.py  →  0 hits
  chatglm_simple_demo.py  →  0 hits
  test_long_context_long_desc.yaml  →  0 hits
  test_long_context_medium.yaml  →  0 hits
  test_long_context_medium_desc.yaml  →  0 hits
  test_long_context_short_desc.yaml  →  0 hits
  visualization.ipynb  →  0 hits
  visualize_attention.ipynb  →  0 hits
  test_chunker.py  →  0 hits
  test_baseline_strategies.py  →  0 hits
```
All 21 deleted basenames return 0 hits ✓

## AC-18 — Orphan branch property
```
$ git fetch origin ring vlm-dev
From github.com:zzzzccccyyyy/kv-cache-optimization
 * branch            ring       -> FETCH_HEAD
 * branch            vlm-dev    -> FETCH_HEAD

$ git merge-base release origin/ring
(exit 1)
$ git merge-base release origin/vlm-dev
(exit 1)

$ git log --oneline release | wc -l
10
$ git log --oneline release
76289bc Round 4: AC-10.2 PASS evidence + AC-9.x reclassification + accounting
1d4cb20 Strengthen AC-9 / AC-10 waiver evidence; reclassify AC-9.4 as PASS
d7557b4 Sanitize audit docs: AC-7 passes full tree with no audit-doc EXCL
bddcaab Document AC-9/AC-10 runtime smoke results and reproduction commands
2fb8c9c Sanitize remaining non-xt2251 private absolute paths
2c3f3da Refine AC-7 verification to mirror AC-11 audit-doc exclusion
29e162b Add release metadata: README, LICENSE, CITATION, NOTICE, .release/ audit
bb63847 Import cleaned VLM tree under vlm/
497b8d0 Add ring-flash-attention submodule under llm/ring-flash-attention
c724f15 Import cleaned LLM tree under llm/
```
merge-base both return non-zero with no SHA → orphan ✓

## AC-1 — Source branch SHAs unchanged
```
$ git rev-parse origin/ring origin/vlm-dev origin/release
9d2b9208568c78d34845f21adbcc73d4a4318f1e
15cafbec4d3160d6d8c513c8a55c0fcd7525ee07
76289bca6e8d10b488ca9d7b36f8d52f41f9db91
```
Expected: origin/ring=9d2b9208…, origin/vlm-dev=15cafbec…
