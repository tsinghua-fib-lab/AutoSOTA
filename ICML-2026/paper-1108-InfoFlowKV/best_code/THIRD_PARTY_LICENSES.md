# Third-Party Licenses and Attributions

This release contains and depends on third-party code. Original works authored for the paper are licensed under the project `LICENSE`. Third-party components retain their own license terms.

## Bundled as a git submodule

### ring-flash-attention

- **Path in this release**: `llm/ring-flash-attention/`
- **Upstream**: <https://github.com/zhuzilin/ring-flash-attention>
- **Pinned commit**: `786677930bce4f6022166899c88ce2c00c814ee2` (release v0.1.8)
- **License**: see `llm/ring-flash-attention/LICENSE` after `git clone --recurse-submodules`. Upstream is licensed under Apache-2.0 (consult upstream for the authoritative text).
- **Attribution**: Used as the underlying ring-attention sequence-parallel attention kernel for the LLM-side `ring_attention` and `sp_guided_recompute`/`sp_cacheblend`/`sp_lego` baselines.

## Used as upstream dependencies (not redistributed)

The following are **not** bundled into this release. They must be installed by the user (e.g., via `pip install`, `git clone`, or HuggingFace Hub download). Their licenses apply to their own distributions.

### Models
- **Qwen3 (Qwen3-14B, etc.)** — used by the LLM pipeline. Obtain from the official source. Cite the Qwen3 technical report.
- **Qwen3-VL-8B-Instruct** — used by the VLM pipeline. Obtain from the official source.
- **Llama** — supported by the LLM pipeline (`llm/models/llama/`). Obtain from the official source under Meta's license terms.
- **ChatGLM** — supported by the LLM pipeline (`llm/models/chatglm/`). Obtain from the official source.

### Benchmarks
- **LongBench v1** — JSONL samples are bundled under `llm/inputs/{hotpotqa,2wikimqa,musique}.jsonl` for reproducibility (these are subsets of the public LongBench v1 dataset). Cite the original LongBench paper.
- **LongBench v2** — loaded at runtime from <https://huggingface.co/datasets/zai-org/LongBench-v2>. Cite the original work.
- **BLINK** — used by the VLM benchmarks. Cite the original work.
- **VLMEvalKit** — used as an external evaluation framework by `vlm/scripts/eval_vlmeval.py`. Install separately and set the `VLMEVALKIT_DIR` env var. License: per upstream.

### Inference / training infrastructure
- **PyTorch** — BSD-style. Required runtime dependency.
- **Transformers (HuggingFace)** — Apache-2.0. Required runtime dependency.
- **flash-attn (FlashAttention 2)** — BSD-3-Clause. Required runtime dependency for the LLM single-GPU path.
- **flashinfer** — Apache-2.0. Optional fast-path used by the cascade attention in `models/parallel/recomputer.py`.
- **Triton** — MIT. Transitive runtime dependency.

Specific pinned versions used by the paper are in `llm/requirements-lock.txt`.

## Notes
- This release ships **no model weights** and **no upstream license texts** for external models, benchmarks, or libraries; users obtain those from the upstream sources, which carry their own license obligations.
- The bundled LongBench v1 JSONL subsets in `llm/inputs/` are included for reproducibility convenience. If the LongBench license terms preclude redistribution in your context, delete `llm/inputs/*.jsonl` and re-fetch them from the upstream HuggingFace dataset.
