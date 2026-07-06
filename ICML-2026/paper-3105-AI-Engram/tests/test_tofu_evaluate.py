"""TOFU "Overall" evaluation (very heavy: ~30-45 min, GPU).

Computes the paper's composite Overall score (14 metrics, rescaled against the
finetuned base and the retain90 gold model) for the engram edits produced by the
*package* (``compute_engram_weights``), in the two paper conditions, and compares
to the paper targets:

    gold (retain90) ~ 0.998 | plain (a=0.6) ~ 0.698 | adaptive (a=1.0,p=1) ~ 0.818

Eval pipeline is ported verbatim from examples/llm_tofu.ipynb (see
tests/_tofu_evaluate.py). Gated behind ENGRAM_RUN_TOFU_EVALUATE=1. Run via
SLURM gpu partition (offline cache):
    ENGRAM_RUN_TOFU_EVALUATE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        pytest -s tests/test_tofu_evaluate.py
"""

from __future__ import annotations

import copy
import math
import os

import pytest
import torch
from torch.utils.data import DataLoader

from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm

pytestmark = pytest.mark.skipif(
    os.environ.get("ENGRAM_RUN_TOFU_EVALUATE") != "1",
    reason="very heavy TOFU evaluation; set ENGRAM_RUN_TOFU_EVALUATE=1 (GPU + cache)",
)

BASE_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
RETAIN_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
SPLIT, HOLDOUT = "forget10", "holdout10"
N_TOTAL = 4000
PLAIN_ALPHA = 0.6
ADAPT_ALPHA, ADAPT_P = 1.0, 1
PAPER = {"gold": 0.998, "plain": 0.698, "adaptive": 0.818}
TOL = 0.12  # band for "similar to paper"


# The two paper conditions as pluggable scaling functions:
#   plain    = count_ratio(1.0)                          -> W - alpha*(n/N)*P  (paper)
#   adaptive = compose(count_ratio(1.0), weight_norm(p))  -> further weighted by ||P||/||W||.
# The old "adaptive" scaled the engram that ALREADY carried n/N, so it is the *product*
# of the two factors — hence compose(count_ratio, weight_norm), not weight_norm alone.
PLAIN_SCALE = count_ratio(1.0)


def test_tofu_evaluate_overall():
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    pytest.importorskip("rouge_score")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    import _tofu_evaluate as T
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(mid):
        return (
            AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
            .to(device)
            .eval()
        )

    D = {
        "fp": load_dataset("locuslab/TOFU", f"{SPLIT}_perturbed")["train"],
        "ho": load_dataset("locuslab/TOFU", HOLDOUT)["train"],
        "rp": load_dataset("locuslab/TOFU", "retain_perturbed")["train"],
        "ra": load_dataset("locuslab/TOFU", "real_authors_perturbed")["train"],
        "wf": load_dataset("locuslab/TOFU", "world_facts_perturbed")["train"],
    }
    forget_total = load_dataset("locuslab/TOFU", "full")["train"].shuffle(seed=0).select(range(N_TOTAL))

    # --- engram weights via the package (answer-token-masked covariance) ---
    base = load(BASE_ID)
    editor = EngramEditor(base, EditorConfig(storage_device=torch.device(device)))

    def feats(b):
        return {"input_ids": b["input_ids"].to(device), "attention_mask": b["attention_mask"].to(device)}

    def covdl(ds):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=8, collate_fn=T.Collator(tok, "right"))

    mask_fn = lambda b: b["labels"] != T.IGNORE_INDEX
    g_forget = editor.collect_statistics(covdl(D["fp"]), batch_fn=feats, mask_fn=mask_fn)
    g_total = editor.collect_statistics(covdl(forget_total), batch_fn=feats, mask_fn=mask_fn)
    engram = editor.compute_engram_weights(g_forget, g_total)  # covariances not retained
    del g_forget, g_total
    torch.cuda.empty_cache()

    # --- anchors + conditions (compute_full_tofu is generation-heavy) ---
    ft = T.compute_full_tofu(base, tok, D)

    plain = editor.apply(engram, alpha=PLAIN_ALPHA, scale=PLAIN_SCALE).eval()
    exp_plain = T.compute_full_tofu(plain, tok, D)
    del plain
    torch.cuda.empty_cache()

    adaptive = editor.apply(
        engram, alpha=ADAPT_ALPHA, scale=compose(count_ratio(1.0), weight_norm(ADAPT_P))
    ).eval()
    exp_adapt = T.compute_full_tofu(adaptive, tok, D)
    del adaptive, base, engram
    torch.cuda.empty_cache()

    rt = T.compute_full_tofu(load(RETAIN_ID), tok, D)

    o_gold = T.evaluate_scores(rt, rt, ft)["Overall"]
    o_plain = T.evaluate_scores(exp_plain, rt, ft)["Overall"]
    o_adapt = T.evaluate_scores(exp_adapt, rt, ft)["Overall"]

    print(
        "\n[TOFU forget10 | Overall]   ours      paper     |diff|"
        f"\n  gold (retain90)            : {o_gold:6.3f}   {PAPER['gold']:6.3f}   {abs(o_gold - PAPER['gold']):.3f}"
        f"\n  plain    (a={PLAIN_ALPHA})           : {o_plain:6.3f}   {PAPER['plain']:6.3f}   {abs(o_plain - PAPER['plain']):.3f}"
        f"\n  adaptive (a={ADAPT_ALPHA}, p={ADAPT_P})      : {o_adapt:6.3f}   {PAPER['adaptive']:6.3f}   {abs(o_adapt - PAPER['adaptive']):.3f}"
    )

    assert all(math.isfinite(x) for x in (o_gold, o_plain, o_adapt)), "non-finite Overall"
    assert o_gold > 0.90, f"gold Overall should be ~1 (got {o_gold:.3f})"
    assert o_adapt > o_plain, f"adaptive ({o_adapt:.3f}) should beat plain ({o_plain:.3f}) [paper's key result]"
    assert abs(o_plain - PAPER["plain"]) < TOL, f"plain {o_plain:.3f} not within {TOL} of paper {PAPER['plain']}"
    assert abs(o_adapt - PAPER["adaptive"]) < TOL, f"adaptive {o_adapt:.3f} not within {TOL} of paper {PAPER['adaptive']}"
