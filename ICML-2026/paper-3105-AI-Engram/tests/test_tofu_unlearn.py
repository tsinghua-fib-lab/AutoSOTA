"""TOFU unlearning integration test (heavy: GPU + cached HF model/dataset).

Faithful to ``examples/llm_tofu.ipynb``: answer-token-masked covariance over the
forget set and a 4000-sample total set, the closed-form engram from the package
(``compute_engram_weights`` -> ``apply``) in two paper conditions:

  * plain    = scale=count_ratio(1.0):  W <- W - alpha*(n/N)*P                     (alpha=0.6)
  * adaptive = scale=compose(count_ratio(1.0), weight_norm(p)):  also weighted by
               ||P||/||W|| per layer                                              (alpha=1.0, p=1)

Pass/fail criterion is *selective* unlearning via answer-token NLL: the forget
set's NLL must rise substantially while the retain set is preserved, for both
conditions. (The paper's composite "Overall" score is a separate, heavier eval.)

Gated behind ``ENGRAM_RUN_TOFU=1``. Run via SLURM (offline cache):
    ENGRAM_RUN_TOFU=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 pytest -s tests/test_tofu_unlearn.py
"""

from __future__ import annotations

import copy
import math
import os

import pytest
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm

pytestmark = pytest.mark.skipif(
    os.environ.get("ENGRAM_RUN_TOFU") != "1",
    reason="heavy TOFU integration test; set ENGRAM_RUN_TOFU=1 (needs GPU + cached TOFU model/data)",
)

IGNORE = -100
SYSTEM = "You are a helpful assistant."
DATE = "10 Apr 2025"
BASE_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
PLAIN_ALPHA = 0.6              # paper forget10 "plain"
ADAPT_ALPHA, ADAPT_P = 1.0, 1  # paper forget10 "adaptive power-norm (p=1)"
N_TOTAL = 4000                 # G_total sample count (preserves the alpha calibration)
N_RETAIN_EVAL = 200            # retain subset for the NLL control


# ---- notebook-faithful preprocessing (chat template + answer-only labels) ----
def _preprocess(tok, q, a):
    chat = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    di = {"date_string": DATE}
    chat_ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, return_dict=False, **di)
    prompt_ids = tok.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True, return_dict=False, **di)
    if chat_ids[-1] != tok.eos_token_id:
        chat_ids = chat_ids + [tok.eos_token_id]
    n = len(prompt_ids)
    labels = [IGNORE] * n + chat_ids[n:]  # loss only on the answer tokens
    return {
        "input_ids": torch.tensor(chat_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.ones(len(chat_ids), dtype=torch.long),
    }


class _QAData(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return _preprocess(self.tok, r["question"], r["answer"])


def _make_collate(pad_id):
    def collate(items):
        ids = pad_sequence([it["input_ids"] for it in items], batch_first=True, padding_value=pad_id)
        labels = pad_sequence([it["labels"] for it in items], batch_first=True, padding_value=IGNORE)
        return {"input_ids": ids, "attention_mask": ids.ne(pad_id).long(), "labels": labels}

    return collate


@torch.no_grad()
def _mean_answer_nll(model, rows, tok, device, bs=16):
    """Mean over examples of the per-example average answer-token NLL."""
    dl = DataLoader(_QAData(rows, tok), batch_size=bs, collate_fn=_make_collate(tok.pad_token_id))
    lf = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="none")
    vals = []
    for batch in dl:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        sl = labels[..., 1:].contiguous()
        lg = logits[..., :-1, :].contiguous()
        losses = lf(lg.transpose(-1, -2), sl).sum(-1)
        avg = losses / (labels != IGNORE).sum(-1)
        vals += avg.float().cpu().tolist()
    return float(sum(vals) / len(vals))


def _assert_selective(tag, f0, f1, r0, r1):
    assert math.isfinite(f1) and math.isfinite(r1), f"{tag}: non-finite loss"
    # strong forgetting: the memorized answers become much less likely
    assert f1 - f0 > 0.5, f"{tag}: forget NLL barely rose ({f0:.3f} -> {f1:.3f})"
    # selective: forget degrades much more than retain
    assert (f1 - f0) > 1.5 * (r1 - r0), f"{tag}: not selective (forget Δ{f1 - f0:+.3f} vs retain Δ{r1 - r0:+.3f})"
    # retain stays usable (bounded degradation)
    assert r1 < r0 + 1.0, f"{tag}: retain degraded too much ({r0:.3f} -> {r1:.3f})"


def test_tofu_forget10_plain_and_adaptive():
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = (
        AutoModelForCausalLM.from_pretrained(
            BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )

    forget = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"]
    retain = load_dataset("locuslab/TOFU", "retain_perturbed")["train"]
    total = load_dataset("locuslab/TOFU", "full")["train"].shuffle(seed=0).select(range(N_TOTAL))

    # --- package API: answer-token-masked covariance, then engram weights ---
    editor = EngramEditor(base, EditorConfig(storage_device=torch.device(device)))

    def covdl(ds):
        return DataLoader(_QAData(ds, tok), batch_size=8, collate_fn=_make_collate(tok.pad_token_id))

    def feats(batch):
        return {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }

    mask_fn = lambda b: b["labels"] != IGNORE
    g_forget = editor.collect_statistics(covdl(forget), batch_fn=feats, mask_fn=mask_fn)
    g_total = editor.collect_statistics(covdl(total), batch_fn=feats, mask_fn=mask_fn)
    engram = editor.compute_engram_weights(g_forget, g_total)  # covariances not retained
    assert len(engram.layers) > 0
    del g_forget, g_total
    torch.cuda.empty_cache()

    retain_eval = retain.select(range(min(N_RETAIN_EVAL, len(retain))))
    f0 = _mean_answer_nll(base, forget, tok, device)
    r0 = _mean_answer_nll(base, retain_eval, tok, device)

    # --- plain = count_ratio(1.0): W - alpha*(n/N)*P ---
    edited_p = editor.apply(engram, alpha=PLAIN_ALPHA, scale=count_ratio(1.0)).eval()
    fp = _mean_answer_nll(edited_p, forget, tok, device)
    rp = _mean_answer_nll(edited_p, retain_eval, tok, device)
    del edited_p
    torch.cuda.empty_cache()

    # --- adaptive = count_ratio(1.0) further weighted by ||P||/||W|| (compose) ---
    edited_a = editor.apply(
        engram, alpha=ADAPT_ALPHA, scale=compose(count_ratio(1.0), weight_norm(ADAPT_P))
    ).eval()
    fa = _mean_answer_nll(edited_a, forget, tok, device)
    ra = _mean_answer_nll(edited_a, retain_eval, tok, device)
    del edited_a
    torch.cuda.empty_cache()

    print(
        "\n[TOFU forget10 | answer-token NLL]"
        f"\n  base                : forget {f0:.3f} | retain {r0:.3f}"
        f"\n  plain   (a={PLAIN_ALPHA})    : forget {fp:.3f} (Δ{fp - f0:+.3f}) | retain {rp:.3f} (Δ{rp - r0:+.3f})"
        f"\n  adaptive(a={ADAPT_ALPHA},p={ADAPT_P}): forget {fa:.3f} (Δ{fa - f0:+.3f}) | retain {ra:.3f} (Δ{ra - r0:+.3f})"
    )

    _assert_selective("plain", f0, fp, r0, rp)
    _assert_selective("adaptive", f0, fa, r0, ra)

    # paper's key finding: adaptive-norm yields at least as good a forget/retain
    # tradeoff as plain (here it forgets more AND preserves retain better)
    net_plain = (fp - f0) - (rp - r0)
    net_adapt = (fa - f0) - (ra - r0)
    assert net_adapt >= net_plain - 0.10, (
        f"adaptive tradeoff ({net_adapt:+.3f}) should be >= plain ({net_plain:+.3f})"
    )
