"""edit_llm plumbing tests (CPU, offline, no training).

Verifies the one-call helper drives the pipeline correctly — tokenization, masking,
loaders, collect -> edit — by checking it edits weights, respects copy/inplace, handles
both ``str`` and ``(prompt, answer)`` items, and matches the manual EngramEditor pipeline.
A fake tokenizer + a tiny GPT-2 (built from config) keep it download- and training-free.
The *unlearning effect* on a real model is shown in the docs demo, not asserted here.
"""
from __future__ import annotations

import pytest
import torch

from engram import EditorConfig, EngramEditor, apply_engram, edit_llm, get_engram, uniform


class _FakeTok:
    """Deterministic char-level tokenizer: no downloads, ids in [2, 41] (< vocab 64)."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [(ord(c) % 40) + 2 for c in text if not c.isspace()] or [2]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _tiny_gpt2():
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=64, vocab_size=64)
    ).eval()


def _cpu_cfg():
    return EditorConfig(storage_device=torch.device("cpu"))


def test_edit_llm_edits_a_copy_and_leaves_original():
    tok, model = _FakeTok(), _tiny_gpt2()
    W0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    forget = ["hello world", "foo bar baz"]
    total = forget + ["lorem ipsum dolor"]

    edited = edit_llm(model, tok, forget, total, alpha=0.5, scale=uniform(), config=_cpu_cfg())

    assert edited is not model
    for n, p in model.named_parameters():  # original untouched
        assert torch.equal(p, W0[n]), n
    ep = dict(edited.named_parameters())
    assert any(not torch.equal(W0[n], ep[n]) for n in W0), "edit_llm changed nothing"


def test_edit_llm_inplace():
    tok, model = _FakeTok(), _tiny_gpt2()
    W0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    same = edit_llm(
        model, tok, ["alpha beta"], ["alpha beta", "gamma delta"],
        alpha=1.0, scale=uniform(), inplace=True, config=_cpu_cfg(),
    )
    assert same is model
    ep = dict(model.named_parameters())
    assert any(not torch.equal(W0[n], ep[n]) for n in W0)


def test_edit_llm_prompt_answer_items_run():
    tok, model = _FakeTok(), _tiny_gpt2()
    forget = [("question one", "answer one")]
    total = forget + [("q two", "ans two")]
    edited = edit_llm(model, tok, forget, total, alpha=1.0, scale=uniform(), config=_cpu_cfg())
    assert edited is not model  # answer-only masking path ran end-to-end


def test_edit_llm_matches_manual_pipeline():
    from engram.llm import _loader

    tok, model = _FakeTok(), _tiny_gpt2()
    forget, total = ["aa bb", "cc dd ee"], ["aa bb", "cc dd ee", "ff gg"]
    cfg = _cpu_cfg()

    ed = EngramEditor(model, cfg)
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
    mf = lambda b: b["labels"] != -100
    t = ed.collect_statistics(_loader(tok, forget, 512, 8, tok.pad_token_id), batch_fn=feats, mask_fn=mf)
    a = ed.collect_statistics(_loader(tok, total, 512, 8, tok.pad_token_id), batch_fn=feats, mask_fn=mf)
    manual = ed.edit(t, a, alpha=0.7, scale=uniform())

    direct = edit_llm(model, tok, forget, total, alpha=0.7, scale=uniform(), config=cfg)

    md, dd = dict(manual.named_parameters()), dict(direct.named_parameters())
    for n in md:
        assert torch.allclose(md[n], dd[n], atol=1e-5), n


def test_get_engram_then_apply_engram_matches_edit_llm():
    # the split (get_engram once + apply_engram) reproduces the one-call edit_llm
    tok, model = _FakeTok(), _tiny_gpt2()
    forget, total = ["aa bb", "cc dd ee"], ["aa bb", "cc dd ee", "ff gg"]
    cfg = _cpu_cfg()

    engram = get_engram(model, tok, forget, total, config=cfg)
    split = apply_engram(model, engram, alpha=0.7, scale=uniform())
    direct = edit_llm(model, tok, forget, total, alpha=0.7, scale=uniform(), config=cfg)

    sd, dd = dict(split.named_parameters()), dict(direct.named_parameters())
    for n in sd:
        assert torch.allclose(sd[n], dd[n], atol=1e-5), n


def test_apply_engram_reuses_engram_and_alpha_zero_is_noop():
    # compute once, apply at several alphas (no recollection); alpha=0 is a no-op
    tok, model = _FakeTok(), _tiny_gpt2()
    W0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    forget, total = ["aa bb", "cc dd ee"], ["aa bb", "cc dd ee", "ff gg"]

    engram = get_engram(model, tok, forget, total, config=_cpu_cfg())

    e0 = apply_engram(model, engram, alpha=0.0, scale=uniform())   # alpha=0 -> unchanged
    for n, p in e0.named_parameters():
        assert torch.allclose(p, W0[n], atol=1e-6), n

    small = dict(apply_engram(model, engram, alpha=0.3, scale=uniform()).named_parameters())
    big = dict(apply_engram(model, engram, alpha=0.9, scale=uniform()).named_parameters())
    changed = [n for n in W0 if not torch.allclose(W0[n], big[n], atol=1e-6)]
    assert changed, "apply_engram changed nothing"
    n0 = changed[0]                                                # bigger alpha -> larger deviation
    assert (big[n0] - W0[n0]).norm() > (small[n0] - W0[n0]).norm()
    for n, p in model.named_parameters():                          # original model untouched
        assert torch.equal(p, W0[n]), n
