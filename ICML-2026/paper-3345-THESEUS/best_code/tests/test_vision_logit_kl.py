from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from merge_and_rebase.eval import vision_logit_kl


def test_batch_kl_sum_matches_torch_reference() -> None:
    ref_logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]])
    merged_logits = torch.tensor([[1.0, 0.25, -0.25], [0.0, 2.0, -1.0]])
    temperature = 1.3

    ref_log_probs = F.log_softmax(ref_logits / temperature, dim=-1)
    got = vision_logit_kl._batch_kl_sum(
        ref_log_probs=ref_log_probs,
        merged_logits=merged_logits,
        temperature=temperature,
    )
    expected = F.kl_div(
        F.log_softmax(merged_logits / temperature, dim=-1),
        ref_log_probs,
        reduction="sum",
        log_target=True,
    )

    assert torch.allclose(got, expected)


def test_batch_kl_per_sample_sums_to_batch_kl() -> None:
    ref_logits = torch.tensor([[3.0, 1.0], [0.0, 2.0], [1.0, 1.0]])
    merged_logits = torch.tensor([[2.0, 0.0], [0.25, 1.25], [1.5, 0.5]])
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    per_sample = vision_logit_kl._batch_kl_per_sample(
        ref_log_probs=ref_log_probs,
        merged_logits=merged_logits,
        temperature=1.0,
    )

    assert per_sample.shape == (3,)
    assert torch.all(per_sample >= -1e-7)
    assert torch.allclose(
        per_sample.sum(),
        vision_logit_kl._batch_kl_sum(ref_log_probs=ref_log_probs, merged_logits=merged_logits, temperature=1.0),
    )


def test_resolve_kl_candidate_prefers_cli_over_summary_and_config(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"best_alpha": 0.7, "best_method_params": {"topk": 0.2}}))

    candidate = vision_logit_kl.resolve_kl_candidate(
        cfg={"alpha": 0.3, "method_params": {"topk": 1.0}},
        cli_alpha=0.9,
        cli_method_params={"topk": 0.5},
        merge_summary_path=str(summary_path),
    )

    assert candidate.alpha == 0.9
    assert candidate.method_params == {"topk": 0.5}
    assert "cli_alpha" in candidate.source
    assert "cli_method_params" in candidate.source


def test_resolve_kl_candidate_uses_summary_when_cli_absent(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"best_alpha": 0.4, "best_method_params": {"svd_dtype": "fp32"}}))

    candidate = vision_logit_kl.resolve_kl_candidate(
        cfg={"alpha": 0.3, "method_params": {"svd_dtype": "fp64"}},
        merge_summary_path=str(summary_path),
    )

    assert candidate.alpha == 0.4
    assert candidate.method_params == {"svd_dtype": "fp32"}
    assert "merge_summary_alpha" in candidate.source


def test_resolve_kl_candidate_uses_config_alpha_and_errors_on_unselected_search() -> None:
    candidate = vision_logit_kl.resolve_kl_candidate(
        cfg={"alpha": 0.25, "method_params": {"keep_ratio": 1.0}, "alpha_search": True},
    )

    assert candidate.alpha == 0.25
    assert candidate.method_params == {"keep_ratio": 1.0}

    with pytest.raises(ValueError, match="selected merged candidate"):
        vision_logit_kl.resolve_kl_candidate(cfg={"alpha_search": True, "method_params": {}})


def test_resolve_kl_candidate_allows_unselected_search_with_direct_checkpoint() -> None:
    candidate = vision_logit_kl.resolve_kl_candidate(
        cfg={"alpha_search": True, "method_params": {"topk": 0.2}},
        merged_ckpt_path="merged.pt",
    )

    assert candidate.alpha != candidate.alpha
    assert candidate.method_params == {"topk": 0.2}
    assert "merged_ckpt" in candidate.source


def test_run_vision_logit_kl_with_dummy_classifier_limits_batches(tmp_path: Path, monkeypatch) -> None:
    class _DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(2))

    class _DummyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _DummyModel()
            self.preprocess = object()
            self.normalize = False
            self.logit_scale = 1.0
            self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)
            self._zs_text_fingerprint = None

        def build_zeroshot_text_features(self, classnames, build_cfg, cache_dir=None, force_rebuild=False):
            del build_cfg, cache_dir, force_rebuild
            self._zs_text_features = torch.eye(len(classnames), dtype=torch.float32)

        def resolve_eval_text_features(
            self,
            *,
            text_features_source,
            classnames,
            build_cfg,
            tuned_text_features,
            cache_dir,
            force_rebuild_zeroshot,
            task_name,
            ckpt_path,
            verbose,
        ):
            del (
                text_features_source,
                classnames,
                build_cfg,
                tuned_text_features,
                cache_dir,
                force_rebuild_zeroshot,
                task_name,
                ckpt_path,
                verbose,
            )
            return None, "zero_shot"

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images + self.model.bias

    ckpt_path = tmp_path / "task_a.pt"
    torch.save({"state_dict": {"bias": torch.tensor([1.0, 0.0])}}, ckpt_path)

    batches = [
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[0.5, 0.5], [2.0, -1.0]]), torch.tensor([0, 0])),
    ]
    suite = SimpleNamespace(tasks=("task_a",), resolver=lambda task: ("hf", None, {"test": "test", "val": "val"}))
    loads: list[tuple[float, float]] = []

    def _recording_load_into_model(model, sd, *, strict=False):
        del strict
        loads.append(tuple(float(v) for v in sd["bias"].tolist()))
        model.load_state_dict(dict(sd), strict=False)
        return 0, 0

    monkeypatch.setattr(vision_logit_kl, "SUITES", {"mini": suite})
    monkeypatch.setattr(vision_logit_kl.OpenClipClassifier, "build", staticmethod(lambda cfg: _DummyClassifier()))
    monkeypatch.setattr(vision_logit_kl, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        vision_logit_kl,
        "build_vision_loaders",
        lambda **kwargs: SimpleNamespace(classnames=["a", "b"], test=batches, val=batches),
    )
    monkeypatch.setattr(vision_logit_kl, "get_templates", lambda task: ["a photo of {}"])
    monkeypatch.setattr(vision_logit_kl, "load_into_model", _recording_load_into_model)

    cfg = {
        "suite": "mini",
        "tasks": "all",
        "clip_model": "dummy",
        "clip_pretrained": "dummy",
        "device": "cpu",
        "dtype": "fp32",
        "method": "task_arithmetic",
        "alpha": 0.5,
        "method_params": {},
        "tuned_ckpts": {"task_a": str(ckpt_path)},
        "batch_size": 2,
        "num_workers": 0,
        "max_batches_per_task": 1,
        "save_plots": False,
        "logit_cache_dir": str(tmp_path / "logit_cache"),
    }
    candidate = vision_logit_kl.resolve_kl_candidate(cfg=cfg)
    output = tmp_path / "kl.json"

    result = vision_logit_kl.run_vision_logit_kl(cfg, candidate=candidate, output_path=output)

    assert output.exists()
    payload = json.loads(output.read_text())
    assert result["per_task"]["task_a"]["samples"] == 2
    assert payload["total_samples"] == 2
    assert payload["avg_kl"] > 0.0
    assert loads == [(1.0, 0.0), (0.5, 0.0)]
    assert payload["per_task"]["task_a"]["text_features_mode"] == "zero_shot"
    assert payload["plots"]["enabled"] is False
    assert payload["logit_cache"]["per_task"]["task_a"]["status"] == "miss"

    loads.clear()
    second = vision_logit_kl.run_vision_logit_kl(cfg, candidate=candidate, output_path=output)
    payload2 = json.loads(output.read_text())

    assert second["logit_cache"]["per_task"]["task_a"]["status"] == "hit"
    assert payload2["logit_cache"]["per_task"]["task_a"]["status"] == "hit"
    assert loads == [(0.5, 0.0)]


def test_run_vision_logit_kl_loads_direct_merged_checkpoint(tmp_path: Path, monkeypatch) -> None:
    class _DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(2))

    class _DummyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _DummyModel()
            self.preprocess = object()
            self.normalize = False
            self.logit_scale = 1.0
            self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)
            self._zs_text_fingerprint = None

        def build_zeroshot_text_features(self, classnames, build_cfg, cache_dir=None, force_rebuild=False):
            del build_cfg, cache_dir, force_rebuild
            self._zs_text_features = torch.eye(len(classnames), dtype=torch.float32)

        def resolve_eval_text_features(
            self,
            *,
            text_features_source,
            classnames,
            build_cfg,
            tuned_text_features,
            cache_dir,
            force_rebuild_zeroshot,
            task_name,
            ckpt_path,
            verbose,
        ):
            del (
                text_features_source,
                classnames,
                build_cfg,
                tuned_text_features,
                cache_dir,
                force_rebuild_zeroshot,
                task_name,
                ckpt_path,
                verbose,
            )
            return None, "zero_shot"

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images + self.model.bias

    tuned_ckpt = tmp_path / "task_a.pt"
    merged_ckpt = tmp_path / "merged.pt"
    torch.save({"state_dict": {"bias": torch.tensor([1.0, 0.0])}}, tuned_ckpt)
    torch.save({"bias": torch.tensor([-0.5, 0.25])}, merged_ckpt)

    batches = [(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))]
    suite = SimpleNamespace(tasks=("task_a",), resolver=lambda task: ("hf", None, {"test": "test", "val": "val"}))
    loads: list[tuple[float, float]] = []

    def _recording_load_into_model(model, sd, *, strict=False):
        del strict
        loads.append(tuple(float(v) for v in sd["bias"].tolist()))
        model.load_state_dict(dict(sd), strict=False)
        return 0, 0

    def _fail_if_merge_called(**kwargs):
        del kwargs
        raise AssertionError("merge reconstruction should be skipped when merged_ckpt is provided")

    monkeypatch.setattr(vision_logit_kl, "SUITES", {"mini": suite})
    monkeypatch.setattr(vision_logit_kl.OpenClipClassifier, "build", staticmethod(lambda cfg: _DummyClassifier()))
    monkeypatch.setattr(vision_logit_kl, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        vision_logit_kl,
        "build_vision_loaders",
        lambda **kwargs: SimpleNamespace(classnames=["a", "b"], test=batches, val=batches),
    )
    monkeypatch.setattr(vision_logit_kl, "get_templates", lambda task: ["a photo of {}"])
    monkeypatch.setattr(vision_logit_kl, "load_into_model", _recording_load_into_model)
    monkeypatch.setattr(vision_logit_kl, "build_merged_state_for_alpha", _fail_if_merge_called)

    cfg = {
        "suite": "mini",
        "tasks": "all",
        "clip_model": "dummy",
        "clip_pretrained": "dummy",
        "device": "cpu",
        "dtype": "fp32",
        "method": "task_arithmetic",
        "method_params": {},
        "alpha_search": True,
        "merged_ckpt": str(merged_ckpt),
        "tuned_ckpts": {"task_a": str(tuned_ckpt)},
        "batch_size": 2,
        "num_workers": 0,
        "save_plots": False,
        "logit_cache": False,
    }
    candidate = vision_logit_kl.resolve_kl_candidate(cfg=cfg, merged_ckpt_path=str(merged_ckpt))
    output = tmp_path / "kl_direct.json"

    payload = vision_logit_kl.run_vision_logit_kl(cfg, candidate=candidate, output_path=output)

    assert payload["alpha"] is None
    assert payload["merged_checkpoint"] == str(merged_ckpt)
    assert payload["per_task"]["task_a"]["samples"] == 2
    assert loads == [(1.0, 0.0), (-0.5, 0.25)]
