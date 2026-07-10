from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.finetune import train_vision as train_vision_module
from merge_and_rebase.finetune.train_vision import (
    ImageEncoder,
    _resolve_text_embeddings_finetune_cfg,
    _resolve_text_prompt_tuning_cfg,
    _run_text_embeddings_finetune_stage,
    _run_text_prompt_tuning_stage,
)
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig


class _TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.eye(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)[:, :2]
        return flat @ self.proj


class _TinyClipInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _TinyVisual()
        self.transformer = nn.Linear(2, 2, bias=False)


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyClipInner()
        self.preprocess = object()
        self.tokenizer = None
        self.normalize = False
        self.logit_scale = 1.0
        # Intentionally swapped classes so the text-embedding stage has to fix it.
        self.register_buffer(
            "_zs_text_features",
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
            persistent=False,
        )
        self._zs_text_fingerprint = "dummy"

    def build_zeroshot_text_features(self, classnames, build_cfg) -> None:
        del classnames, build_cfg
        self._zs_text_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        self._zs_text_fingerprint = "dummy"


class _TinyPromptTokenizer:
    def __init__(self, context_length: int = 8) -> None:
        self.context_length = context_length
        self.sos = 1
        self.eos = 31
        self.ids = {"X": 2, "left": 10, "right": 11}

    def __call__(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        for i, text in enumerate(texts):
            words = text.strip().split()
            out[i, 0] = self.sos
            pos = 1
            for w in words:
                if pos >= self.context_length - 1:
                    break
                out[i, pos] = self.ids.get(w, 5)
                pos += 1
            out[i, pos] = self.eos
        return out


class _TinyTextTransformer(nn.Module):
    def get_cast_dtype(self) -> torch.dtype:
        return torch.float32

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attn_mask
        y = x.clone()
        # Make EOS token representation depend on all tokens so context vectors affect pooled text features.
        y[:, -1, :] = x.sum(dim=1)
        return y


class _TinyPromptClipInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _TinyVisual()
        self.transformer = _TinyTextTransformer()
        self.token_embedding = nn.Embedding(64, 2)
        with torch.no_grad():
            self.token_embedding.weight.zero_()
            self.token_embedding.weight[10] = torch.tensor([1.0, 0.0])  # left
            self.token_embedding.weight[11] = torch.tensor([0.0, 1.0])  # right
            self.token_embedding.weight[31] = torch.tensor([0.2, 0.2])  # eos
        self.positional_embedding = nn.Parameter(torch.zeros(8, 2))
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.eye(2))
        self.text_eos_id = 31
        self.attn_mask = torch.zeros(8, 8)


class _TinyPromptClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyPromptClipInner()
        self.preprocess = object()
        self.tokenizer = _TinyPromptTokenizer()
        self.normalize = False
        self.logit_scale = 1.0
        self.register_buffer("_zs_text_features", torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32), persistent=False)
        self._zs_text_fingerprint = "dummy"

    def build_zeroshot_text_features(self, classnames, build_cfg) -> None:
        del classnames, build_cfg
        self._zs_text_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        self._zs_text_fingerprint = "dummy"


def _build_toy_loaders():
    x = torch.tensor(
        [
            [[[2.0, 0.0]]],
            [[[2.2, 0.0]]],
            [[[0.0, 2.0]]],
            [[[0.0, 2.2]]],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    ds = TensorDataset(x, y)
    return SimpleNamespace(
        train=DataLoader(ds, batch_size=2, shuffle=True),
        val=DataLoader(ds, batch_size=2, shuffle=False),
        test=DataLoader(ds, batch_size=2, shuffle=False),
        classnames=["left", "right"],
    )


def test_resolve_text_embeddings_finetune_cfg_uses_defaults() -> None:
    cfg = _resolve_text_embeddings_finetune_cfg(
        {"text_embeddings_finetune": True},
        default_epochs=1,
        default_lr=1e-4,
        default_weight_decay=0.1,
        default_warmup_length=5,
        default_clip_grad_norm=1.0,
        default_accumulate_grad_batches=2,
    )
    assert cfg is not None
    assert cfg["epochs"] == 1
    assert cfg["lr"] == 1e-4
    assert cfg["weight_decay"] == 0.1
    assert cfg["warmup_length"] == 5
    assert cfg["accumulate_grad_batches"] == 2


def test_resolve_text_embeddings_finetune_cfg_uses_vision_epochs_when_missing() -> None:
    cfg = _resolve_text_embeddings_finetune_cfg(
        {"text_embeddings_finetune": {}},
        default_epochs=7,
        default_lr=1e-4,
        default_weight_decay=0.1,
        default_warmup_length=5,
        default_clip_grad_norm=1.0,
        default_accumulate_grad_batches=2,
    )
    assert cfg is not None
    assert cfg["epochs"] == 7


def test_resolve_text_embeddings_finetune_cfg_uses_vision_epochs_when_null() -> None:
    cfg = _resolve_text_embeddings_finetune_cfg(
        {"text_embeddings_finetune": {"epochs": None}},
        default_epochs=9,
        default_lr=1e-4,
        default_weight_decay=0.1,
        default_warmup_length=5,
        default_clip_grad_norm=1.0,
        default_accumulate_grad_batches=2,
    )
    assert cfg is not None
    assert cfg["epochs"] == 9


def test_text_embedding_stage_updates_matrix_and_improves_acc() -> None:
    torch.manual_seed(0)
    model = ImageEncoder(_TinyClassifier())
    loaders = _build_toy_loaders()
    initial = model.clip_model._zs_text_features.detach().clone()

    summary = _run_text_embeddings_finetune_stage(
        task="toy",
        model=model,
        loaders=loaders,
        device=torch.device("cpu"),
        cfg={
            "epochs": 20,
            "optimizer": "adamw",
            "lr": 0.1,
            "weight_decay": 0.0,
            "warmup_length": 0,
            "clip_grad_norm": 1.0,
            "accumulate_grad_batches": 1,
            "early_stopping": False,
            "early_stopping_patience": 5,
        },
    )

    final = model.clip_model._zs_text_features.detach().clone()
    assert not torch.allclose(initial, final)
    assert model.clip_model._zs_text_fingerprint is None
    assert summary["initial_test_top1"] < 0.5
    assert summary["best_test_top1"] > summary["initial_test_top1"]
    assert summary["best_elapsed_seconds"] >= 0.0
    assert summary["last_elapsed_seconds"] >= summary["best_elapsed_seconds"]


def test_resolve_text_prompt_tuning_cfg_uses_defaults() -> None:
    cfg = _resolve_text_prompt_tuning_cfg(
        {"text_prompt_tuning": True},
        default_lr=1e-4,
        default_weight_decay=0.1,
        default_warmup_length=5,
        default_clip_grad_norm=1.0,
        default_accumulate_grad_batches=2,
    )
    assert cfg is not None
    assert cfg["epochs"] == 1
    assert cfg["lr"] == 1e-4
    assert cfg["weight_decay"] == 0.1
    assert cfg["warmup_length"] == 5
    assert cfg["accumulate_grad_batches"] == 2
    assert cfg["context_length"] == 16


def test_resolve_text_prompt_tuning_cfg_uses_defaults_when_null() -> None:
    cfg = _resolve_text_prompt_tuning_cfg(
        {"text_prompt_tuning": {"epochs": None, "context_length": None}},
        default_epochs=8,
        default_lr=1e-4,
        default_weight_decay=0.1,
        default_warmup_length=5,
        default_clip_grad_norm=1.0,
        default_accumulate_grad_batches=2,
    )
    assert cfg is not None
    assert cfg["epochs"] == 8
    assert cfg["context_length"] == 16


def test_text_prompt_tuning_stage_updates_matrix() -> None:
    torch.manual_seed(0)
    model = ImageEncoder(_TinyPromptClassifier())
    loaders = _build_toy_loaders()
    initial = model.clip_model._zs_text_features.detach().clone()
    before_flags = [bool(p.requires_grad) for p in model.clip_model.model.parameters()]

    summary = _run_text_prompt_tuning_stage(
        task="toy",
        model=model,
        loaders=loaders,
        device=torch.device("cpu"),
        cfg={
            "epochs": 4,
            "optimizer": "adamw",
            "lr": 0.05,
            "weight_decay": 0.0,
            "warmup_length": 0,
            "clip_grad_norm": 1.0,
            "accumulate_grad_batches": 1,
            "early_stopping": False,
            "early_stopping_patience": 5,
            "context_length": 2,
            "ctx_init": None,
            "init_std": 0.02,
        },
    )

    final = model.clip_model._zs_text_features.detach().clone()
    tuned_ctx = getattr(model.clip_model, "_tuned_prompt_context", None)
    after_flags = [bool(p.requires_grad) for p in model.clip_model.model.parameters()]
    assert not torch.allclose(initial, final)
    assert model.clip_model._zs_text_fingerprint is None
    assert after_flags == before_flags
    assert isinstance(tuned_ctx, torch.Tensor)
    assert tuple(tuned_ctx.shape) == (2, 2)
    assert summary["context_length"] == 2
    assert summary["trainable_params"] == 4
    assert summary["best_elapsed_seconds"] >= 0.0
    assert summary["last_elapsed_seconds"] >= summary["best_elapsed_seconds"]


def test_load_model_init_checkpoint_reuses_visual_weights_and_tuned_text(tmp_path) -> None:
    source = ImageEncoder(_TinyClassifier())
    with torch.no_grad():
        source.clip_model.model.visual.proj.copy_(2.0 * torch.eye(2))

    tuned_text = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    ckpt_path = tmp_path / "toy_full.pt"
    torch.save(
        {
            "format": "full",
            "task": "toy",
            "state_dict": source.state_dict(),
            "tuned_text_features": tuned_text,
        },
        ckpt_path,
    )

    target = ImageEncoder(_TinyClassifier())
    with torch.no_grad():
        target.clip_model.model.visual.proj.zero_()

    ckpt_obj, summary = train_vision_module._load_model_init_checkpoint(model=target, ckpt_path=str(ckpt_path))
    selected = train_vision_module._initialize_task_text_features(
        model=target,
        classnames=["left", "right"],
        build_cfg=OpenClipBuildConfig(model_name="m", pretrained="p", device="cpu", dtype="fp32"),
        device=torch.device("cpu"),
        ckpt_obj=ckpt_obj,
        ckpt_path=str(ckpt_path),
        text_features_source="tuned_ckpt",
    )

    assert summary["loaded_tensors"] > 0
    assert summary["load_target"] == "wrapper"
    assert torch.allclose(target.clip_model.model.visual.proj, 2.0 * torch.eye(2))
    assert selected == "tuned_ckpt"
    assert torch.allclose(target.clip_model._zs_text_features, tuned_text)


def test_train_task_text_only_uses_init_checkpoint_and_skips_vision_stage(tmp_path, monkeypatch) -> None:
    init_model = ImageEncoder(_TinyClassifier())
    with torch.no_grad():
        init_model.clip_model.model.visual.proj.copy_(3.0 * torch.eye(2))

    init_ckpt = tmp_path / "init_full.pt"
    torch.save(
        {
            "format": "full",
            "task": "toy",
            "state_dict": init_model.state_dict(),
        },
        init_ckpt,
    )

    loaders = _build_toy_loaders()
    monkeypatch.setattr(train_vision_module, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        train_vision_module.OpenClipClassifier,
        "build",
        staticmethod(lambda cfg: _TinyClassifier()),
    )
    monkeypatch.setattr(train_vision_module, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(
        train_vision_module,
        "get_strategy",
        lambda name: (_ for _ in ()).throw(AssertionError("vision strategy should not be configured in text_only mode")),
    )

    summary = train_vision_module.train_task(
        task="toy",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "validation": "validation", "test": "test"},
        build_cfg=OpenClipBuildConfig(model_name="m", pretrained="p", device="cpu", dtype="fp32"),
        strategy="full",
        epochs=1,
        lr=0.1,
        weight_decay=0.0,
        warmup_length=0,
        clip_grad_norm=1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=5,
        text_only=True,
        seed=0,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        init_checkpoint=str(init_ckpt),
        init_text_features_source="zero_shot",
        strategy_cfg={
            "text_embeddings_finetune": {
                "enabled": True,
                "epochs": 4,
                "optimizer": "adamw",
                "lr": 0.1,
                "weight_decay": 0.0,
                "warmup_length": 0,
                "grad_clip_norm": 1.0,
                "accumulate_grad_batches": 1,
                "early_stopping": False,
                "early_stopping_patience": 5,
            }
        },
    )

    best_state = torch.load(summary["best_ckpt_path"], map_location="cpu")

    assert summary["vision_training_skipped"] is True
    assert summary["trainable"]["mode"] == "text_only"
    assert summary["initialization"]["loaded_tensors"] > 0
    assert summary["text_features_init_source"] == "zero_shot"
    assert torch.allclose(best_state["state_dict"]["clip_model.model.visual.proj"], 3.0 * torch.eye(2))
    assert "tuned_text_features" in best_state


def test_train_task_without_checkpoint_saves_keeps_selected_timing_summary(tmp_path, monkeypatch) -> None:
    loaders = _build_toy_loaders()
    monkeypatch.setattr(train_vision_module, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        train_vision_module.OpenClipClassifier,
        "build",
        staticmethod(lambda cfg: _TinyPromptClassifier()),
    )
    monkeypatch.setattr(train_vision_module, "build_vision_loaders", lambda **kwargs: loaders)

    class _DummyStrategy:
        @staticmethod
        def configure(**kwargs):
            model = kwargs["model"]
            return torch.optim.SGD(model.parameters(), lr=0.1), (lambda step: None), {"trainable_params": 12}

    monkeypatch.setattr(train_vision_module, "get_strategy", lambda name: _DummyStrategy())

    summary = train_vision_module.train_task(
        task="toy",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "validation": "validation", "test": "test"},
        build_cfg=OpenClipBuildConfig(model_name="m", pretrained="p", device="cpu", dtype="fp32"),
        strategy="full",
        epochs=2,
        lr=0.1,
        weight_decay=0.0,
        warmup_length=0,
        clip_grad_norm=1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=True,
        early_stopping_patience=1,
        text_only=False,
        seed=0,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=True,
        strategy_cfg={
            "text_prompt_tuning": {
                "enabled": True,
                "epochs": 2,
                "optimizer": "adamw",
                "lr": 0.05,
                "weight_decay": 0.0,
                "warmup_length": 0,
                "grad_clip_norm": 1.0,
                "accumulate_grad_batches": 1,
                "early_stopping": True,
                "early_stopping_patience": 1,
                "context_length": 2,
                "init_std": 0.02,
            }
        },
    )

    assert summary["save_checkpoints"] is False
    assert summary["best_ckpt_path"] is None
    assert summary["last_ckpt_path"] is None
    assert summary["best_elapsed_seconds"] >= 0.0
    assert summary["last_elapsed_seconds"] >= summary["best_elapsed_seconds"]
    assert summary["text_prompt_tuning"] is not None
    assert summary["selected_timing"]["text_prestage_seconds"] == summary["text_prompt_tuning"]["best_elapsed_seconds"]
    assert summary["selected_timing"]["vision_seconds"] == summary["best_elapsed_seconds"]
    assert summary["selected_timing"]["total_seconds"] == (
        summary["text_prompt_tuning"]["best_elapsed_seconds"] + summary["best_elapsed_seconds"]
    )
