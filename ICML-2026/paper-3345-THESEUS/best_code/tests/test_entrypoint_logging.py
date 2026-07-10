from __future__ import annotations

import importlib.machinery
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn

_datasets_stub = ModuleType("datasets")
_datasets_stub.ClassLabel = object
_datasets_stub.DatasetDict = dict
_datasets_stub.Features = dict
_datasets_stub.Dataset = object
_datasets_stub.load_dataset = lambda *args, **kwargs: {}
_datasets_stub.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
sys.modules.setdefault("datasets", _datasets_stub)


class _DummyTqdm:
    def __init__(self, iterable=None, **kwargs) -> None:
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.iterable or [])

    def update(self, n=1) -> None:
        return None

    def set_postfix(self, value=None, **kwargs) -> None:
        return None


_tqdm_stub = ModuleType("tqdm")
_tqdm_stub.tqdm = _DummyTqdm
_tqdm_stub.__spec__ = importlib.machinery.ModuleSpec("tqdm", loader=None)
sys.modules.setdefault("tqdm", _tqdm_stub)
_attention_stub = ModuleType("torch.nn.attention")
_attention_stub.SDPBackend = object
_attention_stub.sdpa_kernel = lambda *args, **kwargs: nullcontext()
_attention_stub.__spec__ = importlib.machinery.ModuleSpec("torch.nn.attention", loader=None)
sys.modules.setdefault("torch.nn.attention", _attention_stub)

from merge_and_rebase.eval import vision_merge  # noqa: E402
from merge_and_rebase.finetune import train_text, train_vision  # noqa: E402
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig  # noqa: E402
from merge_and_rebase.models.text_lm import TextBuildConfig  # noqa: E402


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.summaries: list[dict[str, object]] = []
        self.finishes: list[str] = []

    def log_event(self, event_type, metrics=None, *, step=None, context=None):
        self.events.append(
            {
                "event_type": event_type,
                "metrics": metrics or {},
                "step": step,
                "context": context or {},
            }
        )

    def log_summary(self, summary_dict):
        self.summaries.append(summary_dict)

    def finish(self, status, error=None):
        self.finishes.append(status)


class _NamedLoader(list):
    def __init__(self, name: str, items):
        super().__init__(items)
        self.name = name


def test_train_text_task_emits_epoch_and_task_events(tmp_path: Path, monkeypatch):
    class _DummyTextModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.score = nn.Linear(4, 2)
            self.config = SimpleNamespace(num_labels=2)

        def forward(self, input_ids, attention_mask=None, labels=None):
            logits = self.score(input_ids.float())
            loss = nn.CrossEntropyLoss()(logits, labels)
            return SimpleNamespace(loss=loss, logits=logits)

    train_batches = _NamedLoader(
        "train",
        [
            {"input_ids": torch.ones(2, 4), "labels": torch.tensor([0, 1])},
            {"input_ids": torch.zeros(2, 4), "labels": torch.tensor([1, 0])},
        ],
    )
    val_batches = _NamedLoader("val", [])
    test_batches = _NamedLoader("test", [])

    monkeypatch.setattr(
        train_text.TextLM,
        "build",
        staticmethod(lambda build_cfg: SimpleNamespace(model=_DummyTextModel(), tokenizer=object())),
    )
    monkeypatch.setattr(
        train_text,
        "_build_task_loaders",
        lambda **kwargs: (
            SimpleNamespace(loader=train_batches),
            SimpleNamespace(loader=val_batches),
            SimpleNamespace(loader=test_batches),
            {"labels": ["n", "y"], "label_texts": ["n", "y"], "head_class_ids": [0, 1]},
        ),
    )
    monkeypatch.setattr(
        train_text,
        "_configure_text_strategy",
        lambda **kwargs: (
            kwargs["model"],
            torch.optim.SGD(kwargs["model"].parameters(), lr=0.1),
            (lambda step: None),
            {"trainable_params": 10},
            {},
        ),
    )
    monkeypatch.setattr(train_text, "_top1", lambda model, loader, device: 0.75 if loader.name == "val" else 0.5)

    logger = _FakeLogger()
    summary, _ = train_text.train_task(
        task="dummy",
        build_cfg=TextBuildConfig(
            model_name_or_path="dummy",
            model_arch="auto",
            device="cpu",
            dtype="fp32",
            model_kind="sequence_classification",
            num_labels=2,
            trust_remote_code=False,
            use_fast_tokenizer=True,
        ),
        strategy="full",
        strategy_cfg={},
        epochs=1,
        lr=0.1,
        weight_decay=0.0,
        warmup_length=0,
        optimizer_name="sgd",
        clip_grad_norm=0.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        max_length=4,
        head_num_labels=2,
        early_stopping=False,
        early_stopping_patience=3,
        seed=0,
        deterministic=False,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        task_cfg={},
        log_every_n_steps=1,
        run_logger=logger,
    )

    assert summary["metrics"]["val_top1"] == 0.75
    assert any(event["event_type"] == "epoch_end" for event in logger.events)
    assert any(event["event_type"] == "task_end" for event in logger.events)


def test_train_vision_task_emits_epoch_and_task_events(tmp_path: Path, monkeypatch):
    class _DummyClipModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Linear(1, 1)
            self.visual = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))

    class _DummyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _DummyClipModel()
            self.preprocess = object()
            self.normalize = False
            self.logit_scale = 1.0
            self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)

        def build_zeroshot_text_features(self, classnames, build_cfg):
            self._zs_text_features = torch.eye(len(classnames), dtype=torch.float32)

    loaders = SimpleNamespace(
        train=_NamedLoader(
            "train",
            [
                (torch.ones(2, 3, 2, 2), torch.tensor([0, 1])),
                (torch.zeros(2, 3, 2, 2), torch.tensor([1, 0])),
            ],
        ),
        val=_NamedLoader("val", []),
        test=_NamedLoader("test", []),
        classnames=["a", "b"],
    )

    class _DummyStrategy:
        @staticmethod
        def configure(**kwargs):
            model = kwargs["model"]
            return torch.optim.SGD(model.parameters(), lr=0.1), (lambda step: None), {"trainable_params": 12}

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _DummyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _DummyStrategy())

    logger = _FakeLogger()
    summary = train_vision.train_task(
        task="dummy",
        hf_path="x",
        hf_config=None,
        split_map={"train": "train", "validation": "validation", "test": "test"},
        build_cfg=OpenClipBuildConfig(model_name="m", pretrained="p", device="cpu", dtype="fp32"),
        strategy="full",
        epochs=1,
        lr=0.1,
        weight_decay=0.0,
        warmup_length=0,
        clip_grad_norm=0.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=3,
        text_only=False,
        seed=0,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        log_every_n_steps=1,
        run_logger=logger,
    )

    assert "metrics" in summary
    assert any(event["event_type"] == "epoch_end" for event in logger.events)
    assert any(event["event_type"] == "task_end" for event in logger.events)


def test_vision_merge_zero_shot_logs_summary(tmp_path: Path, monkeypatch):
    fake_logger = _FakeLogger()

    class _DummyClassifier:
        def __init__(self) -> None:
            self.model = nn.Identity()
            self.preprocess = object()

    suite = SimpleNamespace(tasks=("task_a", "task_b"), resolver=lambda task: ("hf", None, {"test": "test"}))
    monkeypatch.setattr(vision_merge, "SUITES", {"mini": suite})
    monkeypatch.setattr(vision_merge, "start_run", lambda **kwargs: fake_logger)
    monkeypatch.setattr(vision_merge.OpenClipClassifier, "build", staticmethod(lambda cfg: _DummyClassifier()))
    monkeypatch.setattr(vision_merge, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        vision_merge,
        "build_vision_loaders",
        lambda **kwargs: SimpleNamespace(classnames=["a", "b"], train=[], val=[], test=[]),
    )
    monkeypatch.setattr(vision_merge, "get_templates", lambda task: ["template"])
    monkeypatch.setattr(vision_merge, "eval_task_top1", lambda **kwargs: 0.6)
    monkeypatch.setattr(vision_merge, "load_into_model", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(
        vision_merge,
        "get_forward_mode",
        lambda name: SimpleNamespace(bind=lambda **kwargs: None),
    )
    monkeypatch.setattr(vision_merge, "pretty_print_task_accuracies", lambda *args, **kwargs: None)
    monkeypatch.setattr(vision_merge, "print_latex_task_rows", lambda *args, **kwargs: None)

    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vision_merge",
            "--suite",
            "mini",
            "--tasks",
            "all",
            "--zero-shot-only",
            "--single-acc-cache",
            str(tmp_path / "cache.json"),
            "--local-log-dir",
            str(tmp_path),
        ],
    )

    vision_merge.main()

    assert fake_logger.summaries
    assert fake_logger.summaries[-1]["mode"] == "zero_shot_only"


def test_vision_merge_save_helper_writes_raw_state_dict(tmp_path: Path):
    out_path = tmp_path / "merged.pt"
    state = {
        "weight": torch.tensor([1.0, 2.0]),
        "bias": torch.tensor([0.5]),
    }

    saved_path = vision_merge._save_merged_state_dict_if_requested(
        state,
        out_path,
        label="test merged",
    )

    assert saved_path == str(out_path)
    loaded = torch.load(out_path, map_location="cpu")
    assert set(loaded) == {"weight", "bias"}
    assert torch.equal(loaded["weight"], state["weight"])
    assert torch.equal(loaded["bias"], state["bias"])
