from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call

from merge_and_rebase.finetune import train_vision
from merge_and_rebase.finetune.reference_tasks import build_reference_task_resolution_context
from merge_and_rebase.finetune._vision_runtime import ImageEncoder, run_scaled_image_encoder, snapshot_parameter_map
from merge_and_rebase.finetune.regularizers._distill_config import merge_build_cfg
from merge_and_rebase.finetune.regularizers.base import BatchOverride, OptimizerBundle
from merge_and_rebase.finetune.regularizers._distill_runtime import apply_prepared_distillation, compute_distillation_loss
from merge_and_rebase.finetune.regularizers.distillation import DistillationRegularizer
from merge_and_rebase.finetune.regularizers.registry import list_regularizers
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig


class _ToyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class _ToyClipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(1, 1)
        self.visual = _ToyVisual()


class _ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyClipModel()
        self.preprocess = object()
        self.normalize = True
        self.logit_scale = 1.0
        self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)

    def build_zeroshot_text_features(self, classnames, build_cfg):
        del build_cfg
        self._zs_text_features = torch.eye(len(classnames), dtype=torch.float32)


class _FakeLoraLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 2, scale: float = 1.0) -> None:
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_features, bias=False)})
        self.scaling = {"default": float(scale)}
        nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        a = self.lora_A["default"].weight
        b = self.lora_B["default"].weight
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * float(self.scaling["default"])
        return base + delta


class _FakeLoraVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = _FakeLoraLinear(12, 2, rank=2, scale=1.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.adapter(images.reshape(images.shape[0], -1))


class _FakeLoraClipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(1, 1)
        self.visual = _FakeLoraVisual()


class _FakeLoraClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeLoraClipModel()
        self.preprocess = object()
        self.normalize = True
        self.logit_scale = 1.0
        self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)

    def build_zeroshot_text_features(self, classnames, build_cfg):
        del build_cfg
        self._zs_text_features = torch.eye(len(classnames), dtype=torch.float32)


class _DummyStrategy:
    def configure(self, *, model, lr, weight_decay, warmup_length, optimizer="adamw", steps=1, device=None, **kwargs):
        del weight_decay, warmup_length, optimizer, steps, device, kwargs
        params = [p for p in model.parameters() if p.requires_grad]
        opt = optim.Adam(params, lr=lr)
        return opt, (lambda step: None), {"trainable_params": sum(p.numel() for p in params)}


def _build_model() -> ImageEncoder:
    classifier = _ToyClassifier()
    classifier.build_zeroshot_text_features(["a", "b"], None)
    return ImageEncoder(classifier)


def _build_fake_lora_model() -> ImageEncoder:
    classifier = _FakeLoraClassifier()
    classifier.build_zeroshot_text_features(["a", "b"], None)
    return ImageEncoder(classifier)


def _build_cfg() -> OpenClipBuildConfig:
    return OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")


def _build_loaders():
    batch = (torch.randn(2, 3, 2, 2), torch.tensor([0, 1]))
    return SimpleNamespace(
        train=[batch],
        val=[batch],
        test=[batch],
        classnames=["a", "b"],
    )


def test_regularizer_registry_exposes_distillation() -> None:
    assert "distillation" in list_regularizers()


def test_merge_build_cfg_preserves_loader_when_teacher_build_omits_it() -> None:
    base = OpenClipBuildConfig(loader="openai_clip", model_name="ViT-B-32", pretrained="openai", device="cpu")
    merged = merge_build_cfg(base, {"model_name": "ViT-B-32"})
    assert merged.loader == "openai_clip"


def test_train_task_passes_batch_context_and_closes_prepared_regularizer(monkeypatch, tmp_path) -> None:
    prepared_param = nn.Parameter(torch.zeros(()))
    extra_optimizer = optim.SGD([prepared_param], lr=0.1)

    class _Prepared:
        def __init__(self) -> None:
            self.optimizer_bundles = (OptimizerBundle("aux", extra_optimizer, lambda step: None, -1.0),)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _Regularizer:
        def __init__(self) -> None:
            self.prepared = _Prepared()
            self.apply_calls = []

        def prepare(self, **kwargs):
            self.prepare_kwargs = dict(kwargs)
            return self.prepared, {"distillation_locations": 1}

        def apply(self, prepared, **kwargs):
            self.apply_calls.append({"prepared": prepared, **kwargs})
            return next(kwargs["model"].parameters()).sum() * 0.0

    regularizer = _Regularizer()
    loaders = _build_loaders()

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _ToyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _DummyStrategy())
    monkeypatch.setattr(train_vision, "get_regularizer", lambda name: regularizer)

    summary = train_vision.train_task(
        task="Cars",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "test": "test"},
        build_cfg=_build_cfg(),
        strategy="full",
        epochs=1,
        lr=1e-3,
        optimizer_name="adamw",
        weight_decay=0.0,
        warmup_length=1,
        clip_grad_norm=-1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=1,
        text_only=False,
        seed=42,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=False,
        strategy_cfg={},
        regularization_cfg={"name": "distillation"},
        all_tasks=["Cars", "DTD"],
        reference_tasks=["DTD"],
    )

    assert summary["regularization"]["name"] == "distillation"
    assert regularizer.prepared.closed is True
    assert regularizer.apply_calls
    call = regularizer.apply_calls[0]
    assert isinstance(call["inputs"], torch.Tensor)
    assert isinstance(call["targets"], torch.Tensor)
    assert isinstance(call["outputs"], torch.Tensor)


def test_train_task_uses_batch_override_outputs_and_context(monkeypatch, tmp_path) -> None:
    class _Prepared:
        def __init__(self) -> None:
            self.prepare_batch_calls = 0
            self.closed = False

        def prepare_batch(self, **kwargs):
            self.prepare_batch_calls += 1
            model = kwargs["model"]
            inputs = kwargs["inputs"]
            targets = kwargs["targets"]
            logits = model(inputs)
            primary_loss = torch.nn.functional.cross_entropy(logits, targets) * 0.0 + logits.sum() * 0.0 + 2.0
            return BatchOverride(outputs=logits + 1.0, primary_loss=primary_loss, context={"alpha": 0.25})

        def close(self) -> None:
            self.closed = True

    class _Regularizer:
        def __init__(self) -> None:
            self.prepared = _Prepared()
            self.contexts = []

        def prepare(self, **kwargs):
            return self.prepared, {"distillation_locations": 1}

        def apply(self, prepared, **kwargs):
            self.contexts.append(kwargs.get("batch_context"))
            assert prepared is self.prepared
            assert isinstance(kwargs["outputs"], torch.Tensor)
            return kwargs["outputs"].sum() * 0.0

    regularizer = _Regularizer()
    loaders = _build_loaders()

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _ToyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _DummyStrategy())
    monkeypatch.setattr(train_vision, "get_regularizer", lambda name: regularizer)

    summary = train_vision.train_task(
        task="Cars",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "test": "test"},
        build_cfg=_build_cfg(),
        strategy="full",
        epochs=1,
        lr=1e-3,
        optimizer_name="adamw",
        weight_decay=0.0,
        warmup_length=1,
        clip_grad_norm=-1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=1,
        text_only=False,
        seed=42,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=False,
        strategy_cfg={},
        regularization_cfg={"name": "distillation"},
        all_tasks=["Cars", "DTD"],
        reference_tasks=["DTD"],
    )

    assert summary["regularization"]["name"] == "distillation"
    assert regularizer.prepared.prepare_batch_calls == 1
    assert regularizer.prepared.closed is True
    assert regularizer.contexts == [{"alpha": 0.25}]


def test_train_task_finalizes_model_before_strategy_configure(monkeypatch, tmp_path) -> None:
    class _Prepared:
        optimizer_bundles = ()

    class _Regularizer:
        def finalize_model(self, *, model, **kwargs):
            model._finalized_for_regularizer = True
            return {"patched_blocks": 1}

        def prepare(self, **kwargs):
            assert getattr(kwargs["model"], "_finalized_for_regularizer", False) is True
            return _Prepared(), {}

        def apply(self, prepared, **kwargs):
            del prepared
            return kwargs["outputs"].sum() * 0.0

    class _CheckingStrategy(_DummyStrategy):
        def configure(self, **kwargs):
            assert getattr(kwargs["model"], "_finalized_for_regularizer", False) is True
            return super().configure(**kwargs)

    loaders = _build_loaders()

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _ToyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _CheckingStrategy())
    monkeypatch.setattr(train_vision, "get_regularizer", lambda name: _Regularizer())

    summary = train_vision.train_task(
        task="Cars",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "test": "test"},
        build_cfg=_build_cfg(),
        strategy="full",
        epochs=1,
        lr=1e-3,
        optimizer_name="adamw",
        weight_decay=0.0,
        warmup_length=1,
        clip_grad_norm=-1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=1,
        text_only=False,
        seed=42,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=False,
        strategy_cfg={},
        regularization_cfg={"name": "distillation"},
        all_tasks=["Cars"],
        reference_tasks=["DTD"],
    )

    assert summary["regularization"]["name"] == "distillation"
    assert summary["regularization"]["info"]["finalize_model.patched_blocks"] == 1


def test_frozen_teacher_distillation_loss_is_positive_and_teacher_has_no_grads(monkeypatch) -> None:
    torch.manual_seed(0)
    student = _build_model()
    regularizer = DistillationRegularizer()
    build_cfg = _build_cfg()
    loaders = _build_loaders()

    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )

    prepared, info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "teacher": {"mode": "frozen"},
        },
        task="Cars",
        build_cfg=build_cfg,
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=1,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    assert info["distillation_locations"] == 1
    inputs, targets = loaders.train[0]
    outputs = student(inputs)
    loss = regularizer.apply(
        prepared,
        model=student,
        step=0,
        batch_index=0,
        inputs=inputs,
        targets=targets,
        outputs=outputs,
    )
    assert float(loss.item()) > 0.0
    loss.backward()
    assert any(p.grad is not None for p in student.parameters() if p.requires_grad)
    assert all(p.grad is None for p in prepared.teacher.model.parameters())


def test_mse_matches_old_repo_reduction() -> None:
    student = torch.tensor([[1.0, 3.0], [0.0, 2.0]])
    teacher = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    loss = compute_distillation_loss(student, teacher, {"name": "mse"})
    expected = ((student - teacher) ** 2).sum(dim=1).mean()
    assert torch.allclose(loss, expected)


def test_along_path_reuses_alpha_within_virtual_batch_and_resamples_next_step(monkeypatch) -> None:
    student = _build_model()
    regularizer = DistillationRegularizer()
    loaders = _build_loaders()

    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    samples = iter([0.2, 0.7])
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.random.uniform",
        lambda low, high: next(samples),
    )

    prepared, info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "along_path": {"enabled": True, "alpha_range": [0.0, 1.0], "sampling": "uniform"},
            "teacher": {"mode": "frozen"},
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=10,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    inputs, targets = loaders.train[0]
    first = regularizer.prepare_batch(
        prepared,
        model=student,
        step=0,
        batch_index=0,
        inputs=inputs,
        targets=targets,
        virtual_batch_start=True,
    )
    second = regularizer.prepare_batch(
        prepared,
        model=student,
        step=0,
        batch_index=1,
        inputs=inputs,
        targets=targets,
        virtual_batch_start=False,
    )
    third = regularizer.prepare_batch(
        prepared,
        model=student,
        step=1,
        batch_index=0,
        inputs=inputs,
        targets=targets,
        virtual_batch_start=True,
    )

    assert isinstance(first, BatchOverride)
    assert isinstance(second, BatchOverride)
    assert isinstance(third, BatchOverride)
    assert first.context["alpha"] == second.context["alpha"] == 0.2
    assert third.context["alpha"] == 0.7
    assert info["distillation_along_path_last_alpha"] == 0.7


def test_scaled_forward_matches_manual_base_plus_alpha_delta() -> None:
    torch.manual_seed(0)
    student = _build_model()
    base = snapshot_parameter_map(student)
    inputs, _targets = _build_loaders().train[0]

    with torch.no_grad():
        student.clip_model.model.visual.net[1].weight.add_(0.4)
        student.clip_model.model.visual.net[1].bias.add_(0.1)

    alpha = 0.25
    logits = run_scaled_image_encoder(model=student, images=inputs, alpha=alpha, base_params=base)

    current = dict(student.named_parameters())
    manual_params = {}
    for name, value in current.items():
        base_value = base[name]
        manual_params[name] = base_value + alpha * (value - base_value)
    visual_params = {
        key[len("clip_model.model.visual.") :]: value
        for key, value in manual_params.items()
        if key.startswith("clip_model.model.visual.")
    }
    visual_buffers = dict(student.clip_model.model.visual.named_buffers())
    visual_features = functional_call(
        student.clip_model.model.visual,
        (visual_params, visual_buffers),
        args=(inputs,),
        strict=False,
    )
    image_features = visual_features / (visual_features.norm(dim=-1, keepdim=True) + 1e-12)
    expected = student.clip_model.logit_scale * (image_features @ student.clip_model._zs_text_features.t())

    assert torch.allclose(logits, expected, atol=1e-6, rtol=1e-6)


def test_scaled_forward_scales_fake_lora_in_effective_weight_space() -> None:
    torch.manual_seed(0)
    student = _build_fake_lora_model()
    base = snapshot_parameter_map(student)
    inputs, _targets = _build_loaders().train[0]

    with torch.no_grad():
        visual = student.clip_model.model.visual
        visual.adapter.lora_A["default"].weight.add_(0.3)
        visual.adapter.lora_B["default"].weight.add_(0.2)

    alpha = 0.25
    logits = run_scaled_image_encoder(model=student, images=inputs, alpha=alpha, base_params=base)

    current_weight = (
        student.clip_model.model.visual.adapter.base_layer.weight
        + student.clip_model.model.visual.adapter.lora_B["default"].weight
        @ student.clip_model.model.visual.adapter.lora_A["default"].weight
    )
    base_weight = (
        base["clip_model.model.visual.adapter.base_layer.weight"]
        + base["clip_model.model.visual.adapter.lora_B.default.weight"]
        @ base["clip_model.model.visual.adapter.lora_A.default.weight"]
    )
    expected_weight = base_weight + alpha * (current_weight - base_weight)
    visual_features = torch.nn.functional.linear(inputs.reshape(inputs.shape[0], -1), expected_weight)
    image_features = visual_features / (visual_features.norm(dim=-1, keepdim=True) + 1e-12)
    expected = student.clip_model.logit_scale * (image_features @ student.clip_model._zs_text_features.t())

    assert torch.allclose(logits, expected, atol=1e-6, rtol=1e-6)


def test_frozen_teacher_same_architecture_uses_student_base_for_along_path(monkeypatch) -> None:
    student = _build_model()
    regularizer = DistillationRegularizer()
    loaders = _build_loaders()
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )

    prepared, info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "along_path": {"enabled": True, "alpha_range": [0.5, 1.0]},
            "teacher": {"mode": "frozen"},
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=4,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    assert info["distillation_along_path_teacher_enabled"] == 1
    assert prepared.teacher.along_path_enabled is True
    assert prepared.teacher.along_path_base is not None
    for key, value in prepared.student_base.items():
        assert torch.allclose(prepared.teacher.along_path_base[key], value)


def test_frozen_teacher_incompatible_architecture_falls_back_from_along_path(monkeypatch) -> None:
    class _DifferentToyVisual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(12, 5), nn.ReLU(), nn.Linear(5, 2))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.net(images)

    class _DifferentToyClipModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Linear(1, 1)
            self.visual = _DifferentToyVisual()

    class _DifferentToyClassifier(_ToyClassifier):
        def __init__(self) -> None:
            super().__init__()
            self.model = _DifferentToyClipModel()

    def _build_different_model() -> ImageEncoder:
        classifier = _DifferentToyClassifier()
        classifier.build_zeroshot_text_features(["a", "b"], None)
        return ImageEncoder(classifier)

    class _RunLogger:
        def __init__(self) -> None:
            self.events = []

        def log_event(self, name, metrics, context):
            self.events.append((name, metrics, context))

    run_logger = _RunLogger()
    student = _build_model()
    regularizer = DistillationRegularizer()
    loaders = _build_loaders()
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_different_model().to(device=device),
    )

    prepared, info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "along_path": {"enabled": True, "alpha_range": [0.0, 1.0]},
            "teacher": {"mode": "frozen"},
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=4,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
        run_logger=run_logger,
    )

    assert info["distillation_along_path_teacher_enabled"] == 0
    assert info["distillation_along_path_teacher_fallback"] == 1
    assert prepared.teacher.along_path_enabled is False
    assert run_logger.events and run_logger.events[0][0] == "distillation_along_path_warning"


def test_online_teacher_stop_gradient_controls_teacher_grads(monkeypatch) -> None:
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _DummyStrategy(),
    )
    loaders = _build_loaders()
    build_cfg = _build_cfg()
    regularizer = DistillationRegularizer()

    def _run(stop_gradient: bool):
        student = _build_model()
        prepared, _info = regularizer.prepare(
            model=student,
            device=torch.device("cpu"),
            regularization_cfg={
                "name": "distillation",
                "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
                "teacher": {
                    "mode": "online",
                    "stop_gradient": stop_gradient,
                    "strategy": {"name": "full"},
                    "train": {"lr": 1e-3},
                    "supervised": {"enabled": False},
                },
            },
            task="Cars",
            build_cfg=build_cfg,
            loaders=loaders,
            batch_size=2,
            num_workers=0,
            val_fraction=0.1,
            seed=42,
            total_steps=1,
            warmup_length=1,
            train_lr=1e-3,
            train_weight_decay=0.0,
            train_optimizer_name="adamw",
            train_grad_clip_norm=-1.0,
        )
        inputs, targets = loaders.train[0]
        outputs = student(inputs)
        loss = regularizer.apply(
            prepared,
            model=student,
            step=0,
            batch_index=0,
            inputs=inputs,
            targets=targets,
            outputs=outputs,
        )
        loss.backward()
        return SimpleNamespace(student=student, prepared=prepared, loss=loss)

    stopped = _run(True)
    flowing = _run(False)

    assert all(p.grad is None for p in stopped.prepared.teacher.model.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in flowing.prepared.teacher.model.parameters())


def test_online_teacher_peft_strategy_receives_peft_cfg(monkeypatch) -> None:
    student = _build_model()
    regularizer = DistillationRegularizer()
    loaders = _build_loaders()
    captured = {}

    class _CaptureStrategy:
        def configure(self, *, model, lr, weight_decay, warmup_length, optimizer="adamw", steps=1, device=None, **kwargs):
            del lr, weight_decay, warmup_length, optimizer, steps, device
            captured["peft_cfg"] = kwargs.get("peft_cfg")
            params = [p for p in model.parameters() if p.requires_grad]
            opt = optim.Adam(params, lr=1e-3)
            return opt, (lambda step: None), {"trainable_params": sum(p.numel() for p in params)}

    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _CaptureStrategy(),
    )

    regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "teacher": {
                "mode": "online",
                "strategy": {
                    "name": "peft_lora",
                    "peft": {
                        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
                        "r": 16,
                        "lora_alpha": 16,
                        "lora_dropout": 0.0,
                        "bias": "none",
                    },
                },
            },
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=4,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    assert captured["peft_cfg"] == {
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
    }


def test_online_teacher_finalizes_nested_regularizer_before_strategy_configure(monkeypatch) -> None:
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )

    class _NestedPrepared:
        optimizer_bundles = ()

    class _NestedRegularizer:
        def finalize_model(self, *, model, **kwargs):
            model._teacher_regularizer_finalized = True
            return {"patched_blocks": 2}

        def prepare(self, **kwargs):
            assert getattr(kwargs["model"], "_teacher_regularizer_finalized", False) is True
            return _NestedPrepared(), {}

        def apply(self, prepared, **kwargs):
            del prepared, kwargs
            return torch.zeros(())

    class _CheckingStrategy(_DummyStrategy):
        def configure(self, **kwargs):
            assert getattr(kwargs["model"], "_teacher_regularizer_finalized", False) is True
            return super().configure(**kwargs)

    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _CheckingStrategy(),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers._distill_teacher.get_regularizer",
        lambda name: _NestedRegularizer(),
    )

    student = _build_model()
    loaders = _build_loaders()
    regularizer = DistillationRegularizer()

    prepared, info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "teacher": {
                "mode": "online",
                "strategy": {"name": "full"},
                "train": {"lr": 1e-3},
                "supervised": {"enabled": False},
                "regularization": {"name": "capture_nested"},
            },
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=1,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    assert prepared.teacher.regularizer_prepared is not None
    assert info["distillation_teacher_online"] == 1


def test_module_hook_location_and_projection_are_supported(monkeypatch) -> None:
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    regularizer = DistillationRegularizer()
    student = _build_model()
    loaders = _build_loaders()

    prepared, info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "shared_weight": 0.5,
            "locations": [
                {
                    "student": {
                        "source": "module",
                        "path": "clip_model.model.visual.net.1",
                        "capture": "output",
                        "projection": {"kind": "linear", "out_features": 3},
                    },
                    "teacher": {
                        "source": "module",
                        "path": "clip_model.model.visual.net.1",
                        "capture": "output",
                        "projection": {"kind": "linear", "out_features": 3},
                    },
                    "loss": "mse",
                }
            ],
            "teacher": {"mode": "frozen"},
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=1,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    assert info["distillation_adapter_params"] > 0
    assert len(prepared.optimizer_bundles) == 1
    inputs, targets = loaders.train[0]
    outputs = student(inputs)
    loss = regularizer.apply(
        prepared,
        model=student,
        step=0,
        batch_index=0,
        inputs=inputs,
        targets=targets,
        outputs=outputs,
    )
    assert float(loss.item()) >= 0.0


def test_recursive_teacher_regularization_and_checkpoint_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _DummyStrategy(),
    )
    student = _build_model()
    loaders = _build_loaders()
    regularizer = DistillationRegularizer()

    prepared, info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "teacher": {
                "mode": "online",
                "save_checkpoint": True,
                "output_dir": "src/checkpoints/test_teacher",
                "strategy": {"name": "full"},
                "train": {"lr": 1e-3},
                "supervised": {"enabled": False},
                "regularization": {
                    "name": "distillation",
                    "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
                    "teacher": {"mode": "frozen"},
                },
            },
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=1,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    assert info["distillation_teacher_online"] == 1
    assert prepared.teacher.regularizer_prepared is not None
    assert prepared.checkpoint_payload(kind="best_ep") == {}
    artifacts = prepared.checkpoint_artifacts(
        kind="best_ep",
        epoch_i=1,
        val_acc_i=0.5,
        test_acc_i=0.6,
        zero_shot_metrics={"val_top1": 0.1, "test_top1": 0.2},
    )
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.filename == "full__distillation_best_ep.pt"
    assert artifact.output_dir == "src/checkpoints/test_teacher/ViT-B-32/openai/Cars"
    assert artifact.payload["task"] == "Cars"
    assert artifact.payload["forward_mode"] == "standard"
    assert "state_dict" in artifact.payload


def test_nested_teacher_regularization_receives_reference_resolution_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _DummyStrategy(),
    )

    class _NestedPrepared:
        optimizer_bundles = ()

    class _NestedRegularizer:
        def __init__(self) -> None:
            self.prepare_kwargs = None

        def prepare(self, **kwargs):
            self.prepare_kwargs = dict(kwargs)
            return _NestedPrepared(), {}

        def apply(self, prepared, **kwargs):
            del prepared, kwargs
            return torch.zeros(())

    nested_regularizer = _NestedRegularizer()
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers._distill_teacher.get_regularizer",
        lambda name: nested_regularizer,
    )

    student = _build_model()
    loaders = _build_loaders()
    regularizer = DistillationRegularizer()
    context = build_reference_task_resolution_context(training_tasks=["Cars", "DTD"], suite="vision8")

    prepared, info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "teacher": {
                "mode": "online",
                "strategy": {"name": "full"},
                "train": {"lr": 1e-3},
                "supervised": {"enabled": False},
                "regularization": {
                    "name": "capture_nested",
                    "reference_datasets": ["ImageNet1K"],
                },
            },
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        reference_tasks=list(train_vision.SUITES["vision8"].tasks),
        reference_resolution_context=context,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=1,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    assert info["distillation_teacher_online"] == 1
    assert prepared.teacher.regularizer_prepared is not None
    assert nested_regularizer.prepare_kwargs is not None
    assert nested_regularizer.prepare_kwargs["reference_resolution_context"] == context


def test_distillation_apply_records_breakdown_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _DummyStrategy(),
    )

    student = _build_model()
    loaders = _build_loaders()
    regularizer = DistillationRegularizer()
    prepared, _info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"name": "logits", "student": "logits", "teacher": "logits", "loss": "mse"}],
            "along_path": {"enabled": True, "alpha_range": [0.5, 1.5]},
            "teacher": {"mode": "frozen"},
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=4,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    inputs, targets = next(iter(loaders.train))
    batch_override = regularizer.prepare_batch(
        prepared,
        model=student,
        step=0,
        batch_index=0,
        inputs=inputs,
        targets=targets,
        virtual_batch_start=True,
    )
    assert batch_override is not None

    loss = apply_prepared_distillation(
        prepared,
        model=student,
        step=0,
        batch_index=0,
        inputs=inputs,
        targets=targets,
        outputs=batch_override.outputs,
        batch_context=batch_override.context,
    )
    breakdown = getattr(student, "_distillation_last_breakdown", None)

    assert isinstance(breakdown, dict)
    assert "loss_distill" in breakdown
    assert "sampled_alpha" in breakdown
    assert any(str(key).startswith("loss_distill_0_") for key in breakdown.keys())
    assert torch.isclose(loss.detach().cpu(), torch.tensor(float(breakdown["loss_distill"])), atol=1e-6, rtol=1e-6)


def test_online_teacher_breakdown_includes_teacher_losses(monkeypatch) -> None:
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _DummyStrategy(),
    )

    student = _build_model()
    loaders = _build_loaders()
    regularizer = DistillationRegularizer()
    prepared, _info = regularizer.prepare(
        model=student,
        device=torch.device("cpu"),
        regularization_cfg={
            "name": "distillation",
            "locations": [{"name": "logits", "student": "logits", "teacher": "logits", "loss": "mse"}],
            "teacher": {
                "mode": "online",
                "strategy": {"name": "full"},
                "train": {"lr": 1e-3},
                "supervised": {"enabled": True, "weight": 1.0, "loss": {"name": "cross_entropy"}},
            },
        },
        task="Cars",
        build_cfg=_build_cfg(),
        loaders=loaders,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
        total_steps=4,
        warmup_length=1,
        train_lr=1e-3,
        train_weight_decay=0.0,
        train_optimizer_name="adamw",
        train_grad_clip_norm=-1.0,
    )

    inputs, targets = next(iter(loaders.train))
    loss = apply_prepared_distillation(
        prepared,
        model=student,
        step=0,
        batch_index=0,
        inputs=inputs,
        targets=targets,
        outputs=student(inputs),
        batch_context=None,
    )
    breakdown = getattr(student, "_distillation_last_breakdown", None)

    assert isinstance(breakdown, dict)
    assert "loss_teacher_task" in breakdown
    assert "loss_teacher_supervised" in breakdown
    assert "loss_teacher_total" in breakdown
    assert float(breakdown["loss_teacher_total"]) >= float(breakdown["loss_teacher_task"]) > 0.0
    assert float(loss.detach().cpu()) >= float(breakdown["loss_teacher_total"])


def test_train_task_logs_separate_loss_components(monkeypatch, tmp_path) -> None:
    class _Prepared:
        optimizer_bundles = ()

        def close(self) -> None:
            return None

    class _Regularizer:
        def prepare(self, **kwargs):
            return _Prepared(), {}

        def apply(self, prepared, **kwargs):
            del prepared
            model = kwargs["model"]
            model._distillation_last_breakdown = {  # type: ignore[attr-defined]
                "loss_distill": 1.5,
                "loss_teacher_task": 0.25,
                "loss_teacher_supervised": 0.25,
                "loss_reg_teacher": 0.75,
                "loss_teacher_total": 1.0,
                "sampled_alpha": 0.8,
            }
            model._ekfac_ggn_last_breakdown = {  # type: ignore[attr-defined]
                "matrix": 0.1,
                "ffT": 0.2,
                "projection": 0.3,
                "class_embedding": 0.4,
            }
            return kwargs["outputs"].sum() * 0.0 + 1.5

    class _RunLogger:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def log_event(self, event_type, metrics=None, *, step=None, context=None):
            self.events.append(
                {
                    "event_type": event_type,
                    "metrics": metrics or {},
                    "step": step,
                    "context": context or {},
                }
            )

    regularizer = _Regularizer()
    run_logger = _RunLogger()
    loaders = _build_loaders()

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _ToyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _DummyStrategy())
    monkeypatch.setattr(train_vision, "get_regularizer", lambda name: regularizer)

    train_vision.train_task(
        task="Cars",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "test": "test"},
        build_cfg=_build_cfg(),
        strategy="full",
        epochs=1,
        lr=1e-3,
        optimizer_name="adamw",
        weight_decay=0.0,
        warmup_length=1,
        clip_grad_norm=-1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=1,
        text_only=False,
        seed=42,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=False,
        strategy_cfg={},
        regularization_cfg={"name": "distillation"},
        all_tasks=["Cars", "DTD"],
        reference_tasks=["DTD"],
        log_every_n_steps=1,
        run_logger=run_logger,
    )

    train_step_events = [event for event in run_logger.events if event["event_type"] == "train_step"]
    assert train_step_events
    metrics = train_step_events[0]["metrics"]

    assert f"train/Cars/loss_task" in metrics
    assert f"train/Cars/loss_reg" in metrics
    assert f"train/Cars/loss_total_step" in metrics
    assert f"train/Cars/loss_distill" in metrics
    assert f"train/Cars/loss_penalty" in metrics
    assert f"train/Cars/loss_reg_ffT" in metrics
    assert f"train/Cars/loss_ft_proj" in metrics
    assert f"train/Cars/loss_reg_cls_emb" in metrics
    assert f"train/Cars/sampled_alpha" in metrics

    teacher_events = [event for event in run_logger.events if event["event_type"] == "train_step_teacher"]
    assert teacher_events
    teacher_metrics = teacher_events[0]["metrics"]
    assert f"train/Cars/loss_teacher_task" in teacher_metrics
    assert f"train/Cars/loss_teacher_supervised" in teacher_metrics
    assert f"train/Cars/loss_reg_teacher" in teacher_metrics
    assert f"train/Cars/loss_teacher_total" in teacher_metrics


def test_train_task_logs_teacher_accuracy_at_task_end(monkeypatch, tmp_path) -> None:
    run_logger = type(
        "_RunLogger",
        (),
        {
            "__init__": lambda self: setattr(self, "events", []),
            "log_event": lambda self, event_type, metrics=None, *, step=None, context=None: self.events.append(
                {
                    "event_type": event_type,
                    "metrics": metrics or {},
                    "step": step,
                    "context": context or {},
                }
            ),
        },
    )()

    loaders = _build_loaders()

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _ToyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _DummyStrategy())
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _DummyStrategy(),
    )

    summary = train_vision.train_task(
        task="Cars",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "test": "test"},
        build_cfg=_build_cfg(),
        strategy="full",
        epochs=1,
        lr=1e-3,
        optimizer_name="adamw",
        weight_decay=0.0,
        warmup_length=1,
        clip_grad_norm=-1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=1,
        text_only=False,
        seed=42,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=False,
        strategy_cfg={},
        regularization_cfg={
            "name": "distillation",
            "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
            "teacher": {"mode": "frozen"},
        },
        all_tasks=["Cars", "DTD"],
        reference_tasks=["DTD"],
        log_every_n_steps=1,
        run_logger=run_logger,
    )

    assert "teacher_metrics" in summary
    assert "test_top1" in summary["teacher_metrics"]
    task_end_events = [event for event in run_logger.events if event["event_type"] == "task_end"]
    assert task_end_events
    metrics = task_end_events[0]["metrics"]
    assert f"val/Cars/top1_teacher" in metrics
    assert f"test/Cars/top1_teacher" in metrics
    assert metrics[f"val/Cars/top1_teacher"] == metrics[f"val/Cars/top1_teacher"]
    assert metrics[f"test/Cars/top1_teacher"] == metrics[f"test/Cars/top1_teacher"]


def test_train_task_logs_teacher_accuracy_at_task_end_for_composite(monkeypatch, tmp_path) -> None:
    run_logger = type(
        "_RunLogger",
        (),
        {
            "__init__": lambda self: setattr(self, "events", []),
            "log_event": lambda self, event_type, metrics=None, *, step=None, context=None: self.events.append(
                {
                    "event_type": event_type,
                    "metrics": metrics or {},
                    "step": step,
                    "context": context or {},
                }
            ),
        },
    )()

    loaders = _build_loaders()

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _ToyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _DummyStrategy())
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.build_image_encoder",
        lambda build_cfg, device=None: _build_model().to(device=device),
    )
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.distillation.get_strategy",
        lambda name: _DummyStrategy(),
    )

    summary = train_vision.train_task(
        task="Cars",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "test": "test"},
        build_cfg=_build_cfg(),
        strategy="full",
        epochs=1,
        lr=1e-3,
        optimizer_name="adamw",
        weight_decay=0.0,
        warmup_length=1,
        clip_grad_norm=-1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=1,
        text_only=False,
        seed=42,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=False,
        strategy_cfg={},
        regularization_cfg={
            "name": "composite",
            "regularizers": [
                {
                    "name": "distillation",
                    "locations": [{"student": "logits", "teacher": "logits", "loss": "kl_div"}],
                    "teacher": {"mode": "frozen"},
                }
            ],
        },
        all_tasks=["Cars", "DTD"],
        reference_tasks=["DTD"],
        log_every_n_steps=1,
        run_logger=run_logger,
    )

    assert "teacher_metrics" in summary
    assert "test_top1" in summary["teacher_metrics"]
    task_end_events = [event for event in run_logger.events if event["event_type"] == "task_end"]
    assert task_end_events
    metrics = task_end_events[0]["metrics"]
    assert metrics[f"val/Cars/top1_teacher"] == metrics[f"val/Cars/top1_teacher"]
    assert metrics[f"test/Cars/top1_teacher"] == metrics[f"test/Cars/top1_teacher"]


def test_train_task_logs_backward_losses_independently_from_log_every(monkeypatch, tmp_path) -> None:
    class _Prepared:
        optimizer_bundles = ()

        def close(self) -> None:
            return None

    class _Regularizer:
        def prepare(self, **kwargs):
            return _Prepared(), {}

        def apply(self, prepared, **kwargs):
            del prepared
            model = kwargs["model"]
            model._distillation_last_breakdown = {  # type: ignore[attr-defined]
                "loss_distill": 1.5,
                "loss_teacher_task": 0.25,
                "loss_teacher_supervised": 0.25,
                "loss_reg_teacher": 0.75,
                "loss_teacher_total": 1.0,
                "sampled_alpha": 0.8,
            }
            return kwargs["outputs"].sum() * 0.0 + 1.5

    class _RunLogger:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def log_event(self, event_type, metrics=None, *, step=None, context=None):
            self.events.append(
                {
                    "event_type": event_type,
                    "metrics": metrics or {},
                    "step": step,
                    "context": context or {},
                }
            )

    regularizer = _Regularizer()
    run_logger = _RunLogger()
    loaders = _build_loaders()

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _ToyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _DummyStrategy())
    monkeypatch.setattr(train_vision, "get_regularizer", lambda name: regularizer)

    train_vision.train_task(
        task="Cars",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "test": "test"},
        build_cfg=_build_cfg(),
        strategy="full",
        epochs=1,
        lr=1e-3,
        optimizer_name="adamw",
        weight_decay=0.0,
        warmup_length=1,
        clip_grad_norm=-1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=1,
        text_only=False,
        seed=42,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=False,
        strategy_cfg={},
        regularization_cfg={"name": "distillation"},
        all_tasks=["Cars", "DTD"],
        reference_tasks=["DTD"],
        log_every_n_steps=999,
        run_logger=run_logger,
    )

    backward_events = [event for event in run_logger.events if event["event_type"] == "train_backward_loss"]
    assert backward_events
    backward_metrics = backward_events[0]["metrics"]
    assert f"train_backward/Cars/loss_task" in backward_metrics
    assert f"train_backward/Cars/loss_reg" in backward_metrics
    assert f"train_backward/Cars/loss_total" in backward_metrics
    assert f"train_backward/Cars/loss_distill" in backward_metrics
    assert f"train_backward/Cars/loss_teacher_task" in backward_metrics
    assert f"train_backward/Cars/loss_teacher_supervised" in backward_metrics
    assert f"train_backward/Cars/loss_reg_teacher" in backward_metrics
    assert f"train_backward/Cars/loss_teacher_total" in backward_metrics

    train_step_events = [event for event in run_logger.events if event["event_type"] == "train_step"]
    assert not train_step_events
