from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.func import functional_call

from merge_and_rebase.finetune import train_text, train_vision
from merge_and_rebase.finetune.forward_mode import apply_training_forward_mode, resolve_training_forward_mode
from merge_and_rebase.finetune.strategies.registry import get_strategy, list_strategies
from merge_and_rebase.models.forward_modes import (
    bind_training_forward_mode,
    get_forward_mode,
    resolve_auto_forward_mode,
)
from merge_and_rebase.models.openclip_classifier import (
    OpenClipBuildConfig,
    normalize_features,
    zero_shot_logits_from_features,
)
from merge_and_rebase.models.text_lm import TextBuildConfig
from merge_and_rebase.utils.linearization import LinearizedModule
from merge_and_rebase.utils.peft_materialization import materialized_peft_param_map


class _KwargLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.b = nn.Parameter(torch.tensor([[0.5, 0.5], [0.5, -0.5]]))

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return scale * (x @ self.a.t() + x @ self.b.t())


class _SurfaceAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip_model = nn.Module()
        self.clip_model.model = nn.Module()
        self.clip_model.model.visual = nn.Module()
        visual = self.clip_model.model.visual
        visual.class_embedding = nn.Parameter(torch.randn(4))
        visual.proj = nn.Parameter(torch.randn(4, 2))
        visual.conv1 = nn.Linear(4, 4, bias=False)
        visual.ln_pre = nn.LayerNorm(4)
        visual.transformer = nn.Module()
        visual.transformer.resblocks = nn.ModuleList([nn.Module()])
        block = visual.transformer.resblocks[0]
        block.ln_1 = nn.LayerNorm(4)
        block.attn = nn.MultiheadAttention(4, num_heads=1, batch_first=False)
        block.ln_2 = nn.LayerNorm(4)
        block.mlp = nn.Module()
        block.mlp.c_fc = nn.Linear(4, 8)
        block.mlp.c_proj = nn.Linear(8, 4)
        visual.positional_embedding = nn.Parameter(torch.randn(2, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(4, 2)
        self.config = SimpleNamespace(num_labels=2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        logits = self.score(input_ids.float())
        return SimpleNamespace(loss=None, logits=logits)


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
        self.normalize = True
        self.logit_scale = 1.0
        self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)

    def build_zeroshot_text_features(self, classnames, build_cfg):
        del build_cfg
        self._zs_text_features = torch.eye(len(classnames), dtype=torch.float32)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.visual(images)
        return zero_shot_logits_from_features(self, image_features, normalize_image_features=self.normalize)


class _FeatureCachingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip_model = nn.Module()
        self.clip_model._last_visual_features = None
        self.clip_model._last_image_features = None
        self.clip_model._last_logits = None
        self.proj = nn.Linear(3, 2)
        self._last_visual_features = None
        self._last_image_features = None
        self._last_logits = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        visual_features = x + 1.0
        image_features = visual_features * 2.0
        logits = self.proj(image_features)
        self._last_visual_features = visual_features
        self._last_image_features = image_features
        self._last_logits = logits
        self.clip_model._last_visual_features = visual_features
        self.clip_model._last_image_features = image_features
        self.clip_model._last_logits = logits
        return logits


def _base_model_state(clf: _DummyClassifier) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in clf.model.state_dict().items()}


def _manual_linearized_raw_visual_features(
    clf: _DummyClassifier,
    *,
    base_sd: dict[str, torch.Tensor],
    images: torch.Tensor,
) -> torch.Tensor:
    ref_visual = _DummyClipModel().visual
    ref_visual.load_state_dict(
        {key[len("visual.") :]: value for key, value in base_sd.items() if key.startswith("visual.")},
        strict=True,
    )
    linearized = LinearizedModule.from_module(ref_visual, copy_module=False)
    return linearized.forward(
        current_module=clf.model.visual,
        args=(images,),
    )


def _manual_linearized_logits(
    clf: _DummyClassifier,
    *,
    base_sd: dict[str, torch.Tensor],
    images: torch.Tensor,
    linearized_feature_normalization: bool,
    linearized_logit_normalization: bool = True,
) -> torch.Tensor:
    ref_visual = _DummyClipModel().visual
    ref_visual.load_state_dict(
        {key[len("visual.") :]: value for key, value in base_sd.items() if key.startswith("visual.")},
        strict=True,
    )
    linearized = LinearizedModule.from_module(ref_visual, copy_module=False)
    features = linearized.forward(
        current_module=clf.model.visual,
        args=(images,),
        output_transform=normalize_features,
        post_transform=normalize_features if linearized_feature_normalization else None,
    )
    return zero_shot_logits_from_features(
        clf,
        features,
        normalize_image_features=linearized_logit_normalization,
    )


def test_linearized_module_supports_kwargs_and_param_filter() -> None:
    ref = _KwargLinear()
    current = _KwargLinear()
    linearized = LinearizedModule.from_module(current, copy_module=True, param_names=["a"])

    with torch.no_grad():
        current.a.add_(1.0)
        current.b.add_(10.0)

    x = torch.tensor([[1.0, 2.0]])
    out = linearized.forward(current_module=current, args=(x,), kwargs={"scale": 2.0})
    expected = 2.0 * (x @ current.a.t() + x @ ref.b.t())
    assert torch.allclose(out, expected)


def test_linearized_module_supports_current_params_and_post_transform() -> None:
    ref = _KwargLinear()
    current = _KwargLinear()
    linearized = LinearizedModule.from_module(ref, copy_module=True, param_names=["a"])

    with torch.no_grad():
        current.a.add_(1.0)
        current.b.add_(10.0)

    x = torch.tensor([[1.0, 2.0]])
    out = linearized.forward(
        current_params=dict(current.named_parameters()),
        args=(x,),
        post_transform=lambda y: y + 1.0,
    )
    expected = x @ current.a.t() + x @ ref.b.t() + 1.0
    assert torch.allclose(out, expected)


def test_apply_training_forward_mode_linearizes_only_trainable_text_params() -> None:
    model = _TinyTextModel()
    model.score.bias.requires_grad = False
    weight_before = model.score.weight.detach().clone()
    bias_before = model.score.bias.detach().clone()

    info = apply_training_forward_mode(
        model=model,
        forward_mode="linearized_ntk",
        device=torch.device("cpu"),
        output_transform=lambda out: out.logits,
        output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
    )

    with torch.no_grad():
        model.score.weight.add_(2.0)
        model.score.bias.add_(5.0)

    batch = torch.ones(2, 4)
    out = model(input_ids=batch)
    expected = batch @ (weight_before + 2.0).t() + bias_before

    assert out.loss is None
    assert torch.allclose(out.logits, expected)
    assert info["linearized_params"] == 1


def test_forward_mode_helpers_enforce_new_contract() -> None:
    assert "ntk" not in list_strategies()
    assert resolve_auto_forward_mode(["linearized_ntk", "linearized_ntk"]) == "linearized_ntk"
    assert resolve_auto_forward_mode(["linearized_ntk", None]) == "standard"

    try:
        resolve_training_forward_mode({"forward_mode": "auto"})
    except ValueError as exc:
        assert "strategy.forward_mode" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected invalid training forward_mode to raise")


def test_full_delta_parameterization_still_exposes_materialization_hooks() -> None:
    model = _SurfaceAwareModel()
    strategy = get_strategy("full")
    opt, _scheduler, info = strategy.configure(
        model=model,
        lr=1e-3,
        weight_decay=0.0,
        warmup_length=0,
        optimizer="adamw",
        steps=1,
        device=torch.device("cpu"),
        strategy_cfg={"params": {"parameterization": "delta", "trainable_params": "regularized_only"}},
    )

    assert info["parameterization"] == "delta"
    current_param_map = getattr(model, "_current_param_map", None)
    materialized_state_dict = getattr(model, "_materialized_state_dict", None)
    assert callable(current_param_map)
    assert callable(materialized_state_dict)

    param_map = current_param_map()
    assert "clip_model.model.visual.class_embedding" in param_map
    assert "clip_model.model.visual.positional_embedding" in param_map
    delta_module = getattr(model, "_delta_module", None)
    assert delta_module is not None
    assert "clip_model.model.visual.positional_embedding" not in delta_module.names

    first_delta = next(iter(opt.param_groups[0]["params"]))
    with torch.no_grad():
        first_delta.add_(0.5)

    state = materialized_state_dict()
    assert "_delta_module.names" not in state
    assert "_delta_module.params.0" not in state


def test_full_delta_parameterization_keeps_runtime_feature_cache_in_sync() -> None:
    model = _FeatureCachingModel()
    strategy = get_strategy("full")
    strategy.configure(
        model=model,
        lr=1e-3,
        weight_decay=0.0,
        warmup_length=0,
        optimizer="adamw",
        steps=1,
        device=torch.device("cpu"),
        strategy_cfg={"params": {"parameterization": "delta", "trainable_params": "all_trainable"}},
    )

    x = torch.randn(5, 3)
    logits = model(x)

    assert isinstance(model._last_visual_features, torch.Tensor)
    assert isinstance(model._last_image_features, torch.Tensor)
    assert isinstance(model._last_logits, torch.Tensor)
    assert tuple(model._last_visual_features.shape) == (5, 3)
    assert tuple(model._last_image_features.shape) == (5, 3)
    assert torch.allclose(model._last_logits, logits)
    assert torch.allclose(model.clip_model._last_visual_features, model._last_visual_features)
    assert torch.allclose(model.clip_model._last_image_features, model._last_image_features)
    assert torch.allclose(model.clip_model._last_logits, logits)


def test_bind_training_forward_mode_supports_full_delta_backed_vision_models() -> None:
    torch.manual_seed(0)
    clf = _DummyClassifier()
    clf.normalize = False
    clf.build_zeroshot_text_features(["a", "b"], build_cfg=None)
    model = train_vision.ImageEncoder(clf)
    strategy = get_strategy("full")
    strategy.configure(
        model=model,
        lr=1e-3,
        weight_decay=0.0,
        warmup_length=0,
        optimizer="adamw",
        steps=1,
        device=torch.device("cpu"),
        strategy_cfg={"params": {"parameterization": "delta", "trainable_params": "all_trainable"}},
    )
    base_sd = _base_model_state(clf)

    info = bind_training_forward_mode(
        model=model,
        forward_mode="linearized_ntk",
        base_sd=base_sd,
        strict_load=True,
        params={"linearized_feature_normalization": False, "linearized_logit_normalization": False},
    )

    assert info["linearized_params"] > 0
    linearized = getattr(model, "_linearized_module")
    assert any(name.endswith("1.weight") for name in linearized.param_names)

    delta_module = getattr(model, "_delta_module")
    weight_idx = next(i for i, name in enumerate(delta_module.names) if name.endswith("clip_model.model.visual.1.weight"))
    with torch.no_grad():
        delta_module.params[weight_idx].fill_(0.25)

    images = torch.randn(3, 3, 2, 2)
    actual = model(images)
    raw_current = model._current_param_map()
    current_visual = {
        key[len("clip_model.model.visual.") :]: value
        for key, value in raw_current.items()
        if key.startswith("clip_model.model.visual.")
    }
    manual_features = linearized.forward(
        current_module=model.clip_model.model.visual,
        current_params=current_visual,
        args=(images,),
    )
    expected = model.clip_model.logit_scale * (manual_features @ model.clip_model._zs_text_features.t())
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_vision_linearized_ntk_matches_manual_formula() -> None:
    torch.manual_seed(0)
    clf = _DummyClassifier()
    clf.build_zeroshot_text_features(["a", "b"], build_cfg=None)
    base_sd = _base_model_state(clf)
    with torch.no_grad():
        clf.model.visual[1].weight.add_(0.75)
    images = torch.randn(3, 3, 2, 2)

    get_forward_mode("linearized_ntk").bind(
        clf=clf,
        base_sd=base_sd,
        strict_load=True,
        params={"linearized_feature_normalization": True},
    )

    expected = _manual_linearized_logits(
        clf,
        base_sd=base_sd,
        images=images,
        linearized_feature_normalization=True,
    )
    assert torch.allclose(clf(images), expected, atol=1e-6, rtol=1e-5)


def test_vision_linearized_ntk_visual_features_are_raw_pre_normalization() -> None:
    torch.manual_seed(0)
    clf = _DummyClassifier()
    clf.build_zeroshot_text_features(["a", "b"], build_cfg=None)
    base_sd = _base_model_state(clf)
    with torch.no_grad():
        clf.model.visual[1].weight.add_(1.0)
    images = torch.randn(2, 3, 2, 2)

    get_forward_mode("linearized_ntk").bind(
        clf=clf,
        base_sd=base_sd,
        strict_load=True,
        params={
            "linearized_feature_normalization": False,
            "linearized_logit_normalization": True,
        },
    )

    logits = clf(images)
    expected_visual = _manual_linearized_raw_visual_features(
        clf,
        base_sd=base_sd,
        images=images,
    )
    assert isinstance(clf._last_visual_features, torch.Tensor)
    assert isinstance(clf._last_image_features, torch.Tensor)
    assert torch.allclose(clf._last_visual_features, expected_visual, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(clf._last_visual_features, clf._last_image_features)
    expected_logits = zero_shot_logits_from_features(
        clf,
        clf._last_image_features,
        normalize_image_features=True,
    )
    assert torch.allclose(logits, expected_logits, atol=1e-6, rtol=1e-5)


def test_vision_linearized_ntk_can_match_old_distillation_normalization() -> None:
    torch.manual_seed(0)
    clf = _DummyClassifier()
    clf.build_zeroshot_text_features(["a", "b"], build_cfg=None)
    base_sd = _base_model_state(clf)
    with torch.no_grad():
        clf.model.visual[1].weight.add_(1.0)
    images = torch.randn(2, 3, 2, 2)

    get_forward_mode("linearized_ntk").bind(
        clf=clf,
        base_sd=base_sd,
        strict_load=True,
        params={
            "linearized_feature_normalization": False,
            "linearized_logit_normalization": True,
        },
    )

    expected = _manual_linearized_logits(
        clf,
        base_sd=base_sd,
        images=images,
        linearized_feature_normalization=False,
        linearized_logit_normalization=True,
    )
    assert torch.allclose(clf(images), expected, atol=1e-6, rtol=1e-5)


def test_vision_linearized_ntk_flag_controls_post_sum_normalization() -> None:
    torch.manual_seed(0)
    clf_true = _DummyClassifier()
    clf_false = _DummyClassifier()
    clf_false.load_state_dict(clf_true.state_dict())
    clf_true.build_zeroshot_text_features(["a", "b"], build_cfg=None)
    clf_false._zs_text_features = clf_true._zs_text_features.detach().clone()
    base_sd = _base_model_state(clf_true)
    with torch.no_grad():
        clf_true.model.visual[1].weight.add_(1.0)
        clf_false.model.visual[1].weight.copy_(clf_true.model.visual[1].weight)
    images = torch.randn(2, 3, 2, 2)

    get_forward_mode("linearized_ntk").bind(
        clf=clf_true,
        base_sd=base_sd,
        strict_load=True,
        params={"linearized_feature_normalization": True},
    )
    get_forward_mode("linearized_ntk").bind(
        clf=clf_false,
        base_sd=base_sd,
        strict_load=True,
        params={"linearized_feature_normalization": False},
    )

    expected_true = _manual_linearized_logits(
        clf_true,
        base_sd=base_sd,
        images=images,
        linearized_feature_normalization=True,
    )
    expected_false = _manual_linearized_logits(
        clf_false,
        base_sd=base_sd,
        images=images,
        linearized_feature_normalization=False,
    )

    actual_true = clf_true(images)
    actual_false = clf_false(images)
    assert torch.allclose(actual_true, expected_true, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_false, expected_false, atol=1e-6, rtol=1e-5)
    assert isinstance(clf_true._last_image_features, torch.Tensor)
    assert isinstance(clf_false._last_image_features, torch.Tensor)
    assert not torch.allclose(clf_true._last_image_features, clf_false._last_image_features)


def test_vision_training_and_eval_linearized_ntk_share_same_logit_path() -> None:
    torch.manual_seed(0)
    eval_clf = _DummyClassifier()
    eval_clf.build_zeroshot_text_features(["a", "b"], build_cfg=None)
    train_clf = _DummyClassifier()
    train_clf.load_state_dict(eval_clf.state_dict())
    train_clf._zs_text_features = eval_clf._zs_text_features.detach().clone()
    base_sd = _base_model_state(eval_clf)
    train_model = train_vision.ImageEncoder(train_clf)

    get_forward_mode("linearized_ntk").bind(
        clf=eval_clf,
        base_sd=base_sd,
        strict_load=True,
        params={"linearized_feature_normalization": True},
    )
    bind_training_forward_mode(
        model=train_model,
        forward_mode="linearized_ntk",
        base_sd=base_sd,
        strict_load=True,
        params={"linearized_feature_normalization": True},
    )

    with torch.no_grad():
        eval_clf.model.visual[1].weight.add_(0.5)
        train_clf.model.visual[1].weight.copy_(eval_clf.model.visual[1].weight)
    images = torch.randn(4, 3, 2, 2)
    assert torch.allclose(eval_clf(images), train_model(images), atol=1e-6, rtol=1e-5)


def test_bind_training_forward_mode_materializes_lora_weight_space_for_vision() -> None:
    torch.manual_seed(0)
    clf = _DummyClassifier()
    clf.normalize = False
    clf.build_zeroshot_text_features(["a", "b"], build_cfg=None)
    clf.model.visual = get_peft_model(
        clf.model.visual,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["1"],
            bias="none",
        ),
    )
    model = train_vision.ImageEncoder(clf)
    base_sd = {k: v.detach().clone() for k, v in clf.model.state_dict().items()}

    bind_training_forward_mode(
        model=model,
        forward_mode="linearized_ntk",
        base_sd=base_sd,
        strict_load=True,
        params={"linearized_feature_normalization": False, "linearized_logit_normalization": False},
    )

    linearized = getattr(model, "_linearized_module")
    assert all("lora_" not in name for name in linearized.param_names)
    assert any(name.endswith("base_layer.weight") for name in linearized.param_names)

    named_params = dict(model.named_parameters())
    with torch.no_grad():
        named_params["clip_model.model.visual.base_model.model.1.lora_B.default.weight"].fill_(0.25)

    images = torch.randn(3, 3, 2, 2)
    actual = model(images)
    current_params = materialized_peft_param_map(model.clip_model.model.visual)
    manual_features = linearized.forward(
        current_module=model.clip_model.model.visual,
        current_params=current_params,
        args=(images,),
    )
    expected = model.clip_model.logit_scale * (manual_features @ model.clip_model._zs_text_features.t())
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    actual.sum().backward()
    assert float(named_params["clip_model.model.visual.base_model.model.1.lora_A.default.weight"].grad.abs().sum()) > 0.0
    assert float(named_params["clip_model.model.visual.base_model.model.1.lora_B.default.weight"].grad.abs().sum()) > 0.0


def test_apply_training_forward_mode_materializes_lora_weight_space_for_text() -> None:
    torch.manual_seed(0)

    class _TinyPeftTextModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 2)
            self.config = {"tie_word_embeddings": False, "num_labels": 2}

        def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
            del attention_mask, labels, kwargs
            if input_ids is None:
                input_ids = inputs_embeds
            logits = self.proj(input_ids.float())
            return SimpleNamespace(loss=None, logits=logits)

    model = get_peft_model(
        _TinyPeftTextModel(),
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["proj"],
            bias="none",
        ),
    )
    apply_training_forward_mode(
        model=model,
        forward_mode="linearized_ntk",
        device=torch.device("cpu"),
        output_transform=lambda out: out.logits,
        output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
    )

    linearized = getattr(model, "_linearized_module")
    assert all("lora_" not in name for name in linearized.param_names)
    assert any(name.endswith("proj.base_layer.weight") for name in linearized.param_names)

    named_params = dict(model.named_parameters())
    with torch.no_grad():
        named_params["base_model.model.proj.lora_B.default.weight"].fill_(0.25)

    inputs = torch.randn(4, 4)
    actual = model(input_ids=inputs).logits
    current_params = materialized_peft_param_map(model)
    manual = linearized.forward(
        current_module=model,
        current_params=current_params,
        kwargs={"input_ids": inputs},
        output_transform=lambda out: out.logits,
    )
    assert torch.allclose(actual, manual, atol=1e-6, rtol=1e-5)

    actual.sum().backward()
    assert float(named_params["base_model.model.proj.lora_A.default.weight"].grad.abs().sum()) > 0.0
    assert float(named_params["base_model.model.proj.lora_B.default.weight"].grad.abs().sum()) > 0.0


def test_train_text_saves_forward_mode_metadata(tmp_path, monkeypatch) -> None:
    train_batches = [
        {"input_ids": torch.ones(2, 4), "labels": torch.tensor([0, 1])},
        {"input_ids": torch.zeros(2, 4), "labels": torch.tensor([1, 0])},
    ]
    val_batches = [{"input_ids": torch.ones(2, 4), "labels": torch.tensor([0, 1])}]
    test_batches = [{"input_ids": torch.ones(2, 4), "labels": torch.tensor([0, 1])}]

    monkeypatch.setattr(
        train_text.TextLM,
        "build",
        staticmethod(lambda build_cfg: SimpleNamespace(model=_TinyTextModel(), tokenizer=object())),
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
        strategy_cfg={"forward_mode": "linearized_ntk"},
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
    )

    ckpt = torch.load(summary["best_ckpt_path"], map_location="cpu")
    assert summary["forward_mode"] == "linearized_ntk"
    assert ckpt["forward_mode"] == "linearized_ntk"
    assert summary["trainable"]["linearized_params"] > 0


def test_train_vision_saves_forward_mode_metadata(tmp_path, monkeypatch) -> None:
    loaders = SimpleNamespace(
        train=[
            (torch.ones(2, 3, 2, 2), torch.tensor([0, 1])),
            (torch.zeros(2, 3, 2, 2), torch.tensor([1, 0])),
        ],
        val=[(torch.ones(2, 3, 2, 2), torch.tensor([0, 1]))],
        test=[(torch.ones(2, 3, 2, 2), torch.tensor([0, 1]))],
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

    summary = train_vision.train_task(
        task="dummy",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "validation": "validation", "test": "test"},
        build_cfg=OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32"),
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
        strategy_cfg={
            "forward_mode": "linearized_ntk",
            "forward_mode_params": {"linearized_feature_normalization": False, "linearized_logit_normalization": True},
        },
    )

    ckpt = torch.load(summary["best_ckpt_path"], map_location="cpu")
    assert summary["forward_mode"] == "linearized_ntk"
    assert summary["forward_mode_params"] == {"linearized_feature_normalization": False, "linearized_logit_normalization": True}
    assert ckpt["forward_mode"] == "linearized_ntk"
    assert ckpt["forward_mode_params"] == {"linearized_feature_normalization": False, "linearized_logit_normalization": True}
    assert summary["trainable"]["linearized_params"] > 0


def test_train_vision_linear_probe_composes_with_linearized_ntk(tmp_path, monkeypatch) -> None:
    loaders = SimpleNamespace(
        train=[
            (torch.ones(2, 3, 2, 2), torch.tensor([0, 1])),
            (torch.zeros(2, 3, 2, 2), torch.tensor([1, 0])),
        ],
        val=[(torch.ones(2, 3, 2, 2), torch.tensor([0, 1]))],
        test=[(torch.ones(2, 3, 2, 2), torch.tensor([0, 1]))],
        classnames=["a", "b"],
    )

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _DummyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)

    summary = train_vision.train_task(
        task="dummy",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "validation": "validation", "test": "test"},
        build_cfg=OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32"),
        strategy="linear_probe",
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
        strategy_cfg={"forward_mode": "linearized_ntk"},
    )

    ckpt = torch.load(summary["best_ckpt_path"], map_location="cpu")
    assert summary["strategy"] == "linear_probe"
    assert summary["forward_mode"] == "linearized_ntk"
    assert summary["trainable"]["head_params"] > 0
    assert ckpt["forward_mode"] == "linearized_ntk"
    assert "head.weight" in ckpt["state_dict"]
