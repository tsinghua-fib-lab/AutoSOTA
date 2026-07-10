"""Tests for the GradFix rebase method."""
from __future__ import annotations

import torch
import torch.nn as nn

from merge_and_rebase.models.grad_recipes import (
    GradRecipe,
    causal_lm_recipe,
    clip_contrastive_recipe,
    seq_classification_recipe,
)
from merge_and_rebase.rebase.methods.gradfix import (
    GradFixRebase,
    apply_gradfix_mask,
    compute_gradient_signs,
)
from merge_and_rebase.rebase.registry import get_method, list_methods

# ---- Registry -----------------------------------------------------------

def test_gradfix_registered() -> None:
    assert "gradfix" in list_methods()
    m = get_method("gradfix")
    assert m.name == "gradfix"


# ---- apply_gradfix_mask: normal mode ------------------------------------

def test_normal_mask_agreement_kept() -> None:
    """Where sign(grad) == sign(delta) the delta must be preserved."""
    grad_signs = {"w": torch.tensor([1.0, -1.0, 1.0, -1.0])}
    delta = {"w": torch.tensor([3.0, -2.0, 5.0, -7.0])}  # same sign
    out = apply_gradfix_mask(grad_signs, delta, mask_mode="normal")
    assert torch.allclose(out["w"], delta["w"])


def test_normal_mask_disagreement_zeroed() -> None:
    """Where sign(grad) != sign(delta) the delta must be zeroed."""
    grad_signs = {"w": torch.tensor([1.0, -1.0, 1.0, -1.0])}
    delta = {"w": torch.tensor([-3.0, 2.0, -5.0, 7.0])}  # opposite sign
    out = apply_gradfix_mask(grad_signs, delta, mask_mode="normal")
    assert torch.allclose(out["w"], torch.zeros(4))


def test_normal_mask_mixed() -> None:
    grad_signs = {"w": torch.tensor([1.0, -1.0, 1.0, -1.0])}
    delta = {"w": torch.tensor([3.0, 2.0, -5.0, -7.0])}
    out = apply_gradfix_mask(grad_signs, delta, mask_mode="normal")
    expected = torch.tensor([3.0, 0.0, 0.0, -7.0])
    assert torch.allclose(out["w"], expected)


# ---- apply_gradfix_mask: force mode --------------------------------------

def test_force_mask_overrides_sign() -> None:
    grad_signs = {"w": torch.tensor([1.0, -1.0, 1.0, -1.0])}
    delta = {"w": torch.tensor([-3.0, 2.0, -5.0, 7.0])}
    out = apply_gradfix_mask(grad_signs, delta, mask_mode="force")
    # force: |delta| * sign(grad)
    expected = torch.tensor([3.0, -2.0, 5.0, -7.0])
    assert torch.allclose(out["w"], expected)


def test_force_preserves_magnitude() -> None:
    grad_signs = {"w": torch.tensor([1.0, 1.0, -1.0, -1.0])}
    delta = {"w": torch.tensor([4.0, -4.0, 4.0, -4.0])}
    out = apply_gradfix_mask(grad_signs, delta, mask_mode="force")
    expected = torch.tensor([4.0, 4.0, -4.0, -4.0])
    assert torch.allclose(out["w"], expected)


# ---- Passthrough keys not in gradient_signs ------------------------------

def test_uncovered_keys_passthrough() -> None:
    """Keys absent from gradient_signs should be kept unchanged."""
    grad_signs = {"w1": torch.tensor([1.0])}
    delta = {"w1": torch.tensor([2.0]), "w2": torch.tensor([9.0])}
    out = apply_gradfix_mask(grad_signs, delta, mask_mode="normal")
    assert torch.allclose(out["w1"], torch.tensor([2.0]))
    assert torch.allclose(out["w2"], torch.tensor([9.0]))


# ---- Multi-key -----------------------------------------------------------

def test_multi_key() -> None:
    grad_signs = {
        "a": torch.tensor([1.0, -1.0]),
        "b": torch.tensor([-1.0, 1.0]),
    }
    delta = {
        "a": torch.tensor([5.0, 5.0]),
        "b": torch.tensor([-3.0, -3.0]),
    }
    out = apply_gradfix_mask(grad_signs, delta, mask_mode="normal")
    assert torch.allclose(out["a"], torch.tensor([5.0, 0.0]))
    assert torch.allclose(out["b"], torch.tensor([-3.0, 0.0]))


# ---- GradFixRebase.apply direct call ------------------------------------

def test_gradfix_rebase_apply_method() -> None:
    """GradFixRebase.apply() should delegate to apply_gradfix_mask."""
    gf = GradFixRebase()
    grad_signs = {"w": torch.tensor([1.0, -1.0])}
    delta = {"w": torch.tensor([2.0, 2.0])}
    out = gf.apply(grad_signs, delta=delta, mask_mode="normal")
    expected = torch.tensor([2.0, 0.0])
    assert torch.allclose(out["w"], expected)


# ---- Invalid mode raises -------------------------------------------------

def test_invalid_mask_mode_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="mask_mode"):
        apply_gradfix_mask({}, {}, mask_mode="invalid")


# ---- GradRecipe + compute_gradient_signs (model-agnostic) ----------------

class _TinyLinear(nn.Module):
    """Minimal model for testing gradient sign computation."""
    def __init__(self, d_in: int = 4, d_out: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_tiny_recipe(device: str = "cpu") -> GradRecipe:
    """Recipe that takes (x, y) batches and returns CE loss on fc params."""
    loss_fn = nn.CrossEntropyLoss()
    def _recipe(model: nn.Module, batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device).long()
        logits = model(x)
        loss = loss_fn(logits, y)
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        return loss, params
    return _recipe


def _make_tiny_dataloader():
    """Two deterministic batches."""
    torch.manual_seed(0)
    data = [
        (torch.randn(8, 4), torch.randint(0, 2, (8,))),
        (torch.randn(8, 4), torch.randint(0, 2, (8,))),
    ]
    return data


def test_compute_gradient_signs_mean_returns_signs() -> None:
    model = _TinyLinear()
    recipe = _make_tiny_recipe()
    dl = _make_tiny_dataloader()
    signs = compute_gradient_signs(model, dl, recipe=recipe, device="cpu", vote="mean")
    assert len(signs) > 0
    for name, t in signs.items():
        # Every entry must be -1, 0, or +1
        assert torch.all((t == -1) | (t == 0) | (t == 1))


def test_compute_gradient_signs_max_returns_signs() -> None:
    model = _TinyLinear()
    recipe = _make_tiny_recipe()
    dl = _make_tiny_dataloader()
    signs = compute_gradient_signs(model, dl, recipe=recipe, device="cpu", vote="max")
    assert len(signs) > 0
    for name, t in signs.items():
        assert torch.all((t == -1) | (t == 0) | (t == 1))


def test_compute_gradient_signs_deterministic() -> None:
    """Same seed → same signs."""
    torch.manual_seed(42)
    m1 = _TinyLinear()
    torch.manual_seed(42)
    m2 = _TinyLinear()
    recipe = _make_tiny_recipe()
    dl1 = _make_tiny_dataloader()
    dl2 = _make_tiny_dataloader()
    s1 = compute_gradient_signs(m1, dl1, recipe=recipe, device="cpu", vote="mean")
    s2 = compute_gradient_signs(m2, dl2, recipe=recipe, device="cpu", vote="mean")
    for k in s1:
        assert torch.equal(s1[k], s2[k])


def test_gradfix_prepare_with_recipe() -> None:
    """GradFixRebase.prepare() should work with a custom recipe."""
    gf = GradFixRebase()
    model = _TinyLinear()
    recipe = _make_tiny_recipe()
    dl = _make_tiny_dataloader()
    signs = gf.prepare(
        target_model=model,
        target_dataloader=dl,
        recipe=recipe,
        device="cpu",
        vote="mean",
    )
    assert isinstance(signs, dict)
    assert len(signs) > 0


def test_gradfix_transport_with_recipe() -> None:
    """GradFixRebase.transport() end-to-end with a recipe."""
    gf = GradFixRebase()
    model = _TinyLinear()
    recipe = _make_tiny_recipe()
    dl = _make_tiny_dataloader()

    base_a = {"fc.weight": torch.zeros(2, 4)}
    base_b = {"fc.weight": torch.ones(2, 4)}
    delta = {"fc.weight": torch.randn(2, 4)}

    masked = gf.transport(
        source_base=base_a,
        target_base=base_b,
        delta=delta,
        target_model=model,
        target_dataloader=dl,
        recipe=recipe,
        device="cpu",
        mask_mode="normal",
        vote="mean",
    )
    assert "fc.weight" in masked


# ---- Recipe factory smoke tests -----------------------------------------

def test_clip_contrastive_recipe_callable() -> None:
    """clip_contrastive_recipe returns a callable."""

    class _FakeClf:
        normalize = True

        def _compute_zeroshot_text_features(self, classnames, cfg):
            return torch.randn(len(classnames), 4)

    recipe = clip_contrastive_recipe(_FakeClf(), ["cat", "dog"], cfg=None)
    assert callable(recipe)


def test_causal_lm_recipe_callable() -> None:
    """causal_lm_recipe returns a callable."""
    recipe = causal_lm_recipe(device="cpu")
    assert callable(recipe)


def test_seq_classification_recipe_callable() -> None:
    """seq_classification_recipe returns a callable."""
    recipe = seq_classification_recipe(device="cpu")
    assert callable(recipe)


def test_causal_lm_recipe_with_tiny_model() -> None:
    """Integration: causal_lm_recipe with a minimal model."""
    class _FakeLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 8)
            self.head = nn.Linear(8, 10, bias=False)

        def forward(self, input_ids, attention_mask=None, labels=None):
            h = self.embed(input_ids)
            logits = self.head(h)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits.view(-1, 10), labels.view(-1))

            class _Out:
                pass
            out = _Out()
            out.logits = logits
            out.loss = loss
            return out

    model = _FakeLM()
    recipe = causal_lm_recipe(device="cpu")

    batch = {
        "input_ids": torch.randint(0, 10, (4, 6)),
        "attention_mask": torch.ones(4, 6, dtype=torch.long),
        "labels": torch.randint(0, 10, (4, 6)),
    }

    signs = compute_gradient_signs(model, [batch], recipe=recipe, device="cpu", vote="mean")
    assert len(signs) > 0
    for t in signs.values():
        assert torch.all((t == -1) | (t == 0) | (t == 1))
