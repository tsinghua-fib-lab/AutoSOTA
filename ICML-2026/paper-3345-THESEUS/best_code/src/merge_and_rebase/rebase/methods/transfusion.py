"""
TransFusion rebase method: permutation-based task vector transport.

Implements the TransFusion workflow from repro_transfusion_main.py as a
RebaseMethod that can be used with vision_rebase --method transfusion.

Follows the prepare/apply/transport pattern used by GradFix and Theseus:
  - prepare(): one-time expensive setup (model patching, permutation compute, sanity check)
  - apply(): cheap permutation application to a task delta
  - transport(): convenience pipeline = prepare + apply
"""
from __future__ import annotations

import pickle
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..base import TensorDict
from ..permutations import (
    CLIP_Visual_PermutationSpecBuilder,
    LayerIterationOrder,
    WeightMatcher,
    apply_permutation_to_statedict,
)
from ..registry import register


@dataclass(frozen=True)
class TransFusionRebase:
    """
    TransFusion: permutation-based task vector transport.

    Computes permutations between source and target base models using
    WeightMatcher, then applies the permutation to the task vector delta
    so it can be applied to the target model.

    Permutations are cached to disk per (source_tag, target_tag, seed).
    """

    name: str = "transfusion"

    def prepare(
        self,
        *,
        clf_source: Any,
        clf_target: Any,
        source_loaders: Any | None = None,
        classnames: list[str] | None = None,
        source_build_cfg: Any | None = None,
        device: str = "cuda",
        seed: int = 42,
        perm_cache_source: str = "source",
        perm_cache_target: str = "target",
        perm_cache_dir: str = "permutations",
        perm_cache_mode: str = "auto",
        max_iter: int = 100,
        layer_iteration_order: str = "random",
        intra_head: bool = True,
        sanity_check_functional_equivalent: bool = False,
        sanity_check_first_n_batches: int | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        One-time expensive setup for TransFusion rebase.

        1. Deepcopy + patch models (shortcuts + split QKV)
        2. Extract visual-only base state dicts (patched key space)
        3. Compute or load cached permutations
        4. Optional sanity check: evaluate source model zeroshot before/after permutation

        Returns dict with:
          - source_model_unpatched, source_model_patched, target_model_patched
          - source_base_sd, target_base_sd (visual-only, patched keys)
          - perm_indices, heads_indices
          - depth, num_heads
          - sanity_check_pre, sanity_check_post (only if sanity_check_functional_equivalent=True)
        """
        from ...eval.utils import to_cpu_fp32
        from ...io.ckpt import load_into_model
        from ..permutations.models import OpenCLIPModel as TransFusionOpenCLIPModel
        from ..permutations.transport import apply_visual_permutation_to_state

        if verbose:
            print("[transfusion] prepare: start")

        # 1. Deepcopy and patch models
        source_model_unpatched = deepcopy(clf_source.model)
        source_model_patched = deepcopy(clf_source.model)
        target_model_patched = deepcopy(clf_target.model)
        TransFusionOpenCLIPModel(source_model_patched)
        TransFusionOpenCLIPModel(target_model_patched)
        if verbose:
            print("[transfusion] prepare: patched shortcuts and split QKV")

        # 2. Extract visual-only base SDs
        source_base_sd = to_cpu_fp32({k: v for k, v in source_model_patched.visual.state_dict().items()})
        target_base_sd = to_cpu_fp32({k: v for k, v in target_model_patched.visual.state_dict().items()})

        # 3. Infer depth and num_heads
        depth = _infer_depth(source_base_sd)
        num_heads = _infer_num_heads(source_base_sd)

        # 4. Build permutation spec and resolve cache path
        ps = CLIP_Visual_PermutationSpecBuilder(depth=depth, prefix="").create_permutation_spec()
        cache_path = _build_perm_cache_path(perm_cache_dir, perm_cache_source, perm_cache_target, seed)

        # 5. Load or compute permutations
        perm_indices, heads_indices = _load_or_compute_permutations(
            ps=ps,
            source_base=source_base_sd,
            target_base=target_base_sd,
            cache_path=cache_path,
            cache_mode=perm_cache_mode,
            max_iter=max_iter,
            layer_iteration_order=layer_iteration_order,
            intra_head=intra_head,
            num_heads=num_heads,
        )

        # 6. Optional sanity check
        sanity_check_pre: float | None = None
        sanity_check_post: float | None = None
        if sanity_check_functional_equivalent and source_loaders is not None and classnames is not None and source_build_cfg is not None:
            if verbose:
                print("[transfusion] prepare: running sanity check (functional equivalent)")

            zero_pre = _evaluate_source_zeroshot(
                clf_source=clf_source,
                model=source_model_patched,
                loaders_obj=source_loaders,
                classnames_task=classnames,
                source_build_cfg_task=source_build_cfg,
                split="test",
                device=device,
                first_n_batches=sanity_check_first_n_batches,
            )
            sanity_check_pre = float(zero_pre)

            permuted_sd = apply_visual_permutation_to_state(
                state=source_base_sd,
                perm_indices=perm_indices,
                heads_indices=heads_indices,
                prefix="",
                depth=depth,
                num_heads=num_heads,
                split_qkv=True,
                reference=source_base_sd,
                device=device,
            )
            perm_model = deepcopy(source_model_patched)
            load_into_model(perm_model.visual, permuted_sd, strict=False)

            zero_post = _evaluate_source_zeroshot(
                clf_source=clf_source,
                model=perm_model,
                loaders_obj=source_loaders,
                classnames_task=classnames,
                source_build_cfg_task=source_build_cfg,
                split="test",
                device=device,
                first_n_batches=sanity_check_first_n_batches,
            )
            sanity_check_post = float(zero_post)

            if verbose:
                print(
                    f"[transfusion] prepare: sanity check — "
                    f"source zeroshot {sanity_check_pre:.6f} -> permuted {sanity_check_post:.6f} "
                    f"(delta={sanity_check_post - sanity_check_pre:+.6f})"
                )

        if verbose:
            print("[transfusion] prepare: done")

        return {
            "source_model_unpatched": source_model_unpatched,
            "source_model_patched": source_model_patched,
            "target_model_patched": target_model_patched,
            "source_base_sd": source_base_sd,
            "target_base_sd": target_base_sd,
            "perm_indices": {k: v.cpu() for k, v in perm_indices.items()},
            "heads_indices": (
                {k: {hk: hv.cpu() for hk, hv in v.items()} for k, v in heads_indices.items()}
                if heads_indices is not None
                else None
            ),
            "depth": depth,
            "num_heads": num_heads,
            "sanity_check_pre": sanity_check_pre,
            "sanity_check_post": sanity_check_post,
        }

    def apply(
        self,
        prepared: Mapping[str, Any],
        *,
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        **kwargs,
    ) -> TensorDict:
        """Apply pre-computed permutations to a task delta."""
        perm_indices = prepared["perm_indices"]
        heads_indices = prepared["heads_indices"]
        depth = prepared["depth"]
        num_heads = prepared["num_heads"]

        ps = CLIP_Visual_PermutationSpecBuilder(depth=depth, prefix="").create_permutation_spec()

        delta_cuda = {k: v.cuda() for k, v in delta.items()}
        perm_device = {k: v.cuda() for k, v in perm_indices.items()}
        heads_device = None
        if heads_indices is not None:
            heads_device = {
                k: {hk: hv.cuda() for hk, hv in v.items()}
                for k, v in heads_indices.items()
            }

        transported = apply_permutation_to_statedict(
            ps=ps,
            perm_matrices=perm_device,
            model_a_dict=delta_cuda,
            heads_permutation=heads_device,
            skip_params=False,
            num_heads=num_heads,
        )

        return {k: v.cpu() for k, v in transported.items()}

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        prepared: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> TensorDict:
        """
        Full pipeline: use provided prepared state, or fall back to internal prepare.

        You can either:
          - pass ``prepared`` directly (if you already called ``prepare``)
          - pass ``clf_source`` + ``clf_target`` and setup will be done here.
        """
        if prepared is None:
            prepared = self.prepare(
                clf_source=kwargs.pop("clf_source"),
                clf_target=kwargs.pop("clf_target"),
                device=kwargs.pop("device", "cuda"),
                seed=int(kwargs.pop("seed", 42)),
                **kwargs,
            )

        return self.apply(prepared, delta=delta, strict=strict)

    def load_task_checkpoint(
        self,
        ckpt_path: str,
        source_model_unpatched: nn.Module,
        verbose: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Load a fine-tuned checkpoint into an unpatched source model, then patch and extract visual SD."""
        from ...eval.utils import to_cpu_fp32
        from ...io.ckpt import load_ckpt
        from ..permutations.models import OpenCLIPModel as TransFusionOpenCLIPModel

        ft_model = deepcopy(source_model_unpatched)
        sd = load_ckpt(ckpt_path)
        n_miss, n_unexp = ft_model.load_state_dict(sd, strict=False)
        if verbose and n_miss:
            print(f"  TransFusion: {len(n_miss)} missing keys in checkpoint")
        if verbose and n_unexp:
            print(f"  TransFusion: {len(n_unexp)} unexpected keys in checkpoint")
        TransFusionOpenCLIPModel(ft_model)
        return to_cpu_fp32({k: v for k, v in ft_model.visual.state_dict().items()})

    def compute_task_delta(
        self,
        tuned_sd: dict[str, torch.Tensor],
        source_base_sd: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute task delta as key-wise subtraction in patched key space."""
        return {
            k: tuned_sd[k] - source_base_sd[k]
            for k in source_base_sd
            if k in tuned_sd
        }

    def load_into_target_visual(
        self,
        clf_target: Any,
        sd: dict[str, torch.Tensor],
        strict: bool = False,
    ) -> None:
        """Load state dict into the visual submodule of the patched target model."""
        from ...io.ckpt import load_into_model
        load_into_model(clf_target.model.visual, sd, strict=strict)


def _evaluate_source_zeroshot(
    *,
    clf_source: Any,
    model: torch.nn.Module,
    loaders_obj: Any,
    classnames_task: list[str],
    source_build_cfg_task: Any,
    split: str,
    device: str,
    first_n_batches: int | None = None,
) -> float:
    """Evaluate zeroshot top-1 accuracy on a source model."""
    import itertools

    from ...models.openclip_classifier import OpenClipClassifier

    eval_loader = loaders_obj.val if getattr(loaders_obj, "val", None) is not None else loaders_obj.test
    if first_n_batches is not None:
        eval_loader = itertools.islice(iter(eval_loader), max(1, int(first_n_batches)))

    eval_clf = OpenClipClassifier(
        model=model,
        tokenizer=clf_source.tokenizer,
        preprocess=clf_source.preprocess,
        normalize=clf_source.normalize,
        logit_scale=clf_source.logit_scale,
    )
    eval_clf.build_zeroshot_text_features(
        classnames_task,
        source_build_cfg_task,
        cache_dir="src/.cache/zs_cache",
        force_rebuild=False,
    )
    return float(eval_clf.top1(eval_loader, device=device))


def _infer_depth(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer transformer depth from state dict keys."""
    import re

    pattern = re.compile(r"transformer\.resblocks\.(\d+)\.")
    indices = set()
    for key in state_dict:
        m = pattern.search(key)
        if m:
            indices.add(int(m.group(1)))
    if not indices:
        raise ValueError("Could not infer transformer depth from state dict keys.")
    return max(indices) + 1


def _infer_num_heads(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer number of attention heads from state dict."""
    q_key = "transformer.resblocks.0.attn.q.weight"
    if q_key not in state_dict:
        raise ValueError(f"Could not find '{q_key}' in state dict to infer num_heads.")
    embed_dim = state_dict[q_key].shape[1]
    for head_dim in [64, 80, 96]:
        if embed_dim % head_dim == 0:
            return embed_dim // head_dim
    return embed_dim // 64


def _build_perm_cache_path(
    cache_dir: str, source_tag: str, target_tag: str, seed: int
) -> Path:
    """Build permutation cache file path."""
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    fname = f"permutations_visual_{source_tag}_to_{target_tag}_{seed}.pkl"
    return path / fname


def _load_or_compute_permutations(
    *,
    ps,
    source_base: dict[str, torch.Tensor],
    target_base: dict[str, torch.Tensor],
    cache_path: Path,
    cache_mode: str,
    max_iter: int,
    layer_iteration_order: str,
    intra_head: bool,
    num_heads: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]] | None]:
    """Load permutations from cache or compute them via WeightMatcher."""
    can_load = cache_mode in {"auto", "load"} and cache_path.exists()
    can_save = cache_mode in {"auto", "save"}

    if can_load:
        with cache_path.open("rb") as f:
            perm_indices, heads_indices = pickle.load(f)
        print(f"  TransFusion: loaded permutations from {cache_path}")
        return perm_indices, heads_indices

    if not torch.cuda.is_available():
        raise RuntimeError("TransFusion permutation compute requires CUDA.")

    target_cuda = {k: v.cuda() for k, v in target_base.items()}
    source_cuda = {k: v.cuda() for k, v in source_base.items()}

    order_map = {
        "random": LayerIterationOrder.RANDOM,
        "forward": LayerIterationOrder.FORWARD,
        "backward": LayerIterationOrder.BACKWARD,
        "alternate": LayerIterationOrder.ALTERNATE,
    }
    order = order_map.get(layer_iteration_order.lower(), LayerIterationOrder.RANDOM)

    matcher = WeightMatcher(
        ps=ps,
        fixed=target_cuda,
        permutee=source_cuda,
        max_iter=max_iter,
        layer_iteration_order=order,
        intra_head=intra_head,
        num_heads=num_heads,
    )
    perm_indices, heads_indices = matcher.run()

    if can_save:
        with cache_path.open("wb") as f:
            pickle.dump((perm_indices, heads_indices), f)
        print(f"  TransFusion: saved permutations to {cache_path}")

    return perm_indices, heads_indices


register(TransFusionRebase())
