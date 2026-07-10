from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...models.patch_openclip_attention import merge_openclip_vit_attn, split_openclip_vit_attn
from ..base import TensorDict
from ..registry import register

logger = logging.getLogger(__name__)

_VISUAL_PREFIX = "visual."
_ZERO_KEYS = {"class_embedding", "positional_embedding", "conv1.weight"}
_FUSED_IN_PROJ_WEIGHT = ".attn.in_proj_weight"
_FUSED_IN_PROJ_BIAS = ".attn.in_proj_bias"
_Q_PROJ_WEIGHT = ".attn.q_proj.weight"
_K_PROJ_WEIGHT = ".attn.k_proj.weight"
_V_PROJ_WEIGHT = ".attn.v_proj.weight"
_Q_PROJ_BIAS = ".attn.q_proj.bias"
_K_PROJ_BIAS = ".attn.k_proj.bias"
_V_PROJ_BIAS = ".attn.v_proj.bias"

_ACTIVATION_COVARIANCE_MODES = {"activation", "activations"}
_DATA_FREE_COVARIANCE_MODES = {"data_free", "data-free", "weight", "weights", "weight_space", "weight-space"}


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _extract_model_inputs(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        for key in ("pixel_values", "images", "image", "inputs", "x"):
            value = batch.get(key, None)
            if torch.is_tensor(value):
                return value
    if isinstance(batch, (tuple, list)) and batch:
        first = batch[0]
        if torch.is_tensor(first):
            return first
    raise TypeError("Unsupported batch format for Theseus calibration.")


def _extract_output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    raise TypeError("Unsupported module output while collecting Theseus activations.")


def _encode_image(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image") and callable(model.encode_image):
        return model.encode_image(images)
    if hasattr(model, "visual") and callable(model.visual):
        return model.visual(images)
    return model(images)


def _visual_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.visual if hasattr(model, "visual") else model


def _has_fused_mha(visual: torch.nn.Module) -> bool:
    transformer = getattr(visual, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        return False
    for block in resblocks:
        if isinstance(getattr(block, "attn", None), nn.MultiheadAttention):
            return True
    return False


def _split_fused_qkv_if_needed(model: torch.nn.Module) -> int:
    visual = _visual_module(model)
    if not _has_fused_mha(visual):
        return 0

    ref_param = next(visual.parameters(), None)
    ref_device = ref_param.device if ref_param is not None else torch.device("cpu")
    ref_dtype = ref_param.dtype if ref_param is not None else None

    n_patched = int(
        split_openclip_vit_attn(
            visual,
            proj_dropout=0.0,
            attn_impl="softmax",
        )
    )

    if n_patched > 0:
        if ref_dtype is None:
            visual.to(device=ref_device)
        else:
            visual.to(device=ref_device, dtype=ref_dtype)

    return n_patched


def _visual_state_dict(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    visual = {key[len(_VISUAL_PREFIX) :]: value for key, value in sd.items() if key.startswith(_VISUAL_PREFIX)}
    return visual if visual else dict(sd)


def _visual_delta_keys(delta: Mapping[str, torch.Tensor]) -> dict[str, str]:
    visual = {key[len(_VISUAL_PREFIX) :]: key for key in delta if key.startswith(_VISUAL_PREFIX)}
    if visual:
        return visual
    return {key: key for key in delta}


def _split_fused_qkv_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith(_FUSED_IN_PROJ_WEIGHT) and value.ndim == 2 and value.shape[0] % 3 == 0:
            base = key[: -len(_FUSED_IN_PROJ_WEIGHT)]
            c = value.shape[0] // 3
            out[f"{base}{_Q_PROJ_WEIGHT}"] = value[:c, :]
            out[f"{base}{_K_PROJ_WEIGHT}"] = value[c : 2 * c, :]
            out[f"{base}{_V_PROJ_WEIGHT}"] = value[2 * c :, :]
            continue

        if key.endswith(_FUSED_IN_PROJ_BIAS) and value.ndim == 1 and value.shape[0] % 3 == 0:
            base = key[: -len(_FUSED_IN_PROJ_BIAS)]
            c = value.shape[0] // 3
            out[f"{base}{_Q_PROJ_BIAS}"] = value[:c]
            out[f"{base}{_K_PROJ_BIAS}"] = value[c : 2 * c]
            out[f"{base}{_V_PROJ_BIAS}"] = value[2 * c :]
            continue

        out[key] = value
    return out


def _merge_split_qkv_state(
    state: Mapping[str, torch.Tensor],
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = dict(state)

    def _merge_triplet(q_suffix: str, k_suffix: str, v_suffix: str, fused_suffix: str) -> None:
        prefixes: set[str] = set()
        for k in tuple(out.keys()):
            if k.endswith(q_suffix):
                prefixes.add(k[: -len(q_suffix)])
            elif k.endswith(k_suffix):
                prefixes.add(k[: -len(k_suffix)])
            elif k.endswith(v_suffix):
                prefixes.add(k[: -len(v_suffix)])

        for p in prefixes:
            qk = f"{p}{q_suffix}"
            kk = f"{p}{k_suffix}"
            vk = f"{p}{v_suffix}"
            fused = f"{p}{fused_suffix}"
            if qk not in out or kk not in out or vk not in out:
                continue
            if reference is not None and fused not in reference:
                continue

            merged = torch.cat([out[qk], out[kk], out[vk]], dim=0)
            out[fused] = merged
            del out[qk]
            del out[kk]
            del out[vk]

    _merge_triplet(_Q_PROJ_WEIGHT, _K_PROJ_WEIGHT, _V_PROJ_WEIGHT, _FUSED_IN_PROJ_WEIGHT)
    _merge_triplet(_Q_PROJ_BIAS, _K_PROJ_BIAS, _V_PROJ_BIAS, _FUSED_IN_PROJ_BIAS)
    return out


def _is_square(n: int) -> bool:
    if n <= 0:
        return False
    r = int(n**0.5)
    return r * r == n


def _standardize_tokens(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if x.ndim == 1:
        return x.view(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        if x.shape[0] == batch_size:
            return x
        if x.shape[1] == batch_size:
            return x.transpose(0, 1)
        return x
    if x.ndim == 4:
        return x
    return x.reshape(batch_size, -1, x.shape[-1])


def _to_tokens(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    x = _standardize_tokens(x, batch_size=batch_size)
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
    if x.ndim == 3:
        return x
    return x.reshape(batch_size, -1, x.shape[-1])


def _interp_linear_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[1] == target_tokens:
        return tokens
    tokens_t = tokens.transpose(1, 2)
    tokens_t = F.interpolate(tokens_t, size=target_tokens, mode="linear", align_corners=False)
    return tokens_t.transpose(1, 2)


def _interp_2d_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[1] == target_tokens:
        return tokens

    has_cls = _is_square(tokens.shape[1] - 1) and _is_square(target_tokens - 1)
    cls_token: torch.Tensor | None = None
    patch_tokens = tokens
    target_patch_tokens = target_tokens

    if has_cls:
        cls_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]
        target_patch_tokens = target_tokens - 1

    if not _is_square(patch_tokens.shape[1]) or not _is_square(target_patch_tokens):
        resized = _interp_linear_tokens(patch_tokens, target_patch_tokens)
        return torch.cat([cls_token, resized], dim=1) if cls_token is not None else resized

    src_side = int(patch_tokens.shape[1] ** 0.5)
    tgt_side = int(target_patch_tokens**0.5)
    x = patch_tokens.reshape(tokens.shape[0], src_side, src_side, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(tgt_side, tgt_side), mode="bilinear", align_corners=False)
    resized = x.permute(0, 2, 3, 1).reshape(tokens.shape[0], tgt_side * tgt_side, patch_tokens.shape[-1])
    return torch.cat([cls_token, resized], dim=1) if cls_token is not None else resized


def _align_features(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_feat.shape[0] != target_feat.shape[0]:
        raise ValueError(
            f"Theseus calibration expects aligned batch sizes. Got {source_feat.shape[0]} and {target_feat.shape[0]}."
        )

    source_tokens = _to_tokens(source_feat, batch_size=int(source_feat.shape[0]))
    target_tokens = _to_tokens(target_feat, batch_size=int(target_feat.shape[0]))

    if mode == "cls":
        source_tokens = source_tokens[:, :1, :]
        target_tokens = target_tokens[:, :1, :]
    elif mode == "mean":
        source_tokens = source_tokens.mean(dim=1, keepdim=True)
        target_tokens = target_tokens.mean(dim=1, keepdim=True)
    elif mode in {"interpolate2d", "interpolate_2d"}:
        source_tokens = _interp_2d_tokens(source_tokens, int(target_tokens.shape[1]))
    elif mode == "interpolate":
        source_tokens = _interp_linear_tokens(source_tokens, int(target_tokens.shape[1]))

    return source_tokens.reshape(-1, source_tokens.shape[-1]), target_tokens.reshape(-1, target_tokens.shape[-1])


class ActivationStore:
    """Streaming activation statistics with optional Gram and raw storage."""

    def __init__(self, *, store_raw: bool = False, store_a_gram: bool = False, store_b_gram: bool = False) -> None:
        self.store_raw = bool(store_raw)
        self.store_a_gram = bool(store_a_gram)
        self.store_b_gram = bool(store_b_gram)

        self.at_b: torch.Tensor | None = None
        self.at_a: torch.Tensor | None = None
        self.bt_b: torch.Tensor | None = None
        self.sum_a: torch.Tensor | None = None
        self.sum_b: torch.Tensor | None = None
        self.n_samples = 0

        self.h_a_list: list[torch.Tensor] = []
        self.h_b_list: list[torch.Tensor] = []

    def update(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        a = batch_a.detach().cpu().to(torch.float64)
        b = batch_b.detach().cpu().to(torch.float64)

        if self.store_raw:
            self.h_a_list.append(a.float())
            self.h_b_list.append(b.float())

        if self.at_b is None:
            self.at_b = a.T @ b
            self.sum_a = a.sum(dim=0)
            self.sum_b = b.sum(dim=0)
            if self.store_a_gram:
                self.at_a = a.T @ a
            if self.store_b_gram:
                self.bt_b = b.T @ b
        else:
            self.at_b += a.T @ b
            self.sum_a += a.sum(dim=0)
            self.sum_b += b.sum(dim=0)
            if self.store_a_gram and self.at_a is not None:
                self.at_a += a.T @ a
            if self.store_b_gram and self.bt_b is not None:
                self.bt_b += b.T @ b

        self.n_samples += int(a.shape[0])

    def rows(self, *, center: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.store_raw or not self.h_a_list:
            return None, None
        source = torch.cat(self.h_a_list, dim=0)
        target = torch.cat(self.h_b_list, dim=0)
        if center:
            source = source - source.mean(dim=0, keepdim=True)
            target = target - target.mean(dim=0, keepdim=True)
        return source, target

    def get_covariance(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.at_b is None:
            return None
        cov = self.at_b.clone()
        if center:
            assert self.sum_a is not None and self.sum_b is not None
            mu_a = self.sum_a / self.n_samples
            mu_b = self.sum_b / self.n_samples
            cov = cov - self.n_samples * torch.outer(mu_a, mu_b)
        if epsilon > 0 and cov.shape[0] == cov.shape[1]:
            cov = cov + epsilon * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        return cov

    def get_a_gram(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.at_a is None:
            return None
        gram = self.at_a.clone()
        if center:
            assert self.sum_a is not None
            mu_a = self.sum_a / self.n_samples
            gram = gram - self.n_samples * torch.outer(mu_a, mu_a)
        if epsilon > 0 and gram.shape[0] == gram.shape[1]:
            gram = gram + epsilon * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return gram

    def get_b_gram(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.bt_b is None:
            return None
        gram = self.bt_b.clone()
        if center:
            assert self.sum_b is not None
            mu_b = self.sum_b / self.n_samples
            gram = gram - self.n_samples * torch.outer(mu_b, mu_b)
        if epsilon > 0 and gram.shape[0] == gram.shape[1]:
            gram = gram + epsilon * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return gram


class _ActivationHook:
    def __init__(self, model: torch.nn.Module):
        self.model = _visual_module(model)
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self.handles.append(self.model.register_forward_hook(self._make_hook("")))
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if list(module.parameters(recurse=False)):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook_fn(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            inp = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else inputs
            if torch.is_tensor(inp):
                self.inputs[name] = inp.detach().cpu()
            try:
                out = _extract_output_tensor(output)
            except TypeError:
                out = None
            if out is not None and torch.is_tensor(out):
                self.outputs[name] = out.detach().cpu()

        return hook_fn

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def collect_activations(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    *,
    device: str | torch.device,
    seq_align: str,
    n_batches: int | None,
    seed: int = 0,
    batch_size: int | None = None,
    store_raw: bool = False,
    store_a_gram: bool = False,
    store_b_gram: bool = False,
) -> dict[str, ActivationStore]:
    registry: dict[str, ActivationStore] = {}
    source_hook = _ActivationHook(source_model)
    target_hook = _ActivationHook(target_model)
    dev = _resolve_device(device)

    try:
        iterator = _iter_random_dataset_batches(
            source_dataloader,
            target_dataloader,
            n_batches=n_batches,
            seed=seed,
            batch_size=batch_size,
        )
        if iterator is None:
            iterator = zip(source_dataloader, target_dataloader, strict=True)

        for idx, (source_batch, target_batch) in enumerate(iterator):
            if n_batches is not None and idx >= n_batches:
                break

            source_imgs = _extract_model_inputs(source_batch).to(dev)
            target_imgs = _extract_model_inputs(target_batch).to(dev)
            if source_imgs.shape[0] != target_imgs.shape[0]:
                raise ValueError(
                    "Theseus calibration expects aligned batch sizes. "
                    f"Got {source_imgs.shape[0]} and {target_imgs.shape[0]}."
                )

            _encode_image(source_model, source_imgs)
            _encode_image(target_model, target_imgs)

            common_inputs = set(source_hook.inputs.keys()) & set(target_hook.inputs.keys())
            common_outputs = set(source_hook.outputs.keys()) & set(target_hook.outputs.keys())

            for key in common_inputs:
                src_rows, tgt_rows = _align_features(source_hook.inputs[key], target_hook.inputs[key], mode=seq_align)
                reg_key = f"{key}.in"
                registry.setdefault(
                    reg_key,
                    ActivationStore(
                        store_raw=store_raw,
                        store_a_gram=store_a_gram,
                        store_b_gram=store_b_gram,
                    ),
                ).update(src_rows, tgt_rows)

            for key in common_outputs:
                src_rows, tgt_rows = _align_features(source_hook.outputs[key], target_hook.outputs[key], mode=seq_align)
                reg_key = f"{key}.out"
                registry.setdefault(
                    reg_key,
                    ActivationStore(
                        store_raw=store_raw,
                        store_a_gram=store_a_gram,
                        store_b_gram=store_b_gram,
                    ),
                ).update(src_rows, tgt_rows)

            source_hook.clear()
            target_hook.clear()
    finally:
        source_hook.remove()
        target_hook.remove()

    return registry


def _compute_procrustes_map_from_cov(cov: torch.Tensor) -> torch.Tensor:
    u, _, v_h = torch.linalg.svd(cov.double(), full_matrices=False)
    return (u @ v_h).float()


def _matrix_power_psd(matrix: torch.Tensor, *, power: float, eps: float) -> torch.Tensor:
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = torch.linalg.eigh(sym)
    powered = evals.clamp_min(float(eps)).pow(float(power))
    return (evecs * powered.unsqueeze(0)) @ evecs.T


def _partially_whiten_covariance(
    cov: torch.Tensor,
    *,
    a_gram: torch.Tensor,
    b_gram: torch.Tensor,
    power: float,
    eps: float,
) -> torch.Tensor:
    if power <= 0.0:
        return cov
    left = _matrix_power_psd(a_gram, power=-power, eps=eps)
    right = _matrix_power_psd(b_gram, power=-power, eps=eps)
    return left @ cov @ right


def _compute_alignment_map(
    store: ActivationStore,
    *,
    center: bool,
    whiten_power: float,
    whiten_eps: float,
) -> torch.Tensor | None:
    cov = store.get_covariance(center=center)
    if cov is None:
        return None
    if whiten_power > 0.0:
        a_gram = store.get_a_gram(center=center, epsilon=whiten_eps)
        b_gram = store.get_b_gram(center=center, epsilon=whiten_eps)
        if a_gram is not None and b_gram is not None:
            cov = _partially_whiten_covariance(
                cov,
                a_gram=a_gram,
                b_gram=b_gram,
                power=whiten_power,
                eps=whiten_eps,
            )
        else:
            logger.warning(
                "Theseus whitening requested but Gram statistics were unavailable; falling back to raw Procrustes."
            )
    return _compute_procrustes_map_from_cov(cov)


def _resolve_covariance_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in _ACTIVATION_COVARIANCE_MODES:
        return "activations"
    if key in _DATA_FREE_COVARIANCE_MODES:
        return "data_free"
    raise ValueError(
        "Theseus covariance_mode must be one of: activations, activation, data_free, data-free, weights, weight_space."
    )


def _compute_alignment_map_from_matrix_proxies(
    source_proxy: torch.Tensor,
    target_proxy: torch.Tensor,
    *,
    side: str,
    whiten_power: float,
    whiten_eps: float,
) -> torch.Tensor:
    source = source_proxy.detach().cpu().to(torch.float64)
    target = target_proxy.detach().cpu().to(torch.float64)

    if side == "input":
        a_gram = source.T @ source
        b_gram = target.T @ target
    elif side == "output":
        a_gram = source @ source.T
        b_gram = target @ target.T
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported alignment side '{side}'.")

    u_a, s_a, _ = torch.linalg.svd(a_gram, full_matrices=False)
    u_b, s_b, _ = torch.linalg.svd(b_gram, full_matrices=False)
    rank = min(int(u_a.shape[1]), int(u_b.shape[1]))
    basis_power = 0.5 - float(whiten_power)
    scale_a = s_a[:rank].clamp_min(float(whiten_eps)).pow(basis_power)
    scale_b = s_b[:rank].clamp_min(float(whiten_eps)).pow(basis_power)
    source_basis = u_a[:, :rank] * scale_a.unsqueeze(0)
    target_basis = u_b[:, :rank] * scale_b.unsqueeze(0)
    cov = source_basis @ target_basis.T

    return _compute_procrustes_map_from_cov(cov)


def _transport_weight(delta_weight: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor, *, key: str) -> torch.Tensor:
    if key == "proj":
        return (t_out.T @ delta_weight.T @ t_in).T
    return t_out.T @ delta_weight @ t_in


def _transport_bias(delta_vec: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
    return delta_vec @ t_out


def _param_to_module(visual_model: torch.nn.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for module_name, module in visual_model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            out[full_name] = module_name
    return out


def _iter_with_progress(iterable: Any, *, total: int, desc: str, enabled: bool) -> Any:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _iter_random_dataset_batches(
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    *,
    n_batches: int | None,
    seed: int,
    batch_size: int | None,
) -> Iterable[tuple[Any, Any]] | None:
    source_dataset = getattr(source_dataloader, "dataset", None)
    target_dataset = getattr(target_dataloader, "dataset", None)
    if source_dataset is None or target_dataset is None:
        return None

    try:
        n_source = int(len(source_dataset))
        n_target = int(len(target_dataset))
    except Exception:
        return None

    n_samples = min(n_source, n_target)
    if n_samples <= 0:
        return iter(())

    if batch_size is None:
        source_bs = getattr(source_dataloader, "batch_size", None)
        target_bs = getattr(target_dataloader, "batch_size", None)
        if source_bs is None or target_bs is None:
            return None
        batch_size = min(int(source_bs), int(target_bs))
    else:
        batch_size = int(batch_size)

    if batch_size <= 0:
        return None

    source_collate = getattr(source_dataloader, "collate_fn", None)
    target_collate = getattr(target_dataloader, "collate_fn", None)
    if not callable(source_collate) or not callable(target_collate):
        return None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_samples, generator=generator)

    if n_batches is not None:
        max_items = min(n_samples, int(n_batches) * batch_size)
        perm = perm[:max_items]

    def _iterator() -> Iterable[tuple[Any, Any]]:
        for start in range(0, int(perm.numel()), batch_size):
            indices = perm[start : start + batch_size].tolist()
            source_items = [source_dataset[i] for i in indices]
            target_items = [target_dataset[i] for i in indices]
            yield source_collate(source_items), target_collate(target_items)

    return _iterator()


@dataclass(frozen=True)
class _LayerTransform:
    kind: str
    t_in: torch.Tensor | None = None
    t_out: torch.Tensor | None = None


def _precompute_transforms(
    *,
    target_model: torch.nn.Module,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    activation_registry: Mapping[str, ActivationStore],
    center_acts: bool,
    whiten_power: float,
    whiten_eps: float,
    show_progress: bool,
    method_name: str,
) -> dict[str, _LayerTransform]:
    transforms_by_key: dict[str, _LayerTransform] = {}
    t_out_cache: dict[str, torch.Tensor] = {}
    visual_model = _visual_module(target_model)
    param_to_module = _param_to_module(visual_model)

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.prepare: compute transforms",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            print(f"Theseus align: skipping task vector key:{key} as it is not in target visual base.")
            continue

        if key in _ZERO_KEYS:
            transforms_by_key[key] = _LayerTransform(kind="zero")
            continue

        module_name = param_to_module.get(key, key.rsplit(".", 1)[0] if "." in key else "")
        if key == "proj":
            in_key = "ln_post.out"
            out_key = ".out"
        else:
            in_key = f"{module_name}.in"
            out_key = f"{module_name}.out"

        if delta_source.ndim == 2:
            in_store = activation_registry.get(in_key)
            out_store = activation_registry.get(out_key)
            if in_store is not None and out_store is not None:
                t_in = _compute_alignment_map(
                    in_store,
                    center=center_acts,
                    whiten_power=whiten_power,
                    whiten_eps=whiten_eps,
                )
                t_out = _compute_alignment_map(
                    out_store,
                    center=center_acts,
                    whiten_power=whiten_power,
                    whiten_eps=whiten_eps,
                )
                if t_in is not None and t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="weight", t_in=t_in, t_out=t_out)
                    continue
            transforms_by_key[key] = _LayerTransform(kind="weight")
            continue

        if delta_source.ndim == 1:
            if key.endswith(".bias"):
                weight_key = f"{key[: -len('.bias')]}.weight"
                weight_transform = transforms_by_key.get(weight_key)
                if weight_transform is not None and weight_transform.t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=weight_transform.t_out)
                    continue

            # Robustness fallback: covers uncommon ordering/edge cases where
            # the bias has no directly available weight transform yet.
            cached_t_out = t_out_cache.get(out_key)
            if cached_t_out is not None:
                transforms_by_key[key] = _LayerTransform(kind="bias", t_out=cached_t_out)
                continue

            out_store = activation_registry.get(out_key)
            if out_store is not None:
                t_out = _compute_alignment_map(
                    out_store,
                    center=center_acts,
                    whiten_power=whiten_power,
                    whiten_eps=whiten_eps,
                )
                if t_out is not None:
                    t_out_cache[out_key] = t_out
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=t_out)
                    continue
            transforms_by_key[key] = _LayerTransform(kind="bias")
            continue

        transforms_by_key[key] = _LayerTransform(kind="unsupported")

    return transforms_by_key


def _precompute_transforms_data_free(
    *,
    source_visual_base: Mapping[str, torch.Tensor],
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    whiten_power: float,
    whiten_eps: float,
    show_progress: bool,
    method_name: str,
) -> dict[str, _LayerTransform]:
    transforms_by_key: dict[str, _LayerTransform] = {}

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.prepare: compute data-free transforms",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            print(f"Theseus align: skipping task vector key:{key} as it is not in target visual base.")
            continue
        if key not in source_visual_base:
            transforms_by_key[key] = _LayerTransform(kind="unsupported")
            continue

        source_ref = source_visual_base[key]
        target_ref = target_visual_base[key]

        if key in _ZERO_KEYS:
            transforms_by_key[key] = _LayerTransform(kind="zero")
            continue

        if delta_source.ndim == 2 and source_ref.ndim == 2 and target_ref.ndim == 2:
            # CLIP stores `visual.proj` as [hidden, embed], so its axes are
            # flipped relative to nn.Linear([out, in]) weights.
            t_in = _compute_alignment_map_from_matrix_proxies(
                source_ref,
                target_ref,
                side="output" if key == "proj" else "input",
                whiten_power=whiten_power,
                whiten_eps=whiten_eps,
            )
            t_out = _compute_alignment_map_from_matrix_proxies(
                source_ref,
                target_ref,
                side="input" if key == "proj" else "output",
                whiten_power=whiten_power,
                whiten_eps=whiten_eps,
            )
            transforms_by_key[key] = _LayerTransform(kind="weight", t_in=t_in, t_out=t_out)
            continue

        if delta_source.ndim == 1 and source_ref.ndim == 1 and target_ref.ndim == 1:
            if key.endswith(".bias"):
                weight_key = f"{key[: -len('.bias')]}.weight"
                weight_transform = transforms_by_key.get(weight_key)
                if weight_transform is not None and weight_transform.t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=weight_transform.t_out)
                    continue

            t_out = _compute_alignment_map_from_matrix_proxies(
                source_ref.unsqueeze(0),
                target_ref.unsqueeze(0),
                side="input",
                whiten_power=whiten_power,
                whiten_eps=whiten_eps,
            )
            transforms_by_key[key] = _LayerTransform(kind="bias", t_out=t_out)
            continue

        transforms_by_key[key] = _LayerTransform(kind="unsupported")

    return transforms_by_key


def _apply_transforms_to_visual_delta(
    *,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    transforms_by_key: Mapping[str, _LayerTransform],
    show_progress: bool,
    method_name: str,
    sparsify_pct: float = 0.0,
    svd_rank_pct: float = 1.0,
) -> TensorDict:
    aligned: TensorDict = {}

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.apply: transport params",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            print(f"Theseus align: skipping task vector key:{key} as it is not in target visual base.")
            continue

        target_ref = target_visual_base[key]
        transported = torch.zeros_like(target_ref, dtype=torch.float32, device="cpu")

        transform = transforms_by_key.get(key)
        if transform is not None:
            if (
                transform.kind == "weight"
                and delta_source.ndim == 2
                and transform.t_in is not None
                and transform.t_out is not None
            ):
                try:
                    transported = _transport_weight(
                        delta_source.float().cpu(), transform.t_in, transform.t_out, key=key
                    )
                except RuntimeError as exc:
                    logger.warning("Theseus transport failed for %s: %s", key, exc)
            elif transform.kind == "bias" and delta_source.ndim == 1 and transform.t_out is not None:
                try:
                    transported = _transport_bias(delta_source.float().cpu(), transform.t_out)
                except ValueError as exc:
                    logger.warning("Theseus vector transport failed for %s: %s", key, exc)

        if transported.shape != target_ref.shape:
            logger.warning(
                "Theseus produced wrong shape for %s: got %s expected %s. Zeroing.",
                key,
                tuple(transported.shape),
                tuple(target_ref.shape),
            )
            transported = torch.zeros_like(target_ref, dtype=torch.float32, device="cpu")

        # Magnitude-based sparsification: zero out smallest-magnitude components
        if sparsify_pct > 0.0 and transported.numel() > 1 and transported.abs().sum() > 0:
            abs_vals = transported.abs().flatten()
            threshold = torch.quantile(abs_vals, sparsify_pct)
            transported = torch.where(transported.abs() >= threshold, transported, torch.zeros_like(transported))

        # SVD low-rank approximation: keep top-k singular components
        if 0.0 < svd_rank_pct < 1.0 and transported.ndim == 2 and min(transported.shape) > 1:
            try:
                U, S, Vh = torch.linalg.svd(transported.float(), full_matrices=False)
                k = max(1, int(len(S) * svd_rank_pct))
                transported = (U[:, :k] * S[:k]) @ Vh[:k, :]
                transported = transported.to(dtype=transported.dtype)
            except Exception:
                pass  # SVD may fail for non-finite values; skip

        aligned[key] = transported.to(dtype=target_ref.dtype, device=target_ref.device)

    return aligned


@dataclass(frozen=True)
class TheseusRebase:
    name: str = "theseus"

    def prepare(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        source_dataloader: Iterable[Any] | None = None,
        target_dataloader: Iterable[Any] | None = None,
        target_base: Mapping[str, torch.Tensor] | None = None,
        delta: Mapping[str, torch.Tensor] | None = None,
        device: str = "cuda",
        seq_align: str = "interpolate2d",
        center_acts: bool = False,
        covariance_mode: str = "activations",
        whiten_power: float = 0.0,
        whiten_eps: float = 1e-6,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        split_qkv = kwargs.pop("split_qkv", None)
        if split_qkv is not None:
            patch_qkv = bool(split_qkv)
        del kwargs
        # Config fallbacks num_batches -> n_batches
        if n_batches is None:
            n_batches = num_batches
        log_prefix = f"[{self.name}]"
        covariance_mode = _resolve_covariance_mode(covariance_mode)
        whiten_power = float(whiten_power)
        whiten_eps = float(whiten_eps)
        if not (0.0 <= whiten_power <= 0.5):
            raise ValueError("Theseus whiten_power must be in [0, 0.5].")
        if whiten_eps <= 0.0:
            raise ValueError("Theseus whiten_eps must be > 0.")

        if verbose:
            print(
                f"{log_prefix} prepare: start "
                f"(seq_align={seq_align}, center_acts={bool(center_acts)}, "
                f"whiten_power={whiten_power}, covariance_mode={covariance_mode}, "
                f"n_batches={n_batches}, seed={int(seed)})"
            )

        patched_source = 0
        patched_target = 0
        if patch_qkv:
            if verbose:
                print(f"{log_prefix} prepare: patching fused qkv blocks if needed")
            patched_source = _split_fused_qkv_if_needed(source_model)
            patched_target = _split_fused_qkv_if_needed(target_model)
            if patched_source > 0 or patched_target > 0:
                logger.info(
                    "%s prepare: split fused qkv attention blocks (source=%d, target=%d)",
                    self.name,
                    patched_source,
                    patched_target,
                )
        elif verbose:
            print(f"{log_prefix} prepare: patch_qkv disabled")

        activation_registry: dict[str, ActivationStore] = {}
        transforms_by_key: dict[str, _LayerTransform] = {}
        split_fused_qkv = bool(patch_qkv and (patched_source > 0 or patched_target > 0))
        unpatched_source = 0
        unpatched_target = 0
        try:
            if covariance_mode == "activations":
                if source_dataloader is None or target_dataloader is None:
                    raise ValueError(
                        "Theseus activation covariance mode requires both source_dataloader and target_dataloader."
                    )
                if verbose:
                    print(f"{log_prefix} prepare: collecting activations")

                activation_registry = collect_activations(
                    source_model,
                    target_model,
                    source_dataloader,
                    target_dataloader,
                    device=device,
                    seq_align=seq_align,
                    n_batches=n_batches,
                    seed=int(seed),
                    batch_size=batch_size,
                    store_a_gram=whiten_power > 0.0,
                    store_b_gram=whiten_power > 0.0,
                )
                if verbose:
                    print(f"{log_prefix} prepare: collected activation entries = {len(activation_registry)}")

            elif verbose:
                print(f"{log_prefix} prepare: skipping activation collection (data-free covariance mode)")

            if target_base is not None and delta is not None:
                if verbose:
                    print(f"{log_prefix} prepare: precomputing per-layer transforms")
                visual_key_map = _visual_delta_keys(delta)
                source_visual_base = _visual_state_dict(source_model.state_dict())
                target_visual_base = _visual_state_dict(target_base)
                visual_delta = {
                    stripped_key: delta[original_key]
                    for stripped_key, original_key in visual_key_map.items()
                    if stripped_key in target_visual_base
                }

                if split_fused_qkv:
                    source_visual_base = _split_fused_qkv_state(source_visual_base)
                    target_visual_base = _split_fused_qkv_state(target_visual_base)
                    visual_delta = _split_fused_qkv_state(visual_delta)

                if covariance_mode == "activations":
                    transforms_by_key = _precompute_transforms(
                        target_model=target_model,
                        target_visual_base=target_visual_base,
                        visual_delta=visual_delta,
                        activation_registry=activation_registry,
                        center_acts=bool(center_acts),
                        whiten_power=whiten_power,
                        whiten_eps=whiten_eps,
                        show_progress=bool(show_progress),
                        method_name=self.name,
                    )
                else:
                    transforms_by_key = _precompute_transforms_data_free(
                        source_visual_base=source_visual_base,
                        target_visual_base=target_visual_base,
                        visual_delta=visual_delta,
                        whiten_power=whiten_power,
                        whiten_eps=whiten_eps,
                        show_progress=bool(show_progress),
                        method_name=self.name,
                    )
                if verbose:
                    print(f"{log_prefix} prepare: computed transforms = {len(transforms_by_key)}")
            elif verbose:
                print(f"{log_prefix} prepare: target_base/delta missing, skipping transform precompute")
        finally:
            if patch_qkv and (patched_source > 0 or patched_target > 0):
                try:
                    unpatched_source = int(merge_openclip_vit_attn(_visual_module(source_model)))
                    unpatched_target = int(merge_openclip_vit_attn(_visual_module(target_model)))
                    if verbose:
                        print(
                            f"{log_prefix} prepare: recomposed fused qkv blocks "
                            f"(source={unpatched_source}, target={unpatched_target})"
                        )
                except Exception as exc:
                    logger.warning("%s prepare: failed to recompose patched attention blocks: %s", self.name, exc)

        if verbose:
            print(f"{log_prefix} prepare: done")

        return {
            "activation_registry": activation_registry,
            "transforms_by_key": transforms_by_key,
            "covariance_mode": covariance_mode,
            "split_fused_qkv": split_fused_qkv,
            "n_batches": n_batches,
            "patched_source_blocks": patched_source,
            "patched_target_blocks": patched_target,
            "unpatched_source_blocks": unpatched_source,
            "unpatched_target_blocks": unpatched_target,
        }

    def apply(
        self,
        prepared: Mapping[str, Any],
        *,
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> TensorDict:
        _sparsify_pct = float(kwargs.pop("sparsify_pct", 0.0))
        _svd_rank_pct = float(kwargs.pop("svd_rank_pct", 1.0))
        del kwargs
        log_prefix = f"[{self.name}]"

        if verbose:
            print(f"{log_prefix} apply: start")

        transforms_by_key = prepared.get("transforms_by_key", None)
        if transforms_by_key is None:
            raise ValueError("Theseus prepared payload is missing 'transforms_by_key'.")

        visual_key_map = _visual_delta_keys(delta)
        target_visual_base = _visual_state_dict(target_base)

        visual_delta = {
            stripped_key: delta[original_key]
            for stripped_key, original_key in visual_key_map.items()
            if stripped_key in target_visual_base
        }

        split_fused_qkv = bool(prepared.get("split_fused_qkv", False))
        if split_fused_qkv:
            target_visual_base_work = _split_fused_qkv_state(target_visual_base)
            visual_delta_work = _split_fused_qkv_state(visual_delta)
        else:
            target_visual_base_work = target_visual_base
            visual_delta_work = visual_delta

        if strict and not visual_delta_work:
            raise ValueError("Theseus did not find any visual delta keys to transport.")

        aligned_visual = _apply_transforms_to_visual_delta(
            target_visual_base=target_visual_base_work,
            visual_delta=visual_delta_work,
            transforms_by_key=transforms_by_key,
            show_progress=bool(show_progress),
            method_name=self.name,
            sparsify_pct=_sparsify_pct,
            svd_rank_pct=_svd_rank_pct,
        )

        if split_fused_qkv:
            aligned_visual = _merge_split_qkv_state(aligned_visual, reference=target_visual_base)

        out: TensorDict = {}
        processed: set[str] = set()

        for stripped_key, original_key in visual_key_map.items():
            if original_key not in target_base:
                continue
            if stripped_key in aligned_visual:
                out[original_key] = aligned_visual[stripped_key].to(
                    dtype=target_base[original_key].dtype,
                    device=target_base[original_key].device,
                )
            else:
                out[original_key] = torch.zeros_like(target_base[original_key], device=target_base[original_key].device)
            processed.add(original_key)

        for key in delta:
            if key in processed or key not in target_base:
                continue
            out[key] = torch.zeros_like(target_base[key], device=target_base[key].device)

        if strict:
            missing = sorted(set(delta.keys()) - set(out.keys()))
            if missing:
                raise KeyError(f"Theseus did not transport all delta keys. Example: {missing[:10]}")

        if verbose:
            print(f"{log_prefix} apply: done (transported_keys={len(out)})")

        return out

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        source_model: torch.nn.Module | None = None,
        target_model: torch.nn.Module | None = None,
        source_dataloader: Iterable[Any] | None = None,
        target_dataloader: Iterable[Any] | None = None,
        device: str = "cuda",
        seq_align: str = "interpolate2d",
        center_acts: bool = False,
        covariance_mode: str = "activations",
        whiten_power: float = 0.0,
        whiten_eps: float = 1e-6,
        prepared: Mapping[str, Any] | None = None,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> TensorDict:
        del source_base
        log_prefix = f"[{self.name}]"

        if n_batches is None:
            n_batches = num_batches

        prepared_payload: Mapping[str, Any]
        if prepared is None:
            if source_model is None or target_model is None:
                raise ValueError("Theseus transport requires both source_model and target_model.")
            if _resolve_covariance_mode(covariance_mode) == "activations" and (
                source_dataloader is None or target_dataloader is None
            ):
                raise ValueError("Theseus transport requires both source_dataloader and target_dataloader.")

            prepared_payload = self.prepare(
                source_model=source_model,
                target_model=target_model,
                source_dataloader=source_dataloader,
                target_dataloader=target_dataloader,
                target_base=target_base,
                delta=delta,
                device=device,
                seq_align=seq_align,
                center_acts=bool(center_acts),
                covariance_mode=covariance_mode,
                whiten_power=float(whiten_power),
                whiten_eps=float(whiten_eps),
                n_batches=n_batches,
                seed=int(seed),
                batch_size=batch_size,
                patch_qkv=patch_qkv,
                verbose=bool(verbose),
                show_progress=bool(show_progress),
                **kwargs,
            )
        else:
            prepared_payload = prepared
            if verbose:
                print(f"{log_prefix} transport: using provided prepared payload")

        return self.apply(
            prepared_payload,
            target_base=target_base,
            delta=delta,
            strict=bool(strict),
            verbose=bool(verbose),
            show_progress=bool(show_progress),
            **kwargs,
        )


register(TheseusRebase())
