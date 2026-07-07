"""
Estimators for TracIn and Step-Decomposed Influence (SDI).

Design goals:
- **Explicit parameter-set selection**: the caller provides the exact submodule(s)
  to analyze. No silent inclusion/exclusion based on parameter names.
- **Strict Opacus-style coverage**: if a selected module contains trainable
  parameters without Opacus grad-sampler coverage (and we can't handle them),
  we raise an error. We do not fall back to vmap/functorch.
- **Step-resolved features**: for looped/weight-tied models, parameters are
  used multiple times in one forward pass. We capture one feature per use
  (per example) and stack them into a (B, steps, ...) tensor.

Two main classes are exposed:
- `ProjectedTracInSDI`: hook-based TensorSketch/CountSketch estimator
- `FullGradientTracInSDI`: full per-sample-gradient baseline (exact, expensive)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
)

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import opt_einsum

    contract = opt_einsum.contract
except Exception:  # pragma: no cover
    contract = torch.einsum

from opacus.grad_sample.grad_sample_module import GradSampleModule

from .runner import (
    CheckpointSpec,
    InfluenceOutputs,
    default_eta_from_checkpoint,
    default_load_checkpoint,
    default_load_model_state_dict,
    resolve_checkpoint_weights,
)
from .sketch import (
    CountSketchSpec,
    TensorSketchSpec,
    count_sketch_from_grad_sample,
    count_sketch_vector,
    make_count_sketch,
    make_tensor_sketch,
    tensor_sketch_embedding_weight,
    tensor_sketch_linear_weight,
)

LossFn = Callable[[nn.Module, Any], torch.Tensor]


@dataclass(frozen=True)
class FeatureSet:
    """
    Stepwise per-example features for a fixed parameter set.

    `phi` maps parameter name -> tensor of shape:
    - Projected estimator: (N, steps, m)
    - Full estimator: (N, steps, *param_shape)
    """

    phi: Dict[str, torch.Tensor]
    steps: int

    @property
    def n(self) -> int:
        first = next(iter(self.phi.values()))
        return int(first.shape[0])


def _as_module_list(modules: Union[nn.Module, Sequence[nn.Module]]) -> List[nn.Module]:
    if isinstance(modules, nn.Module):
        return [modules]
    return list(modules)


def _unique_modules(modules: Iterable[nn.Module]) -> List[nn.Module]:
    seen: set[int] = set()
    out: list[nn.Module] = []
    for m in modules:
        mid = id(m)
        if mid in seen:
            continue
        seen.add(mid)
        out.append(m)
    return out


def _build_param_to_name(model: nn.Module) -> Dict[nn.Parameter, str]:
    mapping: Dict[nn.Parameter, str] = {}
    for name, p in model.named_parameters():
        mapping.setdefault(p, name)
    return mapping


def _gather_target_params(
    target_modules: Sequence[nn.Module],
) -> List[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for tm in target_modules:
        for p in tm.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(p)
    return params


def _module_direct_trainable_params(
    module: nn.Module, *, target_params_set: set[nn.Parameter]
) -> List[nn.Parameter]:
    out: list[nn.Parameter] = []
    for p in module.parameters(recurse=False):
        if (not p.requires_grad) or (p not in target_params_set):
            continue
        out.append(p)
    return out


def _assert_supported_loss_reduction(loss_reduction: str) -> None:
    if loss_reduction not in ("sum", "mean"):
        raise ValueError(
            f"loss_reduction must be 'sum' or 'mean', got {loss_reduction!r}"
        )


def _reduce_loss(loss: torch.Tensor, *, loss_reduction: str) -> torch.Tensor:
    """
    Convert `loss` into a scalar suitable for backward().
    If loss is already scalar, return it unchanged.
    """
    if loss.dim() == 0:
        return loss
    if loss.dim() != 1:
        raise ValueError(
            f"Expected loss to be scalar or shape (B,), got {tuple(loss.shape)}"
        )
    if loss_reduction == "sum":
        return loss.sum()
    if loss_reduction == "mean":
        return loss.mean()
    raise AssertionError("unreachable")


def _scale_backprops(
    backprops: torch.Tensor, *, loss_reduction: str, batch_first: bool
) -> torch.Tensor:
    """
    Undo batch reduction so Opacus grad samplers match per-example losses.

    Mirrors Opacus' `rearrange_grad_samples` scaling for the common batch_first=True case.
    """
    _assert_supported_loss_reduction(loss_reduction)
    if loss_reduction == "sum":
        return backprops
    # mean: multiply by batch size
    n = backprops.shape[0] if batch_first else backprops.shape[1]
    return backprops * n


def _flatten_phi_for_dot(phi: torch.Tensor) -> torch.Tensor:
    """
    Flatten per-step feature tensors to (N, steps, D).
    - Already-projected: (N, steps, m) -> unchanged
    - Full gradients:    (N, steps, *shape) -> (N, steps, prod(shape))
    """
    if phi.dim() < 3:
        raise ValueError(f"Expected phi dim>=3 (N,steps,...), got {tuple(phi.shape)}")
    return phi.flatten(2)


def _iter_chunks(n: int, chunk_size: Optional[int]) -> Iterable[Tuple[int, int]]:
    if n <= 0:
        return []
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        return [(0, n)]
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def compute_query_sdi(
    *,
    train: FeatureSet,
    query: FeatureSet,
    train_chunk_size: Optional[int] = None,
    query_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute query-step SDI tensor:
      SDI[i, j, t] = sum_p < grad_train[i,p], phi_query[j,t,p] >
    where grad_train = sum_s phi_train[:, s, :].

    Returns: (N_train, N_query, steps_query)
    """
    if train.phi.keys() != query.phi.keys():
        missing = sorted(set(train.phi.keys()) - set(query.phi.keys()))
        extra = sorted(set(query.phi.keys()) - set(train.phi.keys()))
        raise ValueError(
            "Parameter key-set mismatch between train and query features.\n"
            f"- Missing in query ({len(missing)}): {missing}\n"
            f"- Extra in query ({len(extra)}): {extra}"
        )
    n_train = train.n
    n_query = query.n
    steps_q = query.steps
    device = next(iter(query.phi.values())).device
    out = torch.zeros(n_train, n_query, steps_q, device=device, dtype=torch.float32)
    for name in train.phi.keys():
        phi_tr = _flatten_phi_for_dot(train.phi[name]).to(
            dtype=torch.float32
        )  # (N_train, steps_tr, D)
        phi_q = _flatten_phi_for_dot(query.phi[name]).to(
            dtype=torch.float32
        )  # (N_query, steps_q, D)
        g_tr = phi_tr.sum(dim=1)  # (N_train, D)
        for tr0, tr1 in _iter_chunks(n_train, train_chunk_size):
            g_tr_chunk = g_tr[tr0:tr1]
            for q0, q1 in _iter_chunks(n_query, query_chunk_size):
                phi_q_chunk = phi_q[q0:q1]
                out[tr0:tr1, q0:q1] += contract("id,jtd->ijt", g_tr_chunk, phi_q_chunk)
    return out


def compute_step_matrix(
    *,
    train: FeatureSet,
    query: FeatureSet,
    train_chunk_size: Optional[int] = None,
    query_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the full step×step SDI matrix:
      M[i, j, s, t] = sum_p < phi_train[i,s,p], phi_query[j,t,p] >

    Returns: (N_train, N_query, steps_train, steps_query)
    """
    if train.phi.keys() != query.phi.keys():
        missing = sorted(set(train.phi.keys()) - set(query.phi.keys()))
        extra = sorted(set(query.phi.keys()) - set(train.phi.keys()))
        raise ValueError(
            "Parameter key-set mismatch between train and query features.\n"
            f"- Missing in query ({len(missing)}): {missing}\n"
            f"- Extra in query ({len(extra)}): {extra}"
        )
    n_train = train.n
    n_query = query.n
    steps_tr = train.steps
    steps_q = query.steps
    device = next(iter(query.phi.values())).device
    out = torch.zeros(
        n_train, n_query, steps_tr, steps_q, device=device, dtype=torch.float32
    )
    for name in train.phi.keys():
        phi_tr = _flatten_phi_for_dot(train.phi[name]).to(
            dtype=torch.float32
        )  # (N_train, steps_tr, D)
        phi_q = _flatten_phi_for_dot(query.phi[name]).to(
            dtype=torch.float32
        )  # (N_query, steps_q, D)
        for tr0, tr1 in _iter_chunks(n_train, train_chunk_size):
            phi_tr_chunk = phi_tr[tr0:tr1]
            for q0, q1 in _iter_chunks(n_query, query_chunk_size):
                phi_q_chunk = phi_q[q0:q1]
                out[tr0:tr1, q0:q1] += contract(
                    "isd,jtd->ijst", phi_tr_chunk, phi_q_chunk
                )
    return out


def compute_tracin(
    *,
    train: FeatureSet,
    query: FeatureSet,
    train_chunk_size: Optional[int] = None,
    query_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute scalar TracIn without materializing SDI:
      TracIn[i, j] = sum_p < sum_s phi_train[i,s,p], sum_t phi_query[j,t,p] >
    """
    if train.phi.keys() != query.phi.keys():
        missing = sorted(set(train.phi.keys()) - set(query.phi.keys()))
        extra = sorted(set(query.phi.keys()) - set(train.phi.keys()))
        raise ValueError(
            "Parameter key-set mismatch between train and query features.\n"
            f"- Missing in query ({len(missing)}): {missing}\n"
            f"- Extra in query ({len(extra)}): {extra}"
        )
    n_train = train.n
    n_query = query.n
    device = next(iter(query.phi.values())).device
    out = torch.zeros(n_train, n_query, device=device, dtype=torch.float32)
    for name in train.phi.keys():
        phi_tr = _flatten_phi_for_dot(train.phi[name]).to(
            dtype=torch.float32
        )  # (N_train, steps_tr, D)
        phi_q = _flatten_phi_for_dot(query.phi[name]).to(
            dtype=torch.float32
        )  # (N_query, steps_q, D)
        g_tr = phi_tr.sum(dim=1)  # (N_train, D)
        g_q = phi_q.sum(dim=1)  # (N_query, D)
        for tr0, tr1 in _iter_chunks(n_train, train_chunk_size):
            g_tr_chunk = g_tr[tr0:tr1]
            for q0, q1 in _iter_chunks(n_query, query_chunk_size):
                g_q_chunk = g_q[q0:q1]
                out[tr0:tr1, q0:q1] += g_tr_chunk @ g_q_chunk.T
    return out


class _BaseHookFeaturizer:
    """
    Common hook logic for stepwise per-parameter features.

    Subclasses implement `_handle_module_backward(...)` to append one per-use
    feature tensor per parameter into `self._phi_lists[p]`.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        target_modules: Sequence[nn.Module],
        loss_reduction: str,
        batch_first: bool,
        expected_steps: Optional[int],
        allow_mixed_steps: bool,
        strict: bool,
    ) -> None:
        _assert_supported_loss_reduction(loss_reduction)
        self.model = model
        self.loss_reduction = loss_reduction
        self.batch_first = bool(batch_first)
        self.expected_steps = expected_steps
        self.allow_mixed_steps = bool(allow_mixed_steps)
        self.strict = bool(strict)

        self._param_to_name = _build_param_to_name(model)
        self._target_modules = _unique_modules(_as_module_list(target_modules))
        self._target_params = _gather_target_params(self._target_modules)
        self._target_param_set = set(self._target_params)

        if len(self._target_params) == 0:
            raise ValueError("No trainable parameters found in target_modules.")

        # Ensure all target parameters live on a single device (for sketch tensor placement).
        devices = {p.device for p in self._target_params}
        if len(devices) != 1:
            raise ValueError(
                f"Target parameters span multiple devices: {sorted(map(str, devices))}"
            )
        self._device = next(iter(devices))

        # Hook state
        self._hooks_enabled = False
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._act_stack: Dict[nn.Module, List[List[torch.Tensor]]] = {}
        self._phi_lists: Dict[nn.Parameter, List[torch.Tensor]] = {
            p: [] for p in self._target_params
        }
        self._param_owner: Dict[nn.Parameter, nn.Module] = {}
        self._module_call_indices: Dict[nn.Module, List[int]] = {}
        self._call_counter: int = 0

        # Precompute the list of modules we need hooks on: those with direct target params.
        self._hook_modules: list[nn.Module] = []
        for _name, m in model.named_modules():
            direct = _module_direct_trainable_params(
                m, target_params_set=self._target_param_set
            )
            if len(direct) == 0:
                continue
            for p in direct:
                if p in self._param_owner and self._param_owner[p] is not m:
                    raise RuntimeError(
                        "A parameter appears to be owned by multiple modules, which is not supported. "
                        f"Param={self._param_to_name.get(p, '<unnamed>')} owners: "
                        f"{type(self._param_owner[p])} and {type(m)}"
                    )
                self._param_owner[p] = m
            self._hook_modules.append(m)

        if len(self._hook_modules) == 0:
            raise RuntimeError(
                "Internal error: found target params but no owning modules to hook."
            )

        self._validate_supported_modules()
        self._register_hooks()

    def close(self) -> None:
        """Remove hooks and clear internal state."""
        self._hooks_enabled = False
        while self._handles:
            h = self._handles.pop()
            h.remove()
        self._act_stack.clear()
        for p in self._phi_lists:
            self._phi_lists[p].clear()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    # -----------------------------
    # Hook plumbing
    # -----------------------------

    def _register_hooks(self) -> None:
        for m in self._hook_modules:
            self._handles.append(m.register_forward_hook(self._forward_hook))
            self._handles.append(m.register_full_backward_hook(self._backward_hook))

    def _forward_hook(
        self, module: nn.Module, forward_input: Tuple[Any, ...], _forward_output: Any
    ) -> None:
        if (not self._hooks_enabled) or (not torch.is_grad_enabled()):
            return

        # Only keep tensor inputs (Opacus grad samplers expect tensors).
        acts: list[torch.Tensor] = []
        for t in forward_input:
            if torch.is_tensor(t):
                acts.append(t.detach())
        if len(acts) == 0:
            return
        self._act_stack.setdefault(module, []).append(acts)
        self._module_call_indices.setdefault(module, []).append(self._call_counter)
        self._call_counter += 1

    def _backward_hook(
        self,
        module: nn.Module,
        _grad_input: Any,
        grad_output: Any,
    ) -> None:
        if not self._hooks_enabled:
            return
        if not grad_output or (not torch.is_tensor(grad_output[0])):
            return
        if module not in self._act_stack or len(self._act_stack[module]) == 0:
            raise RuntimeError(
                f"Missing cached activations for module {type(module)}. "
                "This usually indicates that the forward hook didn't run (e.g., no grad enabled)."
            )
        backprops = grad_output[0].detach()
        activations = self._act_stack[module].pop()
        backprops = _scale_backprops(
            backprops, loss_reduction=self.loss_reduction, batch_first=self.batch_first
        )

        with torch.no_grad():
            self._handle_module_backward(
                module=module, activations=activations, backprops=backprops
            )

    def _validate_supported_modules(self) -> None:
        """Subclass-specific validation."""
        raise NotImplementedError

    def _handle_module_backward(
        self,
        *,
        module: nn.Module,
        activations: List[torch.Tensor],
        backprops: torch.Tensor,
    ) -> None:
        """Subclass should append one per-use feature tensor per target param."""
        raise NotImplementedError

    # -----------------------------
    # Public API (batch featurization)
    # -----------------------------

    def reset_state(self) -> None:
        self._act_stack.clear()
        self._module_call_indices.clear()
        self._call_counter = 0
        for p in self._phi_lists:
            self._phi_lists[p].clear()

    def featurize_batch(self, *, loss: torch.Tensor) -> FeatureSet:
        """
        Given a loss tensor computed from a forward pass, run backward and return a FeatureSet.

        Assumes the forward pass was already run with the hooks enabled.
        """
        self.reset_state()
        # The forward hooks populate activation stacks; we need them before backward.
        # So we must enable hooks, run forward in user code, then call this method.
        raise NotImplementedError("Use compute_batch(model, batch, loss_fn) instead.")

    def compute_batch(self, *, batch: Any, loss_fn: LossFn) -> FeatureSet:
        """
        Compute a FeatureSet for a single batch.

        `loss_fn(model, batch)` must return either:
        - a scalar loss, in which case `loss_reduction` must match how that scalar
          was reduced over the batch, or
        - a vector of per-example losses of shape (B,), in which case we reduce it
          using `loss_reduction` before backward.
        """
        self.reset_state()
        self.model.zero_grad(set_to_none=True)
        self._hooks_enabled = True
        try:
            loss = loss_fn(self.model, batch)
            loss_scalar = _reduce_loss(loss, loss_reduction=self.loss_reduction)
            if loss_scalar.dim() != 0:
                raise RuntimeError("Internal error: loss_scalar is not scalar")
            loss_scalar.backward()
            return self._finalize_features()
        finally:
            self._hooks_enabled = False

    def _finalize_features(self) -> FeatureSet:
        # Ensure all target params were observed.
        counts: Dict[str, int] = {}
        for p, lst in self._phi_lists.items():
            name = self._param_to_name.get(p, "<unnamed>")
            counts[name] = len(lst)

        zero = sorted([n for n, c in counts.items() if c == 0])
        if zero:
            raise RuntimeError(
                "Some selected parameters were never observed during forward/backward.\n"
                f"First few: {zero[:20]}\n"
                "This usually means the parameter-set includes modules that were not executed, "
                "or the model ran under no_grad()."
            )

        step_counts = sorted(set(counts.values()))

        # Common (strict) case: all parameters used the same number of times.
        if len(step_counts) == 1:
            steps_main = step_counts[0]
            if self.expected_steps is not None and steps_main != int(
                self.expected_steps
            ):
                raise RuntimeError(
                    f"Observed steps={steps_main} does not match expected_steps={self.expected_steps}. "
                    "If you are analyzing longer sequences/loops than training, set expected_steps=None "
                    "or update it accordingly."
                )
            phi_out: Dict[str, torch.Tensor] = {}
            for p, lst in self._phi_lists.items():
                name = self._param_to_name.get(p)
                if name is None:
                    raise RuntimeError(
                        "Encountered parameter without name mapping; model may have been mutated."
                    )
                phi_out[name] = torch.stack(list(reversed(lst)), dim=1)
            return FeatureSet(phi=phi_out, steps=steps_main)

        if not self.allow_mixed_steps:
            items = sorted(counts.items(), key=lambda kv: kv[1])
            smallest = items[:10]
            largest = items[-10:]
            raise RuntimeError(
                "Inconsistent step counts across selected parameters. "
                "SDI requires that all selected parameters are used the same number of times "
                "in the analyzed forward pass.\n"
                f"Unique counts: {step_counts}\n"
                f"Smallest counts: {smallest}\n"
                f"Largest counts: {largest}\n"
                "Tip: pass only the looped/weight-tied body modules (and any per-step injection modules), "
                "or set allow_mixed_steps=True to bin single-use params into a prelude/readout slot."
            )

        # Mixed step counts: support the common case "looped params" (count=max) plus
        # "single-use params" (count=1) binned into prelude/readout.
        steps_main = max(step_counts)
        if any((c not in (1, steps_main)) for c in step_counts):
            raise RuntimeError(
                "allow_mixed_steps=True supports only counts in {1, max_count}. "
                f"Observed unique counts: {step_counts}"
            )
        if self.expected_steps is not None and steps_main != int(self.expected_steps):
            raise RuntimeError(
                f"Observed loop steps={steps_main} does not match expected_steps={self.expected_steps}."
            )

        # Determine loop region in the module-call timeline.
        loop_call_idxs: list[int] = []
        for p, lst in self._phi_lists.items():
            if len(lst) != steps_main:
                continue
            owner = self._param_owner.get(p)
            if owner is None:
                continue
            loop_call_idxs.extend(self._module_call_indices.get(owner, []))
        if not loop_call_idxs:
            raise RuntimeError(
                "allow_mixed_steps=True but could not identify any looped parameters (count=max)."
            )
        min_loop, max_loop = min(loop_call_idxs), max(loop_call_idxs)

        single_assign: Dict[nn.Parameter, str] = {}
        for p, lst in self._phi_lists.items():
            if len(lst) != 1:
                continue
            owner = self._param_owner.get(p)
            if owner is None:
                raise RuntimeError("Internal error: missing param owner mapping")
            call_idxs = self._module_call_indices.get(owner, [])
            if len(call_idxs) != 1:
                raise RuntimeError(
                    "Internal error: single-use param but module call count != 1"
                )
            ci = call_idxs[0]
            if ci < min_loop:
                single_assign[p] = "prelude"
            elif ci > max_loop:
                single_assign[p] = "readout"
            else:
                raise RuntimeError(
                    "Single-use parameter executed within the loop region; cannot bin into prelude/readout.\n"
                    f"Param={self._param_to_name.get(p, '<unnamed>')} call_idx={ci} loop_range=[{min_loop},{max_loop}]"
                )

        need_prelude = any(v == "prelude" for v in single_assign.values())
        need_readout = any(v == "readout" for v in single_assign.values())
        prelude_offset = 1 if need_prelude else 0
        steps_out = steps_main + (1 if need_prelude else 0) + (1 if need_readout else 0)
        readout_index = prelude_offset + steps_main  # valid only if need_readout

        phi_out: Dict[str, torch.Tensor] = {}
        for p, lst in self._phi_lists.items():
            name = self._param_to_name.get(p)
            if name is None:
                raise RuntimeError(
                    "Encountered parameter without name mapping; model may have been mutated."
                )
            phi_raw = torch.stack(list(reversed(lst)), dim=1)  # (B, count, ...)
            # Pad into common (B, steps_out, ...)
            pad_shape = (phi_raw.shape[0], steps_out) + phi_raw.shape[2:]
            phi_pad = torch.zeros(pad_shape, device=phi_raw.device, dtype=phi_raw.dtype)
            if phi_raw.shape[1] == steps_main:
                phi_pad[:, prelude_offset : prelude_offset + steps_main] = phi_raw
            elif phi_raw.shape[1] == 1:
                slot = single_assign.get(p)
                if slot == "prelude":
                    if not need_prelude:
                        raise RuntimeError("Internal error: missing prelude bin")
                    phi_pad[:, 0:1] = phi_raw
                elif slot == "readout":
                    if not need_readout:
                        raise RuntimeError("Internal error: missing readout bin")
                    phi_pad[:, readout_index : readout_index + 1] = phi_raw
                else:
                    raise RuntimeError(
                        "Internal error: single-use param missing assignment"
                    )
            else:
                raise RuntimeError(
                    "Internal error: unexpected count under allow_mixed_steps"
                )
            phi_out[name] = phi_pad
        return FeatureSet(phi=phi_out, steps=steps_out)


class _ProjectedHookFeaturizer(_BaseHookFeaturizer):
    """
    Internal hook-based featurizer that sketches per-sample gradients.

    Produces stepwise projected features without materializing full matrix
    gradients by using TensorSketch (2D params) and CountSketch (1D params).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        target_modules: Sequence[nn.Module],
        projection_size: int,
        seed: int,
        loss_reduction: str,
        batch_first: bool,
        expected_steps: Optional[int],
        allow_mixed_steps: bool,
        strict: bool,
    ) -> None:
        self.projection_size = int(projection_size)
        if self.projection_size <= 0:
            raise ValueError("projection_size must be > 0")
        self.seed = int(seed)
        self._rng: Optional[torch.Generator] = None
        self._count_specs: Dict[nn.Parameter, CountSketchSpec] = {}
        self._tensor_specs: Dict[nn.Parameter, TensorSketchSpec] = {}
        super().__init__(
            model=model,
            target_modules=target_modules,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
            expected_steps=expected_steps,
            allow_mixed_steps=allow_mixed_steps,
            strict=strict,
        )

    def _validate_supported_modules(self) -> None:
        # Initialize RNG on the target device for deterministic hash generation.
        self._rng = torch.Generator(device=self._device)
        self._rng.manual_seed(self.seed)

        # For every direct trainable param in hook modules, precompute sketch specs and validate.
        for m in self._hook_modules:
            direct = _module_direct_trainable_params(
                m, target_params_set=self._target_param_set
            )
            if not direct:
                continue

            # Disallow buffers in strict mode (mirrors Opacus validate rationale).
            if self.strict and len(list(m.buffers(recurse=False))) > 0:
                raise NotImplementedError(
                    f"Selected module {type(m)} has buffers; per-sample gradients are not supported in strict mode."
                )

            mtype = type(m)
            has_gs = mtype in GradSampleModule.GRAD_SAMPLERS

            for p in direct:
                pname = self._param_to_name.get(p, "<unnamed>")
                if p.dim() == 2:
                    if not isinstance(m, (nn.Linear, nn.Embedding)):
                        raise NotImplementedError(
                            "Projected SDI only supports 2D parameters for nn.Linear and nn.Embedding "
                            f"(saw {pname} owned by {mtype})."
                        )
                    d_out, d_in = p.shape
                    self._tensor_specs[p] = make_tensor_sketch(
                        d_out=int(d_out),
                        d_in=int(d_in),
                        m=self.projection_size,
                        device=p.device,
                        generator=self._rng,
                    )
                elif p.dim() == 1:
                    # For 1D params, we will CountSketch their per-sample gradients.
                    # We require a grad sampler unless this is Linear.bias (handled analytically).
                    if isinstance(m, nn.Linear) and getattr(m, "bias", None) is p:
                        pass
                    else:
                        if not has_gs:
                            raise NotImplementedError(
                                f"No Opacus grad sampler registered for module type {mtype} "
                                f"(needed for 1D param {pname})."
                            )
                    self._count_specs[p] = make_count_sketch(
                        d=int(p.numel()),
                        m=self.projection_size,
                        device=p.device,
                        generator=self._rng,
                    )
                else:
                    raise NotImplementedError(
                        f"Projected SDI currently supports only 1D/2D parameters, got {pname} with dim={p.dim()}."
                    )

    def _handle_module_backward(
        self,
        *,
        module: nn.Module,
        activations: List[torch.Tensor],
        backprops: torch.Tensor,
    ) -> None:
        # Only consider direct target params for this module.
        direct = _module_direct_trainable_params(
            module, target_params_set=self._target_param_set
        )
        if not direct:
            return

        if isinstance(module, nn.Linear):
            act = activations[0]
            for p in direct:
                if getattr(module, "weight", None) is p:
                    spec = self._tensor_specs[p]
                    sketch = tensor_sketch_linear_weight(
                        activations=act, backprops=backprops, spec=spec
                    )
                    self._phi_lists[p].append(sketch.half())
                elif getattr(module, "bias", None) is p:
                    # bias grad is sum over all but batch and last dims
                    # reshape to (B,M,d_out) then sum over M.
                    bp = backprops
                    if bp.dim() == 2:
                        bp_sum = bp
                    else:
                        bp_sum = bp.reshape(bp.shape[0], -1, bp.shape[-1]).sum(dim=1)
                    spec = self._count_specs[p]
                    sketch = count_sketch_vector(bp_sum, spec=spec)
                    self._phi_lists[p].append(sketch.half())
                else:
                    raise RuntimeError("Unexpected parameter for nn.Linear module.")
            return

        if isinstance(module, nn.Embedding):
            ids = activations[0]
            for p in direct:
                if getattr(module, "weight", None) is not p:
                    raise RuntimeError("Unexpected parameter for nn.Embedding module.")
                spec = self._tensor_specs[p]
                sketch = tensor_sketch_embedding_weight(
                    token_ids=ids, backprops=backprops, spec=spec
                )
                self._phi_lists[p].append(sketch.half())
            return

        # Generic 1D handling via Opacus grad samplers
        mtype = type(module)
        gs_fn = GradSampleModule.GRAD_SAMPLERS.get(mtype)
        if gs_fn is None:
            # Should have been caught in validation.
            raise NotImplementedError(
                f"No Opacus grad sampler registered for module type {mtype}"
            )
        grads = gs_fn(module, activations, backprops)
        for p in direct:
            if p not in grads:
                raise RuntimeError(
                    f"Grad sampler for {mtype} did not return grad_sample for parameter {self._param_to_name.get(p, '<unnamed>')}"
                )
            spec = self._count_specs[p]
            self._phi_lists[p].append(
                count_sketch_from_grad_sample(gs=grads[p], spec=spec).half()
            )


class _FullGradHookFeaturizer(_BaseHookFeaturizer):
    """
    Internal hook-based featurizer that materializes per-sample gradients.

    Requires Opacus GRAD_SAMPLERS coverage for all selected modules and stores
    exact per-step gradient tensors (memory-intensive).
    """

    def _validate_supported_modules(self) -> None:
        for m in self._hook_modules:
            direct = _module_direct_trainable_params(
                m, target_params_set=self._target_param_set
            )
            if not direct:
                continue
            if self.strict and len(list(m.buffers(recurse=False))) > 0:
                raise NotImplementedError(
                    f"Selected module {type(m)} has buffers; per-sample gradients are not supported in strict mode."
                )
            mtype = type(m)
            if mtype not in GradSampleModule.GRAD_SAMPLERS:
                # No fallback to functorch/vmap: fail fast.
                # This matches the desired "Opacus would do" behavior for unsupported module types.
                raise NotImplementedError(
                    f"No Opacus grad sampler registered for module type {mtype}. "
                    "FullGradientTracInSDI requires GRAD_SAMPLERS coverage for all selected modules."
                )

    def _handle_module_backward(
        self,
        *,
        module: nn.Module,
        activations: List[torch.Tensor],
        backprops: torch.Tensor,
    ) -> None:
        direct = _module_direct_trainable_params(
            module, target_params_set=self._target_param_set
        )
        if not direct:
            return
        mtype = type(module)
        gs_fn = GradSampleModule.GRAD_SAMPLERS[mtype]
        grads = gs_fn(module, activations, backprops)
        for p in direct:
            if p not in grads:
                raise RuntimeError(
                    f"Grad sampler for {mtype} did not return grad_sample for parameter {self._param_to_name.get(p, '<unnamed>')}"
                )
            self._phi_lists[p].append(grads[p].detach())


def _concat_feature_sets(batches: Sequence[FeatureSet]) -> FeatureSet:
    if not batches:
        raise ValueError("No batches to concatenate")
    keys = batches[0].phi.keys()
    steps = batches[0].steps
    for b in batches[1:]:
        if b.phi.keys() != keys:
            raise ValueError("FeatureSet keys mismatch across batches")
        if b.steps != steps:
            raise ValueError(
                f"FeatureSet steps mismatch across batches: {b.steps} vs {steps}"
            )
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.cat([b.phi[k] for b in batches], dim=0)
    return FeatureSet(phi=out, steps=steps)


class ProjectedTracInSDI:
    """
    Hook-based projected TracIn + SDI estimator.

    - 2D params: TensorSketch for `nn.Linear.weight` and `nn.Embedding.weight`
    - 1D params: CountSketch of true per-sample gradients via Opacus grad samplers
      (or analytic bias gradients for `nn.Linear.bias`).

    This estimator never materializes per-sample matrix gradients for Linear/Embedding,
    which is what makes SDI feasible at transformer scale.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        target_modules: Union[nn.Module, Sequence[nn.Module]],
        projection_size: int = 2048,
        seed: int = 0,
        loss_reduction: str = "sum",
        batch_first: bool = True,
        expected_steps: Optional[int] = None,
        allow_mixed_steps: bool = False,
        strict: bool = True,
    ) -> None:
        self.model = model
        self._featurizer = _ProjectedHookFeaturizer(
            model=model,
            target_modules=_as_module_list(target_modules),
            projection_size=projection_size,
            seed=seed,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
            expected_steps=expected_steps,
            allow_mixed_steps=allow_mixed_steps,
            strict=strict,
        )

    def close(self) -> None:
        self._featurizer.close()

    def featurize_loader(self, *, loader: Iterable[Any], loss_fn: LossFn) -> FeatureSet:
        batches: list[FeatureSet] = []
        for batch in loader:
            batches.append(self._featurizer.compute_batch(batch=batch, loss_fn=loss_fn))
        return _concat_feature_sets(batches)

    def influence_across_checkpoints(
        self,
        *,
        checkpoints: Sequence[CheckpointSpec],
        train_loader: Iterable[Any],
        query_loader: Iterable[Any],
        loss_fn: LossFn,
        mode: Literal["sdi", "tracin"] = "sdi",
        with_step_matrix: bool = False,
        train_chunk_size: Optional[int] = None,
        query_chunk_size: Optional[int] = None,
        load_checkpoint_fn: Callable[
            [Union[str, Path]], Dict[str, Any]
        ] = default_load_checkpoint,
        load_model_state_dict_fn: Callable[
            [nn.Module, Dict[str, Any]], None
        ] = default_load_model_state_dict,
        eta_from_checkpoint_fn: Callable[
            [Dict[str, Any]], float
        ] = default_eta_from_checkpoint,
        progress: bool = False,
        progress_desc: str = "checkpoints",
    ) -> InfluenceOutputs:
        if mode not in ("sdi", "tracin"):
            raise ValueError(f"mode must be 'sdi' or 'tracin', got {mode!r}")
        if with_step_matrix and mode != "sdi":
            raise ValueError("with_step_matrix=True is only supported when mode='sdi'.")
        checkpoints = resolve_checkpoint_weights(
            checkpoints,
            load_checkpoint_fn=load_checkpoint_fn,
            eta_from_checkpoint_fn=eta_from_checkpoint_fn,
        )
        sdi_total: Optional[torch.Tensor] = None
        sdi_mat_total: Optional[torch.Tensor] = None
        tracin_total: Optional[torch.Tensor] = None
        it = (
            tqdm(checkpoints, desc=str(progress_desc), dynamic_ncols=True)
            if progress
            else checkpoints
        )
        for spec in it:
            ckpt = load_checkpoint_fn(spec.path)
            load_model_state_dict_fn(self.model, ckpt)
            self.model.eval()
            train_feat = self.featurize_loader(loader=train_loader, loss_fn=loss_fn)
            query_feat = self.featurize_loader(loader=query_loader, loss_fn=loss_fn)
            w = float(spec.weight if spec.weight is not None else 1.0)
            if mode == "sdi":
                sdi = compute_query_sdi(
                    train=train_feat,
                    query=query_feat,
                    train_chunk_size=train_chunk_size,
                    query_chunk_size=query_chunk_size,
                )
                sdi_total = sdi * w if sdi_total is None else (sdi_total + sdi * w)
                if with_step_matrix:
                    sm = compute_step_matrix(
                        train=train_feat,
                        query=query_feat,
                        train_chunk_size=train_chunk_size,
                        query_chunk_size=query_chunk_size,
                    )
                    sdi_mat_total = (
                        sm * w if sdi_mat_total is None else (sdi_mat_total + sm * w)
                    )
            else:
                tracin = compute_tracin(
                    train=train_feat,
                    query=query_feat,
                    train_chunk_size=train_chunk_size,
                    query_chunk_size=query_chunk_size,
                )
                tracin_total = (
                    tracin * w if tracin_total is None else (tracin_total + tracin * w)
                )
        if mode == "sdi":
            if sdi_total is None:
                raise ValueError("No checkpoints provided")
            tracin = sdi_total.sum(dim=2)
            return InfluenceOutputs(
                sdi=sdi_total, tracin=tracin, sdi_matrix=sdi_mat_total
            )
        if tracin_total is None:
            raise ValueError("No checkpoints provided")
        return InfluenceOutputs(sdi=None, tracin=tracin_total, sdi_matrix=None)


class FullGradientTracInSDI:
    """
    Full per-sample-gradient TracIn + SDI estimator (exact baseline).

    This uses Opacus GRAD_SAMPLERS to materialize per-sample gradients for every
    selected module invocation, then stacks them into stepwise tensors.

    Warning: This is extremely memory-intensive for large matrix parameters
    (e.g., transformer linear layers, embeddings).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        target_modules: Union[nn.Module, Sequence[nn.Module]],
        loss_reduction: str = "sum",
        batch_first: bool = True,
        expected_steps: Optional[int] = None,
        allow_mixed_steps: bool = False,
        strict: bool = True,
    ) -> None:
        self.model = model
        self._featurizer = _FullGradHookFeaturizer(
            model=model,
            target_modules=_as_module_list(target_modules),
            loss_reduction=loss_reduction,
            batch_first=batch_first,
            expected_steps=expected_steps,
            allow_mixed_steps=allow_mixed_steps,
            strict=strict,
        )

    def close(self) -> None:
        self._featurizer.close()

    def featurize_loader(self, *, loader: Iterable[Any], loss_fn: LossFn) -> FeatureSet:
        batches: list[FeatureSet] = []
        for batch in loader:
            batches.append(self._featurizer.compute_batch(batch=batch, loss_fn=loss_fn))
        return _concat_feature_sets(batches)

    def influence_across_checkpoints(
        self,
        *,
        checkpoints: Sequence[CheckpointSpec],
        train_loader: Iterable[Any],
        query_loader: Iterable[Any],
        loss_fn: LossFn,
        mode: Literal["sdi", "tracin"] = "sdi",
        with_step_matrix: bool = False,
        train_chunk_size: Optional[int] = None,
        query_chunk_size: Optional[int] = None,
        load_checkpoint_fn: Callable[
            [Union[str, Path]], Dict[str, Any]
        ] = default_load_checkpoint,
        load_model_state_dict_fn: Callable[
            [nn.Module, Dict[str, Any]], None
        ] = default_load_model_state_dict,
        eta_from_checkpoint_fn: Callable[
            [Dict[str, Any]], float
        ] = default_eta_from_checkpoint,
        progress: bool = False,
        progress_desc: str = "checkpoints",
    ) -> InfluenceOutputs:
        if mode not in ("sdi", "tracin"):
            raise ValueError(f"mode must be 'sdi' or 'tracin', got {mode!r}")
        if with_step_matrix and mode != "sdi":
            raise ValueError("with_step_matrix=True is only supported when mode='sdi'.")
        checkpoints = resolve_checkpoint_weights(
            checkpoints,
            load_checkpoint_fn=load_checkpoint_fn,
            eta_from_checkpoint_fn=eta_from_checkpoint_fn,
        )
        sdi_total: Optional[torch.Tensor] = None
        sdi_mat_total: Optional[torch.Tensor] = None
        tracin_total: Optional[torch.Tensor] = None
        it = (
            tqdm(checkpoints, desc=str(progress_desc), dynamic_ncols=True)
            if progress
            else checkpoints
        )
        for spec in it:
            ckpt = load_checkpoint_fn(spec.path)
            load_model_state_dict_fn(self.model, ckpt)
            self.model.eval()
            train_feat = self.featurize_loader(loader=train_loader, loss_fn=loss_fn)
            query_feat = self.featurize_loader(loader=query_loader, loss_fn=loss_fn)
            w = float(spec.weight if spec.weight is not None else 1.0)
            if mode == "sdi":
                sdi = compute_query_sdi(
                    train=train_feat,
                    query=query_feat,
                    train_chunk_size=train_chunk_size,
                    query_chunk_size=query_chunk_size,
                )
                sdi_total = sdi * w if sdi_total is None else (sdi_total + sdi * w)
                if with_step_matrix:
                    sm = compute_step_matrix(
                        train=train_feat,
                        query=query_feat,
                        train_chunk_size=train_chunk_size,
                        query_chunk_size=query_chunk_size,
                    )
                    sdi_mat_total = (
                        sm * w if sdi_mat_total is None else (sdi_mat_total + sm * w)
                    )
            else:
                tracin = compute_tracin(
                    train=train_feat,
                    query=query_feat,
                    train_chunk_size=train_chunk_size,
                    query_chunk_size=query_chunk_size,
                )
                tracin_total = (
                    tracin * w if tracin_total is None else (tracin_total + tracin * w)
                )
        if mode == "sdi":
            if sdi_total is None:
                raise ValueError("No checkpoints provided")
            tracin = sdi_total.sum(dim=2)
            return InfluenceOutputs(
                sdi=sdi_total, tracin=tracin, sdi_matrix=sdi_mat_total
            )
        if tracin_total is None:
            raise ValueError("No checkpoints provided")
        return InfluenceOutputs(sdi=None, tracin=tracin_total, sdi_matrix=None)
