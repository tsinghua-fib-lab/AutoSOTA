"""EngramEditor: collect input covariances, extract engram projections, apply edits.

Pipeline: :meth:`~EngramEditor.collect_statistics` (mean input covariance + counts) ->
:meth:`~EngramEditor.compute_engram_weights` (per-layer projection
``P = W . C_target . pinv(C_total)``, bias-absorbed when applicable) ->
:meth:`~EngramEditor.apply` / :meth:`~EngramEditor.edit`
(``W <- W - alpha * f_l * P_l`` with a pluggable per-layer scaling function ``f_l``).
The paper's ``n / N`` sample-count factor is the default scaling
(:func:`engram.scaling.count_ratio`), applied at edit time so it can be swapped freely.
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .collectors import CovarianceCollector
from .config import EditorConfig
from .handlers import Conv1DHandler, LinearHandler, get_conv1d_class, handler_for
from .scaling import EngramResult, LayerScaleInfo, ScaleFn, _erank, count_ratio
from .stats import Statistics

logger = logging.getLogger("engram")


class EngramEditor:
    """Covariance-based engram extractor for PyTorch / HuggingFace models.

    The editor hooks every module whose type is in ``self.registry`` (by default
    ``nn.Linear`` and, when ``transformers`` is importable, HF ``Conv1D`` for the
    GPT-2 family). To restrict covariance to answer tokens, pass a ``mask_fn`` to
    ``collect_statistics`` (e.g. ``mask_fn=lambda b: b["labels"] != -100``).

    Example::

        editor = EngramEditor(model, EditorConfig())
        target = editor.collect_statistics(forget_loader, batch_fn=bf)   # Statistics
        total = editor.collect_statistics(all_loader, batch_fn=bf)
        edited = editor.edit(target, total, alpha=1.0)                   # paper edit (count_ratio)
        # or, to reuse / re-scale the projection without recollecting:
        #   engram = editor.compute_engram_weights(target, total)        # EngramResult
        #   edited = editor.apply(engram, alpha=0.6, scale=weight_norm(1.0))
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[EditorConfig] = None,
        adapters: Optional[List[Any]] = None,
    ) -> None:
        self.model = model
        self.config = config or EditorConfig()
        self.registry: Dict[type, Any] = {nn.Linear: LinearHandler()}
        conv1d = get_conv1d_class()
        if conv1d is not None:
            self.registry[conv1d] = Conv1DHandler()
        # opt-in extensions for layers the registry can't hook (e.g. fused MoE
        # experts via engram.moe.FusedExpertAdapter). Empty -> core is MoE-unaware.
        self.adapters = list(adapters or [])

    @property
    def _model_device(self) -> torch.device:
        """Device the model lives on — covariances are computed and the solve runs here."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def _storage_device(self) -> Any:
        """Where covariances are held: ``config.storage_device``, or the model device if None."""
        sd = self.config.storage_device
        return sd if sd is not None else self._model_device

    # ---- forward helpers (support tensor / tuple / HF dict batches) ----
    def _move_to_device(self, data: Any) -> Any:
        if torch.is_tensor(data):
            return data.to(self._model_device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(v) for v in data)
        return data

    def _forward(self, inputs: Any) -> None:
        if isinstance(inputs, dict):
            self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) > 0 and isinstance(inputs[0], dict):
                self.model(**inputs[0])
            else:
                self.model(*inputs)
        else:
            self.model(inputs)

    def collect_statistics(
        self,
        dataloader: Iterable[Any],
        target_modules: Optional[Union[str, List[str]]] = None,
        batch_fn: Optional[Callable[[Any], Any]] = None,
        mask_fn: Optional[Callable[[Any], torch.Tensor]] = None,
        layers_to_transform: Optional[Union[int, List[int]]] = None,
        layers_pattern: Optional[Union[str, List[str]]] = None,
        target_layers: Optional[List[str]] = None,
    ) -> Statistics:
        """Accumulate the mean input covariance ``mean(x^T x)`` + counts per supported layer.

        Args:
            dataloader: iterable of batches.
            target_modules: which modules to collect, **LoRA/PEFT convention**. A
                list matches by name suffix (``["down_proj", "q_proj"]`` hits every
                layer's ``down_proj``/``q_proj``); a string is a regex matched
                against the full module path (``r".*layers\\.5\\..*down_proj"``).
                ``None`` (default) collects every supported layer ("all-linear").
            layers_to_transform: restrict to these decoder-layer indices (an int or
                list of ints), like PEFT. Combined with ``target_modules`` as an AND
                filter; ``None`` (default) applies no index restriction.
            layers_pattern: container name(s) holding the layer index
                (``"layers"``, ``"h"``, …). ``None`` scans common patterns.
            target_layers: **deprecated** alias of ``target_modules`` (exact module
                names still match, as they did before).
            batch_fn: maps a batch to model inputs (a tensor, a tuple of
                positional args, or a dict of keyword args). Defaults to
                ``batch[0]`` (vision-style loaders). For HF models pass e.g.
                ``lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}``.
            mask_fn: optional ``batch -> bool tensor`` selecting which tokens enter
                the covariance (one entry per flattened token, e.g.
                ``lambda b: b["labels"] != -100`` for answer-token-only LLM editing).
                Applied to every layer type (``nn.Linear``, ``Conv1D``, …).

        Returns:
            A :class:`engram.stats.Statistics`: per-layer **mean** covariance ``[D, D]``
            on ``config.storage_device`` plus the sample counts. ``D`` is the layer's
            input dim, or ``input dim + 1`` when bias absorption applies
            (``config.absorb_bias`` and the layer has a bias).
        """
        if target_layers is not None:
            if target_modules is None:
                target_modules = target_layers
            warnings.warn(
                "target_layers is deprecated; use target_modules "
                "(LoRA-style: list = name suffix, str = regex).",
                DeprecationWarning,
                stacklevel=2,
            )
        collector = CovarianceCollector(
            self.model,
            self.config,
            self.registry,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
            adapters=self.adapters,
        )
        self.model.eval()

        with collector, torch.inference_mode():
            for batch in tqdm(
                dataloader, disable=None, desc="Collecting covariance"
            ):
                if mask_fn is not None:
                    collector.set_mask(mask_fn(batch))
                raw_inputs = batch_fn(batch) if batch_fn else batch[0]
                self._forward(self._move_to_device(raw_inputs))

        return Statistics(collector.covariance_matrices, collector.sample_counts)

    @staticmethod
    def merge_statistics(*stats: Statistics) -> Statistics:
        """Count-weighted merge of several :class:`Statistics` (used to build totals)."""
        return Statistics.merge(*stats)

    @torch.no_grad()
    def compute_engram_weights(
        self,
        target_covariances: Union[Statistics, List[Statistics]],
        total_covariance: Statistics,
        *,
        compute_erank: bool = False,
    ) -> EngramResult:
        """Compute the per-layer engram projection ``P = W . C_target . pinv(C_total)``.

        ``C_*`` are the **mean** covariances from :meth:`collect_statistics`. ``P`` is the
        *pure* projection — the paper's ``n / N`` sample-count factor is **not** folded in
        here; it is applied at edit time by the scaling function (default
        :func:`engram.scaling.count_ratio`), so the scaling can be swapped without
        recomputing. Whether a layer was bias-absorbed is inferred from the covariance
        size (``D == in + 1``), kept consistent with collection without re-passing a flag.

        Args:
            target_covariances: statistics of the data to isolate (the "forget"/target
                set). A single :class:`Statistics`, or a list merged first
                (count-weighted) via :meth:`merge_statistics`.
            total_covariance: statistics of the total/reference set.
            compute_erank: also compute each layer's target/total effective rank and store
                them in the result (needed only by :func:`engram.scaling.effective_rank`).
                Default ``False`` — skips the extra eigendecompositions.

        Returns:
            An :class:`engram.scaling.EngramResult`: ``layers[name]`` holds the projection
            plus the per-layer scale inputs (counts ``n``/``N``, weight, total covariance);
            ``bias[name]`` holds the bias projection for bias-absorbed layers.
        """
        if isinstance(target_covariances, list):
            target_covariances = self.merge_statistics(*target_covariances)

        missing = [k for k in target_covariances if k not in total_covariance]
        if missing:
            warnings.warn(
                f"compute_engram_weights: {len(missing)} target layer(s) are absent from "
                f"total_covariance and were skipped (e.g. {missing[:3]}) — is total a superset "
                f"of target?",
                stacklevel=2,
            )

        result = EngramResult()
        modules = dict(self.model.named_modules())
        dev = self._model_device  # float32 throughout — float64's pinv is catastrophic on
        prec = torch.float32      # ill-conditioned C_total (see CovarianceCollector).

        for layer_name, c_target in tqdm(
            target_covariances.items(), disable=None, desc="Computing engram"
        ):
            if layer_name not in total_covariance:
                continue
            c_target = c_target.to(dev, dtype=prec)
            c_total = total_covariance[layer_name].to(dev, dtype=prec)
            n = int(target_covariances.count.get(layer_name, 0))
            N = int(total_covariance.count.get(layer_name, 0))
            # pinv's rcond is load-bearing: float32's coarse singular-value cut discards the
            # near-null directions of the ill-conditioned C_total (float64 keeps and
            # 1/sigma-amplifies them -> catastrophic edit). Pin rtol to torch's own default
            # formula (max(M,N) * eps) so this regularization is explicit and independent of
            # any future change to the library's default tolerance. C_total is square (D x D).
            rtol = c_total.shape[-1] * torch.finfo(prec).eps
            pinv_total = torch.linalg.pinv(c_total, rtol=rtol)
            er_t = _erank(c_target) if compute_erank else None  # only effective_rank needs these
            er_a = _erank(c_total) if compute_erank else None

            module = modules.get(layer_name)
            if module is None:
                # not a hooked module — an opt-in adapter (e.g. fused MoE experts) may own it
                adapter = next((a for a in self.adapters if a.owns(layer_name, self.model)), None)
                if adapter is None:
                    continue
                w = adapter.weight_for(layer_name, self.model)  # [out, in] (a Parameter slice)
                result.layers[layer_name] = LayerScaleInfo(
                    name=layer_name, weight_fro=float(w.float().norm()),  # scalar; full weight not retained
                    projection=w.to(dev, dtype=prec) @ c_target @ pinv_total,
                    n=n, N=N, target_erank=er_t, total_erank=er_a,
                )
                continue

            handler = handler_for(self.registry, module)
            if handler is None:
                continue

            in_dim = handler.get_input_dim(module, absorb_bias=False)
            absorbed = c_total.shape[0] == in_dim + 1 and getattr(module, "bias", None) is not None
            full = handler.weight_matrix(module, absorb_bias=absorbed).to(dev, dtype=prec) @ c_target @ pinv_total

            if absorbed:
                proj = handler.to_weight_shape(full[:, :in_dim], module)
                result.bias[layer_name] = full[:, in_dim:].reshape(-1)
            else:
                proj = handler.to_weight_shape(full, module)
            result.layers[layer_name] = LayerScaleInfo(
                name=layer_name, weight_fro=float(module.weight.float().norm()),  # scalar; immune to later in-place edits
                projection=proj, n=n, N=N, target_erank=er_t, total_erank=er_a,
            )

        return result

    @torch.no_grad()
    def apply(
        self,
        engram: EngramResult,
        *,
        alpha: float = 1.0,
        scale: Optional[ScaleFn] = None,
        inplace: bool = False,
    ) -> nn.Module:
        """Subtract the engram from the model: ``W <- W - alpha * f_l * P_l``.

        Args:
            engram: result of :meth:`compute_engram_weights` (projections + scale inputs).
            alpha: global edit strength (the paper's strength; ``1.0`` = full removal at
                the default scaling).
            scale: a scaling function ``{name: LayerScaleInfo} -> {name: float}`` giving the
                per-layer factor ``f_l``. Defaults to :func:`engram.scaling.count_ratio`
                (``f_l = n / N``), which reproduces the paper. See :mod:`engram.scaling`
                for ``weight_norm`` / ``effective_rank`` / ``uniform`` / ``compose``.
            inplace: edit ``self.model`` in place; otherwise edit and return a deep copy.

        Returns:
            the edited model. Fused-expert keys are written to their 3D-Parameter slices
            via the registered adapter.
        """
        if scale is None:
            scale = count_ratio(1.0)
        model = self.model if inplace else copy.deepcopy(self.model)
        modules = dict(model.named_modules())
        factors = scale(engram.layers)

        for name, info in engram.layers.items():
            s = alpha * float(factors.get(name, 0.0))
            if s == 0.0:
                continue
            module = modules.get(name)
            if module is not None:
                module.weight.data -= (s * info.projection).to(module.weight.device, module.weight.dtype)
                b = engram.bias.get(name)
                if b is not None and getattr(module, "bias", None) is not None:
                    module.bias.data -= (s * b).to(module.bias.device, module.bias.dtype)
                continue
            adapter = next((a for a in self.adapters if a.owns(name, model)), None)
            if adapter is not None:
                adapter.apply_delta(model, name, s * info.projection)
        return model

    def edit(
        self,
        target_covariances: Union[Statistics, List[Statistics]],
        total_covariance: Statistics,
        *,
        alpha: float = 1.0,
        scale: Optional[ScaleFn] = None,
        inplace: bool = False,
        compute_erank: bool = False,
    ) -> nn.Module:
        """One call: :meth:`compute_engram_weights` then :meth:`apply`; returns the edited model."""
        engram = self.compute_engram_weights(
            target_covariances, total_covariance, compute_erank=compute_erank
        )
        return self.apply(engram, alpha=alpha, scale=scale, inplace=inplace)

    def save_statistics(self, stats: Statistics, path: Union[str, Path]) -> None:
        """Save collected :class:`Statistics` (mean covariances + counts) with ``torch.save``."""
        stats.save(path)
        logger.info("Saved statistics to %s", path)

    def load_statistics(self, path: Union[str, Path]) -> Statistics:
        """Load :class:`Statistics` onto the storage device (the model's device by default)."""
        return Statistics.load(path, map_location=self._storage_device)
