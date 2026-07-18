from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
import shutil
from typing import Any

import torch
from torch import nn

Time = float | torch.Tensor
BioEmuPairField = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def _parse_bioemu_sequence(sequence: str | Path) -> str:
    """Parse a BioEmu sequence argument into a single-letter amino-acid string."""
    from bioemu.seq_io import parse_sequence

    return parse_sequence(sequence)


def bioemu_cached_embedding_paths(
    sequence: str | Path,
    cache_embeds_dir: str | Path | None = None,
    *,
    single_embeds_file: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return the canonical BioEmu cache paths for a sequence.

    BioEmu names cached Evoformer embeddings from the sequence hash. When an
    explicit ``single_embeds_file`` is provided and ``cache_embeds_dir`` is not,
    its parent directory is used as the cache destination.
    """
    from bioemu.get_embeds import _get_default_embeds_dir, shahexencode

    parsed_sequence = _parse_bioemu_sequence(sequence)
    if cache_embeds_dir is None and single_embeds_file is not None:
        cache_dir = Path(single_embeds_file).expanduser().resolve().parent
    else:
        cache_dir = Path(cache_embeds_dir or _get_default_embeds_dir()).expanduser()
    seqsha = shahexencode(parsed_sequence)
    return cache_dir / f"{seqsha}_single.npy", cache_dir / f"{seqsha}_pair.npy"


def _copy_embedding_file(source: str | Path, destination: Path) -> None:
    """Copy an embedding file into the canonical BioEmu cache location."""
    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"BioEmu embedding file does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _validate_cached_embeddings(
    sequence: str,
    single_path: Path,
    pair_path: Path,
) -> None:
    """Check that cached BioEmu embedding arrays have sequence-compatible shapes."""
    import numpy as np

    single = np.load(single_path, mmap_mode="r")
    pair = np.load(pair_path, mmap_mode="r")
    n_residues = len(sequence)

    if single.ndim != 2 or single.shape[0] != n_residues:
        raise ValueError(
            "single BioEmu embeddings must have shape "
            f"({n_residues}, n_features), got {single.shape}"
        )
    if pair.ndim != 3 or pair.shape[0] != n_residues or pair.shape[1] != n_residues:
        raise ValueError(
            "pair BioEmu embeddings must have shape "
            f"({n_residues}, {n_residues}, n_features), got {pair.shape}"
        )


def prepare_bioemu_embedding_cache(
    sequence: str | Path,
    cache_embeds_dir: str | Path | None = None,
    *,
    single_embeds_file: str | Path | None = None,
    pair_embeds_file: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    allow_embedding_generation: bool = False,
    validate: bool = True,
) -> tuple[Path, Path]:
    """Create or validate the BioEmu embedding cache needed for a sequence.

    If ``single_embeds_file`` and ``pair_embeds_file`` are provided, they are
    copied into BioEmu's canonical hash-based cache filenames. If the canonical
    files already exist, they are reused. If embeddings are missing and
    ``allow_embedding_generation`` is true, BioEmu is allowed to generate them.
    """
    parsed_sequence = _parse_bioemu_sequence(sequence)
    single_path, pair_path = bioemu_cached_embedding_paths(
        parsed_sequence,
        cache_embeds_dir,
        single_embeds_file=single_embeds_file,
    )
    single_path.parent.mkdir(parents=True, exist_ok=True)

    if (single_embeds_file is None) != (pair_embeds_file is None):
        raise ValueError(
            "Pass both single_embeds_file and pair_embeds_file, or neither."
        )

    if single_embeds_file is not None and pair_embeds_file is not None:
        _copy_embedding_file(single_embeds_file, single_path)
        _copy_embedding_file(pair_embeds_file, pair_path)

    missing = [path for path in (single_path, pair_path) if not path.exists()]
    if missing and allow_embedding_generation:
        from bioemu.get_embeds import get_colabfold_embeds

        generated_single, generated_pair = get_colabfold_embeds(
            seq=parsed_sequence,
            cache_embeds_dir=single_path.parent,
            msa_file=msa_file,
            msa_host_url=msa_host_url,
        )
        single_path = Path(generated_single)
        pair_path = Path(generated_pair)
        missing = [path for path in (single_path, pair_path) if not path.exists()]

    if missing:
        missing_text = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "BioEmu embeddings are not cached. Provide single_embeds_file and "
            "pair_embeds_file, precompute the files in cache_embeds_dir, or set "
            "allow_embedding_generation=True in an environment where BioEmu's "
            "AlphaFold/ColabFold embedding path is stable.\n\n"
            "Missing files:\n"
            f"{missing_text}"
        )

    if validate:
        _validate_cached_embeddings(parsed_sequence, single_path, pair_path)

    return single_path, pair_path


def _get_field(batch: Any, name: str) -> torch.Tensor:
    """Read a named tensor from a BioEmu object or mapping."""
    if hasattr(batch, name):
        return getattr(batch, name)
    return batch[name]


def _batch_time_like(t: Time, batch_size: int, x: torch.Tensor) -> torch.Tensor:
    """Convert a scalar or batch time to a tensor on x's device and dtype."""
    if torch.is_tensor(t):
        t = t.to(device=x.device, dtype=x.dtype)
    else:
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)

    if t.ndim == 0:
        return t.expand(batch_size)

    if t.shape == (batch_size,):
        return t

    if t.numel() == 1:
        return t.reshape(()).expand(batch_size)

    raise ValueError(
        f"t must be a scalar or have shape ({batch_size},), got {tuple(t.shape)}"
    )


def _to_flat_state(
    pos: torch.Tensor, node_orientations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    """Flatten batched BioEmu position and orientation tensors for ChemGraph use."""
    if pos.ndim not in (2, 3):
        raise ValueError(
            "pos must have shape (n_residues, 3) or "
            f"(batch, n_residues, 3), got {tuple(pos.shape)}"
        )
    if node_orientations.ndim not in (3, 4):
        raise ValueError(
            "node_orientations must have shape (n_residues, 3, 3) or "
            f"(batch, n_residues, 3, 3), got {tuple(node_orientations.shape)}"
        )
    if pos.shape[-1] != 3:
        raise ValueError(f"pos last dimension must be 3, got {tuple(pos.shape)}")
    if node_orientations.shape[-2:] != (3, 3):
        raise ValueError(
            "node_orientations last two dimensions must be (3, 3), "
            f"got {tuple(node_orientations.shape)}"
        )
    if pos.shape[:-1] != node_orientations.shape[:-2]:
        raise ValueError(
            "pos and node_orientations must have matching leading dimensions, "
            f"got {tuple(pos.shape)} and {tuple(node_orientations.shape)}"
        )

    pos_shape = tuple(pos.shape)
    rot_shape = tuple(node_orientations.shape)

    if pos.ndim == 3:
        pos = pos.reshape(-1, pos.shape[-1])
    if node_orientations.ndim == 4:
        node_orientations = node_orientations.reshape(
            -1, *node_orientations.shape[-2:]
        )

    return pos, node_orientations, pos_shape, rot_shape


def _restore_shape(value: torch.Tensor, state_shape: tuple[int, ...]) -> torch.Tensor:
    """Restore a flat field to the leading dimensions of its original state tensor."""
    if len(state_shape) <= 2:
        return value
    return value.reshape(*state_shape[:-1], value.shape[-1])


def _maybe_repeat_context_batch(batch: Any, batch_size: int) -> Any:
    """Repeat a single BioEmu context graph to match a requested batch size."""
    if getattr(batch, "num_graphs", None) == batch_size:
        return batch

    if getattr(batch, "num_graphs", None) == 1 and batch_size > 1:
        from torch_geometric.data.batch import Batch

        return Batch.from_data_list(batch.to_data_list() * batch_size)

    raise ValueError(
        f"BioEmu batch has {batch.num_graphs} graphs, but state requires {batch_size}"
    )


def _state_batch(
    *,
    pos: torch.Tensor,
    node_orientations: torch.Tensor,
    batch: Any | None,
    default_batch: Any | None,
) -> tuple[Any, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    """Build a BioEmu batch carrying the requested position and rotation state."""
    if batch is None:
        batch = default_batch
    if batch is None:
        raise ValueError("Pass a BioEmu ChemGraph/Batch, or provide sequence= at load time.")

    pos, node_orientations, pos_shape, rot_shape = _to_flat_state(pos, node_orientations)
    if len(pos_shape) == 3:
        batch = _maybe_repeat_context_batch(batch, pos_shape[0])

    batch = batch.to(pos.device)
    batch = batch.replace(pos=pos, node_orientations=node_orientations)
    return batch, pos, node_orientations, pos_shape, rot_shape


@torch.no_grad()
def _score_fields(
    *,
    score_model: nn.Module,
    sdes: Mapping[str, Any],
    pos: torch.Tensor,
    node_orientations: torch.Tensor,
    t: Time,
    batch: Any | None = None,
    default_batch: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], Any, torch.Tensor]:
    """Evaluate BioEmu position and SO(3) scores with the required model scalings."""
    batch, pos, node_orientations, pos_shape, rot_shape = _state_batch(
        pos=pos,
        node_orientations=node_orientations,
        batch=batch,
        default_batch=default_batch,
    )
    bioemu_t = 1.0 - _batch_time_like(t, batch.num_graphs, pos)

    raw_score = score_model(batch, bioemu_t)

    _, pos_std = sdes["pos"].marginal_prob(
        x=torch.ones_like(_get_field(raw_score, "pos")),
        t=bioemu_t,
        batch_idx=batch.batch,
    )
    pos_score = _get_field(raw_score, "pos") / pos_std
    rot_score = _get_field(raw_score, "node_orientations") * sdes[
        "node_orientations"
    ].get_score_scaling(bioemu_t, batch_idx=batch.batch)[:, None]

    return pos_score, rot_score, pos_shape, rot_shape, batch, bioemu_t


def _reverse_probability_flow_drift(
    *,
    sde: Any,
    x: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
    marginal_concentration_factor: float,
) -> torch.Tensor:
    """Compute BioEmu's reverse-time probability-flow drift for one SDE field."""
    drift, diffusion = sde.sde(x=x, t=t, batch_idx=batch_idx)
    if diffusion.ndim == score.ndim - 1:
        diffusion = diffusion.unsqueeze(-1)
    score_weight = 0.5 * marginal_concentration_factor
    return drift - diffusion**2 * score * score_weight


@torch.no_grad()
def _bioemu_pair_field(
    pos: torch.Tensor,
    node_orientations: torch.Tensor,
    t: Time,
    *,
    kind: str,
    score_model: nn.Module,
    sdes: Mapping[str, Any],
    marginal_concentration_factor: float,
    batch: Any | None = None,
    default_batch: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate both BioEmu position and SO(3) fields with one model call."""
    pos_score, rot_score, pos_shape, rot_shape, batch, bioemu_t = _score_fields(
        score_model=score_model,
        sdes=sdes,
        pos=pos,
        node_orientations=node_orientations,
        t=t,
        batch=batch,
        default_batch=default_batch,
    )

    if kind == "score":
        return (
            _restore_shape(pos_score, pos_shape),
            _restore_shape(rot_score, rot_shape[:-1]),
        )

    pos_reverse_drift = _reverse_probability_flow_drift(
        sde=sdes["pos"],
        x=_get_field(batch, "pos"),
        t=bioemu_t,
        batch_idx=batch.batch,
        score=pos_score,
        marginal_concentration_factor=marginal_concentration_factor,
    )
    rot_reverse_drift = _reverse_probability_flow_drift(
        sde=sdes["node_orientations"],
        x=_get_field(batch, "node_orientations"),
        t=bioemu_t,
        batch_idx=batch.batch,
        score=rot_score,
        marginal_concentration_factor=marginal_concentration_factor,
    )

    return (
        _restore_shape(-pos_reverse_drift, pos_shape),
        _restore_shape(-rot_reverse_drift, rot_shape[:-1]),
    )


def _build_single_sequence_batch(
    *,
    sequence: str | Path,
    cache_embeds_dir: str | Path | None,
    single_embeds_file: str | Path | None,
    pair_embeds_file: str | Path | None,
    msa_file: str | Path | None,
    msa_host_url: str | None,
    device: torch.device | str | None,
    allow_embedding_generation: bool,
) -> Any:
    """Build one BioEmu context graph; callbacks repeat it for any batch size."""
    parsed_sequence = _parse_bioemu_sequence(sequence)
    single_path, _ = prepare_bioemu_embedding_cache(
        parsed_sequence,
        cache_embeds_dir=cache_embeds_dir,
        single_embeds_file=single_embeds_file,
        pair_embeds_file=pair_embeds_file,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
        allow_embedding_generation=allow_embedding_generation,
    )

    from bioemu.sample import get_context_chemgraph
    from torch_geometric.data.batch import Batch

    context = get_context_chemgraph(
        sequence=parsed_sequence,
        cache_embeds_dir=single_path.parent,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    batch = Batch.from_data_list([context])
    if device is not None:
        batch = batch.to(device)
    return batch


def bioemu_vector_fields_from_model(
    score_model: nn.Module,
    sequence: str | Path,
    *,
    sdes: Mapping[str, Any] | None = None,
    model_config_path: str | Path | None = None,
    cache_embeds_dir: str | Path | None = None,
    single_embeds_file: str | Path | None = None,
    pair_embeds_file: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    marginal_concentration_factor: float = 1.0,
    device: torch.device | str | None = None,
    allow_embedding_generation: bool = False,
) -> tuple[BioEmuPairField, BioEmuPairField]:
    """Expose paired stochastic-interpolant callbacks from a loaded BioEmu model.

    The returned ``b`` and ``score`` callables accept
    ``(pos, node_orientations, t, batch=None)`` and return
    ``(pos_component, rotation_component)``. Their batch size is inferred from
    the leading dimension of ``pos`` at call time, so it does not need to be
    fixed when this factory is called.

    Each callback evaluates ``score_model`` exactly once and then applies the
    BioEmu marginal score scalings, SO(3) geometric score scaling, diffusion
    time reversal, and probability-flow drift conversion needed for the
    stochastic-interpolant ``score`` and ``b`` fields.

    Pass ``sdes`` directly when they were loaded with the model. If ``sdes`` is
    omitted, ``model_config_path`` is required to load the matching BioEmu SDEs.
    Pass ``single_embeds_file`` and ``pair_embeds_file`` to import existing
    embedding arrays into BioEmu's cache. By default, missing sequence
    embeddings raise ``FileNotFoundError`` instead of letting BioEmu generate
    them in-process; set ``allow_embedding_generation=True`` only in
    environments where BioEmu's AlphaFold/ColabFold embedding path is known to
    be stable.
    """
    if sdes is None:
        from bioemu.model_utils import load_sdes

        if model_config_path is None:
            raise ValueError(
                "Pass sdes= or model_config_path= when using an already-loaded "
                "BioEmu model."
            )
        sdes = load_sdes(
            model_config_path=model_config_path,
            cache_so3_dir=cache_so3_dir,
        )

    if device is not None:
        score_model = score_model.to(device)
        for sde in sdes.values():
            if isinstance(sde, nn.Module):
                sde.to(device)

    score_model.eval()
    batch = _build_single_sequence_batch(
        sequence=sequence,
        cache_embeds_dir=cache_embeds_dir,
        single_embeds_file=single_embeds_file,
        pair_embeds_file=pair_embeds_file,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
        device=device,
        allow_embedding_generation=allow_embedding_generation,
    )

    field_kwargs = {
        "score_model": score_model,
        "sdes": sdes,
        "marginal_concentration_factor": marginal_concentration_factor,
        "default_batch": batch,
    }
    b = partial(_bioemu_pair_field, kind="b", **field_kwargs)
    score = partial(_bioemu_pair_field, kind="score", **field_kwargs)

    return b, score
