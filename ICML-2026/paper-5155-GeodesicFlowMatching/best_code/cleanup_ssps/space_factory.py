from __future__ import annotations

import numpy as np

from cleanup_ssps.sspspace import HexagonalSSPSpace, RandomSSPSpace


SPACE_BUILDERS: dict[str, callable] = {}


def register_space_builder(*names: str):
    """Decorator to register one builder under one or more names."""
    def _inner(fn):
        for name in names:
            SPACE_BUILDERS[name.lower()] = fn
        return fn
    return _inner


def resolve_encoded_dim(ssp_config: dict) -> int:
    """Resolve encoded dim from explicit value or hex parameters."""
    if "encoded_dim" in ssp_config and ssp_config["encoded_dim"] is not None:
        return int(ssp_config["encoded_dim"])

    n_rot = ssp_config.get("n_rotates")
    n_scl = ssp_config.get("n_scales")
    if n_rot is None or n_scl is None:
        raise ValueError("encoded_dim missing and cannot be derived without n_rotates and n_scales")

    return int(n_rot) * int(n_scl) * 6 + 1


def _default_bounds(domain_bounds):
    return np.array([[-1, 1], [-1, 1]]) if domain_bounds is None else domain_bounds


@register_space_builder("hex", "hexagonal")
def _build_hexagonal(ssp_config: dict, *, domain_dim: int = 2, domain_bounds=None):
    encoded_dim = resolve_encoded_dim(ssp_config)
    length_scale = float(ssp_config.get("length_scale", 0.2))
    n_rot = int(ssp_config.get("n_rotates", 4))
    n_scl = int(ssp_config.get("n_scales", 4))

    return HexagonalSSPSpace(
        domain_dim=domain_dim,
        ssp_dim=encoded_dim,
        domain_bounds=_default_bounds(domain_bounds),
        length_scale=length_scale,
        n_rotates=n_rot,
        n_scales=n_scl,
    )


@register_space_builder("random")
def _build_random(ssp_config: dict, *, domain_dim: int = 2, domain_bounds=None):
    encoded_dim = resolve_encoded_dim(ssp_config)
    length_scale = float(ssp_config.get("length_scale", 0.2))
    seed = ssp_config.get("random_seed")
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    return RandomSSPSpace(
        domain_dim=domain_dim,
        ssp_dim=encoded_dim,
        domain_bounds=_default_bounds(domain_bounds),
        length_scale=length_scale,
        rng=rng,
    )


def get_registered_bundle_types() -> list[str]:
    return sorted(SPACE_BUILDERS.keys())


def build_ssp_space(ssp_config: dict, *, domain_dim: int = 2, domain_bounds=None):
    """
    Build an SSP space from config using a registry.

    To extend: add a new builder and decorate with @register_space_builder("name").
    """
    bundle_type = str(ssp_config.get("bundle_type", "hexagonal")).lower()
    builder = SPACE_BUILDERS.get(bundle_type)
    if builder is None:
        supported = ", ".join(get_registered_bundle_types())
        raise ValueError(f"Unsupported bundle_type '{bundle_type}'. Registered: {supported}")

    return builder(ssp_config, domain_dim=domain_dim, domain_bounds=domain_bounds)
