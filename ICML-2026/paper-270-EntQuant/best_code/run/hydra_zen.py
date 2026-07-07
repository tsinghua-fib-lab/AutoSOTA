from functools import wraps
from typing import Any, Callable, TypeVar

import torch
from hydra_zen import BuildsFn, make_custom_builds_fn, store
from hydra_zen.typing import CustomConfigType, HydraSupportedType

from entquant.utils import str_to_dtype

SupportedCustomTypes = torch.dtype


class CustomBuilds(BuildsFn[CustomConfigType[SupportedCustomTypes]]):
    """
    Custom BuildsFn that knows how to auto-config torch.dtype instances.
    """

    @classmethod
    def _make_hydra_compatible(cls, value: Any, **kwargs) -> HydraSupportedType:
        if isinstance(value, torch.dtype):
            # Convert torch.dtype to a config that uses get_obj to look it up
            dtype_name = str(value).replace("torch.", "")
            return cls.builds(str_to_dtype, dtype=dtype_name)
        else:
            return super()._make_hydra_compatible(value, **kwargs)


builds = CustomBuilds.builds
just = CustomBuilds.just
kwargs_of = CustomBuilds.kwargs_of
make_config = CustomBuilds.make_config
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True, builds_fn=CustomBuilds)
fbuilds = make_custom_builds_fn(zen_partial=False, populate_full_signature=True, builds_fn=CustomBuilds)

F = TypeVar("F", bound=Callable)


def register_workflow(subgroup: str, **store_kwargs) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        store(pbuilds(fn), group=f"workflow/{subgroup}", name=fn.__name__, **store_kwargs)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator
