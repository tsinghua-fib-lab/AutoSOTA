__all__ = [
    "load_model", "ModelSpec",
    "load_benchmark",
]


def __getattr__(name):
    if name in {"load_model", "ModelSpec"}:
        from .models import ModelSpec, load_model
        return {"load_model": load_model, "ModelSpec": ModelSpec}[name]

    if name == "load_benchmark":
        from .tasks import load_benchmark
        return load_benchmark

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
