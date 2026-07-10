"""Conformal Path Reasoning (CPR) for trustworthy KGQA."""

__all__ = ["CPRCore", "ResidualValueMLP"]


def __getattr__(name: str):
    if name in ("CPRCore"):
        from cpr.core import CPRCore
        return CPRCore
    if name == "ResidualValueMLP":
        from cpr.models.rcvnet import ResidualValueMLP
        return ResidualValueMLP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
