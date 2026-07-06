__all__ = [
    "BigE3NN",
    "BigE3NN_NoSH",
    "GotenNetProtein",
    "MACEProteinNet",
]


def __getattr__(name):
    if name in {"BigE3NN", "BigE3NN_NoSH"}:
        from .e3nn import BigE3NN, BigE3NN_NoSH

        return {"BigE3NN": BigE3NN, "BigE3NN_NoSH": BigE3NN_NoSH}[name]
    if name == "GotenNetProtein":
        from .gotennet import GotenNetProtein

        return GotenNetProtein
    if name == "MACEProteinNet":
        from .mace import MACEProteinNet

        return MACEProteinNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
