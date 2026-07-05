from .dcfr import DCFR


class LCFR(DCFR):
    """Linear CFR: DCFR with linear discounting (alpha = beta = gamma = 1)."""

    def __init__(self):
        super().__init__(alpha=1, beta=1, gamma=1)
