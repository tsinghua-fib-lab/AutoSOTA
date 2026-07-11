from ntkunlimited.recursions.symmetric_tensor import SymmetricTensor, NO_4D_SYM
import numpy as np
from functools import partial


def test_symmetric_tensor():
    F = SymmetricTensor(NO_4D_SYM, partial(np.zeros, dtype=float))
    special_comp = (0, 1, 2, 2)
    F[special_comp] = 1.0
    assert F[special_comp] == 1.0
    for comp in F.sym_map:
        if comp != special_comp:
            assert F[comp] == 0.0


if __name__ == "__main__":
    test_symmetric_tensor()
