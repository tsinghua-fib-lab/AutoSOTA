import numpy as np
from numpy.typing import ArrayLike
from itertools import product
from functools import partial
from sympy.combinatorics import Permutation, PermutationGroup
from functools import lru_cache


def generate_full_symmetry_map(tensor_dim: int, data_size: int):
    sym_map = dict(
        [
            (inds, tuple(sorted(inds)))
            for inds in product(range(data_size), repeat=tensor_dim)
        ]
    )
    return sym_map, tensor_dim, data_size


@lru_cache(maxsize=1024)
def generate_symmetry_map(
    symmetries: tuple[tuple[int, ...]], tensor_dim: int = None, data_size: int = None
):
    # We only convert to tuple and then back to list because we want to cache this function
    symmetries = list(symmetries)
    num_syms = len(symmetries)

    if len(symmetries) != 0 and tensor_dim is None:
        tensor_dim = len(symmetries[0])
    else:
        if tensor_dim is None:
            raise ValueError(
                "Either symmetries or tensor_dim must be provided to infer the "
                "dimension."
            )
    if data_size is None:
        data_size = tensor_dim

    identity = tuple(range(tensor_dim))
    if identity not in symmetries:
        symmetries.append(identity)

    # ----------------------------
    sequences = set(product(range(data_size), repeat=tensor_dim))

    # """Partition a set of sequences into orbits under a permutation group."""
    orbits = []

    permutations = [Permutation(sym) for sym in symmetries]
    group = list(PermutationGroup(permutations).generate())

    left_sequences = sequences.copy()
    while left_sequences:
        seq = left_sequences.pop()
        orbit = set(tuple(seq[g(i)] for i in range(len(seq))) for g in group)
        orbits.append(orbit)
        left_sequences -= orbit  # remove all elements in this orbit

    sym_map = {}
    for seq in sequences:
        for orbit in orbits:
            if seq in orbit:
                rep = min(orbit)
                break
        sym_map[seq] = rep
    return sym_map, tensor_dim, data_size


class TensorSymmetry:
    def __init__(
        self,
        symmetries: list[tuple[int, ...]] | None,
        fully_sym: bool = False,
        tensor_dim: int = None,
        data_size: int = None,
    ):
        if fully_sym:
            self.sym_map, self.tensor_dim, self.data_size = generate_full_symmetry_map(
                tensor_dim, data_size
            )
        else:
            self.sym_map, self.tensor_dim, self.data_size = generate_symmetry_map(
                tuple(symmetries), tensor_dim, data_size
            )
        self.unique_inds = sorted(list(set(self.sym_map.values())))
        self.num_unique = len(self.unique_inds)
        self.flat_map = self.comp_unique_flat_ind()
        self.full_size = self.data_size**self.tensor_dim

    def comp_unique_flat_ind(self):
        flat_map = {}
        for k in self.sym_map.keys():
            flat_map[k] = self.unique_inds.index(self.sym_map[k])
        return flat_map

    def get_unique_flat_ind(self, inds):
        if not isinstance(inds, tuple):
            inds = tuple(inds)
        return self.flat_map[inds]

    def get_canonical_ind(self, ind):
        return self.unique_inds[ind]

    def __iter__(self):
        for ind in self.unique_inds:
            yield ind


pairwise_4d_symms = [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]


FULL_4D_SYM = TensorSymmetry(None, fully_sym=True, tensor_dim=4, data_size=4)
FULL_2D_SYM = TensorSymmetry(None, fully_sym=True, tensor_dim=2, data_size=4)
NO_2D_T_IN_4D_D_SYM = TensorSymmetry([], tensor_dim=2, data_size=4)
NO_4D_SYM = TensorSymmetry([], tensor_dim=4, data_size=4)
PAIRWISE_4D_SYM = TensorSymmetry(pairwise_4d_symms)


class SymmetricTensor:
    def __init__(self, sym_map: TensorSymmetry, numpy_init):
        self.sym_map = sym_map
        self.data = numpy_init(self.sym_map.num_unique)

    @classmethod
    def init_from_array(cls, sym_map: TensorSymmetry, arr: ArrayLike):
        arr = np.asarray(arr)
        if arr.shape != (sym_map.num_unique,):
            raise ValueError(
                f"Expected array of shape ({sym_map.num_unique},), got {arr.shape}."
            )

        instance = cls(sym_map, partial(np.empty, dtype=float))
        instance.data[:] = arr
        return instance

    def __getitem__(self, inds: tuple[int, ...]):
        if not isinstance(inds, tuple):
            inds = tuple(inds)
        if len(inds) != self.sym_map.tensor_dim:
            raise IndexError(
                f"Expected {self.sym_map.tensor_dim} indices, got {len(inds)}."
            )
        flat_ind = self.sym_map.get_unique_flat_ind(inds)
        return self.data[flat_ind]

    def __setitem__(self, inds: tuple[int, ...], value):
        if not isinstance(inds, tuple):
            inds = tuple(inds)
        if len(inds) != self.sym_map.tensor_dim:
            raise IndexError(
                f"Expected {self.sym_map.tensor_dim} indices, got {len(inds)}."
            )
        flat_ind = self.sym_map.get_unique_flat_ind(inds)
        self.data[flat_ind] = value

    def __len__(self):
        return self.sym_map.num_unique

    def __iter__(self):
        for i in range(self.sym_map.num_unique):
            yield self.data[i]

    def tolist(self):
        return self.data.tolist()

    def to_dense(self):
        dense_shape = (self.sym_map.data_size,) * self.sym_map.tensor_dim
        dense = np.empty(dense_shape, dtype=self.data.dtype)
        for inds in product(
            range(self.sym_map.data_size), repeat=self.sym_map.tensor_dim
        ):
            dense[inds] = self[inds]
        return dense


# Example usage
if __name__ == "__main__":
    sym_map = TensorSymmetry(pairwise_4d_symms)
    tensor = SymmetricTensor(sym_map, partial(np.empty, dtype=float))
    print("Number of unique elements:", len(tensor))
    tensor[0, 0, 1, 1] = 1.0
    tensor[1, 1, 0, 0] = 2.0  # Should map to the same element as (0,0,1,1)
    tensor[0, 1, 0, 1] = 3.0
    print("Tensor data:", tensor.tolist())
    print("Element (0,0,1,1):", tensor[0, 0, 1, 1])
    print("Element (1,1,0,0):", tensor[1, 1, 0, 0])
    print("Element (0,1,0,1):", tensor[0, 1, 0, 1])
    # print("Dense representation:\n", tensor.to_dense())
