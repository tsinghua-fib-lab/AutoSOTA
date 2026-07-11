# MIT License
#
# Copyright (c) 2025 tummfm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import haiku as hk

import jax.numpy as jnp

from . import ScaleShiftLayer


class AtomicEnergyLayer(hk.Module):
    """Adds atomic energy to the model."""

    def __init__(self,
                 num_species: int,
                 scale: float = 1.0,
                 shift: float = 0.0,
                 species_shift: bool=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.species_shift = species_shift
        self.scale_shift = ScaleShiftLayer(scale=scale, shift=shift)
        self.atomic_energies = hk.Embed(
            num_species, 1,
            embedding_matrix=jnp.zeros((num_species, 1)),
        )

    def __call__(self, per_atom_energies, species):
        if self.species_shift:
            atomic_energies = self.atomic_energies(species).squeeze(axis=1)
        else:
            atomic_energies = jnp.zeros_like(species, dtype=jnp.float32)
        return atomic_energies + self.scale_shift(per_atom_energies)
