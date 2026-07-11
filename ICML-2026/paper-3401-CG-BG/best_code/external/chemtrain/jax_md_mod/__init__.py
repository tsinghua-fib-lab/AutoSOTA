# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

def uncache(exclude):
  """Remove package modules from cache except excluded ones.
  On next import they will be reloaded.

  Args:
      exclude (iter<str>): Sequence of module paths.
  """
  pkgs = []
  for mod in exclude:
    pkg = mod.split('.', 1)[0]
    pkgs.append(pkg)

  to_uncache = []
  for mod in sys.modules:
    if mod in exclude:
      continue

    if mod in pkgs:
      to_uncache.append(mod)
      continue

    for pkg in pkgs:
      if mod.startswith(pkg + '.'):
        to_uncache.append(mod)
        break

  for mod in to_uncache:
    del sys.modules[mod]


import jax
from jax import Array
import jax.numpy as jnp

# Fix jax_md error
jax.random.KeyArray = jax.Array
uncache("jax.random")

import jax_md.partition
from jax_md import dataclasses

from jax import random

def is_box_valid(box: Array) -> bool:
  if jnp.isscalar(box) or box.ndim == 0 or box.ndim == 1:
    return True
  if box.ndim == 2:
    return jnp.bool_(jnp.logical_not(jnp.all(jnp.triu(box) == box)))
  return False

@dataclasses.dataclass
class PartitionError:
  """A struct containing error codes while building / updating neighbor lists.

  Attributes:
    code: An array storing the error code. See `PartitionErrorCode` for
      details.
  """
  code: Array

  def update(self, bit: bytes, pred: Array) -> Array:
    """Possibly adds an error based on a predicate."""
    zero = jnp.zeros((), jnp.uint8)
    bit = jnp.array(bit, dtype=jnp.uint8)
    return PartitionError(self.code | jnp.where(pred, bit, zero))

  def __str__(self) -> str:
    """Produces a string representation of the error code."""

    try:
      if not jnp.any(self.code):
        return ''
    except Exception as err:
      return f'Error code not available ({self.code})'

    if jnp.any(self.code & jax_md.partition.PEC.NEIGHBOR_LIST_OVERFLOW):
      return 'Partition Error: Neighbor list buffer overflow.'

    if jnp.any(self.code & jax_md.partition.PEC.CELL_LIST_OVERFLOW):
      return 'Partition Error: Cell list buffer overflow'

    if jnp.any(self.code & jax_md.partition.PEC.CELL_SIZE_TOO_SMALL):
      return 'Partition Error: Cell size too small'

    if jnp.any(self.code & jax_md.partition.PEC.MALFORMED_BOX):
      return ('Partition Error: Incorrect box format. Expecting upper '
              'triangular.')

    raise ValueError(f'Unexpected Error Code {self.code}.')

  __repr__ = __str__

jax_md.partition.is_box_valid = is_box_valid
jax_md.partition.PartitionError = PartitionError
random.KeyArray = jax.Array
uncache('jax_md.partition')
uncache('jax.random')

# Do some e3nn_jax modifications

import e3nn_jax._src.scatter
from jax_md_mod import e3nn_mod

e3nn_jax._src.scatter._distinct_but_small = e3nn_mod._distinct_but_small
uncache('e3nn_jax._src.scatter')
