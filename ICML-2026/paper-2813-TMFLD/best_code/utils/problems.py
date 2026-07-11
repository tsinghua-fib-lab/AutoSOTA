from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
import jax
from typing import NamedTuple
from jaxtyping import Array 
from utils.kernel import Distribution

@dataclass(frozen=True)
class Problem_nn:
    particle_d: int
    input_d: int
    output_d: int
    R1_prime: Callable[[Array], Array]
    q1: Callable[[Array], Array] = None
    q2: Callable[[Array, Array], Array] = None
    data: Optional[Array] = None


@dataclass(frozen=True)
class Problem_vlm:
    particle_d: int
    R1_prime: Callable[[Array], Array] = None
    q1: Callable[[Array], Array] = None
    q2: Callable[[Array, Array], Array] = None
    data : Optional[Array] = None


@dataclass(frozen=True)
class Problem_mmd_flow:
    particle_d: int
    R1_prime: Callable[[Array], Array] = None
    q1: Callable[[Array], Array] = None
    q2: Callable[[Array, Array], Array] = None
    data : Optional[Array] = None
    distribution: Distribution = None

