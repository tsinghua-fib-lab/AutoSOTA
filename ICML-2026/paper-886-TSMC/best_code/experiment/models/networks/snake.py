import jax

from .. import blocks
from ._jumanji_base import CNNPolicyNetwork, CNNValueNetwork, CNNQValueNetwork


class PreProcessSnake:

    def preprocess(
            self, obs: tuple[jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array]]:
        grid, step_count, mask = obs

        # Binary encoding of unbounded int32
        bit_code = blocks.binary_encoding(step_count)

        return grid, (bit_code, )


class PolicyNetwork(PreProcessSnake, CNNPolicyNetwork):
    output_size: int = 4  # Snake-v1 defaults
    output_sizes: tuple[int, int, int] = (4, )
    continuous: bool = False

    def enumerate_atoms(self) -> jax.Array | None:
        return jax.numpy.arange(self.output_size)


class ValueNetwork(PreProcessSnake, CNNValueNetwork):
    ...


class QValueNetwork(PreProcessSnake, CNNQValueNetwork):
    output_size: int = 4  # Snake-v1 defaults
    output_sizes: tuple[int, int, int] = (4, )
