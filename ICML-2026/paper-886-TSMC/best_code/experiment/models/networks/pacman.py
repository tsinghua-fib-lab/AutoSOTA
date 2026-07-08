import jax
import jax.numpy as jnp

from ._jumanji_base import CNNPolicyNetwork, CNNValueNetwork, CNNQValueNetwork


def process_image(observation):
    """Process the `Observation` to be usable by the critic model.

    Args:
        observation: the observation as returned by the environment.

    Returns:
        rgb: a 2D, RGB image of the current observation.

    See:
    Code taken from jumanji (11-July-2024):
    https://github.com/instadeepai/jumanji/blob/main/jumanji/training/networks/pac_man/actor_critic.py
    """

    layer_1 = jnp.array(observation.grid) * 0.66
    layer_2 = jnp.array(observation.grid) * 0.0
    layer_3 = jnp.array(observation.grid) * 0.33
    player_loc = observation.player_locations
    ghost_pos = observation.ghost_locations
    pellets_loc = observation.power_up_locations
    is_scatter = observation.frightened_state_time
    idx = observation.pellet_locations

    # Pellets are light orange
    for i in range(len(idx)):
        if jnp.array(idx[i]).sum != 0:
            loc = idx[i]
            layer_3 = layer_3.at[loc[1], loc[0]].set(1)
            layer_2 = layer_2.at[loc[1], loc[0]].set(0.8)
            layer_1 = layer_1.at[loc[1], loc[0]].set(0.6)

    # Power pellet is purple
    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1 = layer_1.at[p[1], p[0]].set(0.5)
        layer_2 = layer_2.at[p[1], p[0]].set(0)
        layer_3 = layer_3.at[p[1], p[0]].set(0.5)

    # Set player is yellow
    layer_1 = layer_1.at[player_loc.x, player_loc.y].set(1)
    layer_2 = layer_2.at[player_loc.x, player_loc.y].set(1)
    layer_3 = layer_3.at[player_loc.x, player_loc.y].set(0)

    cr = jnp.array([1, 1, 0, 1])
    cg = jnp.array([0, 0.7, 1, 0.7])
    cb = jnp.array([0, 1, 1, 0.35])

    layers = (layer_1, layer_2, layer_3)
    scatter = 1 * (is_scatter / 60)

    def set_ghost_colours(
        layers
    ):
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            layer_1 = layer_1.at[x, y].set(cr[0])
            layer_2 = layer_2.at[x, y].set(cg[0] + scatter)
            layer_3 = layer_3.at[x, y].set(cb[0] + scatter)
        return layer_1, layer_2, layer_3

    layers = set_ghost_colours(layers)
    layer_1, layer_2, layer_3 = layers
    layer_1 = layer_1.at[0, 0].set(0)
    layer_2 = layer_2.at[0, 0].set(0)
    layer_3 = layer_3.at[0, 0].set(0)
    obs = [layer_1, layer_2, layer_3]
    rgb = jnp.stack(obs, axis=-1)

    return rgb


def process_metadata(x):
    # Get player position, scatter_time and ghost locations
    player_pos = jnp.array([x.player_locations.x, x.player_locations.y])
    scatter_time = x.frightened_state_time / 60
    scatter_time = jnp.expand_dims(scatter_time, axis=-1)
    ghost_locations_x = x.ghost_locations[..., 0]
    ghost_locations_y = x.ghost_locations[..., 1]

    return player_pos, scatter_time, ghost_locations_x, ghost_locations_y


class PreProcessPacman:

    def preprocess(
            self, obs: tuple[jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:

        # Pre-Process input
        image = process_image(obs)
        metadata = process_metadata(obs)

        return image, metadata


class PolicyNetwork[Action](PreProcessPacman, CNNPolicyNetwork):
    output_size: int = 4  # PacMan-v0 defaults
    output_sizes: tuple[int, int, int] = (4, )
    continuous: bool = False

    def enumerate_atoms(self) -> jax.Array | None:
        return jnp.arange(self.output_size)


class ValueNetwork(PreProcessPacman, CNNValueNetwork):
    ...


class QValueNetwork(PreProcessPacman, CNNQValueNetwork):
    output_size: int = 4  # PacMan-v0 defaults
    output_sizes: tuple[int, int, int] = (4, )
