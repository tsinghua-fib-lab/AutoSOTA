"""
Pricing and dispatch ridesharing environments
"""
from dataclasses import field
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
from flax import linen as nn
import jax
from jax import Array
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from jaxtyping import Float, Integer, Bool
import funcy as f
import pooch
import numpy as np
import pandas as pd

from picard.nn import Policy, MLP


@struct.dataclass
class RideshareEvent:
    t: Integer[Array, "n_events"]
    src: Integer[Array, "n_events"]
    dest: Integer[Array, "n_events"]


@struct.dataclass
class EnvState(environment.EnvState):
    locations: Integer[
        Array, "n_cars"
    ]  # Ending point of the car's most recent trip
    times: Integer[Array, "n_cars"]  # Ending time of the car's most recent trip
    key: Integer[Array, "2"]
    event: RideshareEvent


@struct.dataclass
class EnvParams(environment.EnvParams):
    events: RideshareEvent = RideshareEvent(
        jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)
    )
    distances: Integer[Array, "nodes nodes"] = field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    n_cars: int = 1

    @property
    def n_nodes(self) -> int:
        return self.distances.shape[0]

    @property
    def n_events(self) -> int:
        return self.events.t.shape[0]


@struct.dataclass
class PricingEnvParams(environment.EnvParams):
    dispatch_env_params: EnvParams = field(default_factory=lambda: EnvParams())
    w_price: float = -1.0
    w_eta: float = -1.0
    w_intercept: float = 1.0

    @property
    def events(self) -> RideshareEvent:
        return self.dispatch_env_params.events

    @property
    def distances(self) -> Integer[Array, "nodes nodes"]:
        return self.dispatch_env_params.distances

    @property
    def n_cars(self) -> int:
        return self.dispatch_env_params.n_cars


def get_nth_event(event: RideshareEvent, n: int) -> RideshareEvent:
    return RideshareEvent(event.t[n], event.src[n], event.dest[n])


@partial(jax.jit, static_argnums=(0,))
def obs_to_state(n_cars: int, obs: Integer[Array, "o_dim"]):
    obs = obs.astype(int)
    locations = obs[3 : 3 + n_cars]
    times = obs[3 + n_cars :]
    event = RideshareEvent(obs[0], obs[1], obs[2])
    return event, locations, times


class RideshareDispatch(environment.Environment[EnvState, EnvParams]):
    def __init__(
        self, n_cars: int = 100, n_nodes: int = 100, n_events: int = 1000
    ):
        super(RideshareDispatch, self).__init__()
        self.n_cars = n_cars
        self.n_nodes = n_nodes
        self.n_events = n_events

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        return jax.lax.cond(
            action >= 0,
            lambda: self.step_env_dispatch(key, state, action, params),
            lambda: self.step_env_unfulfill(key, state, action, params),
        )

    def step_env_unfulfill(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        next_event = get_nth_event(params.events, state.time + 1)
        next_state = EnvState(
            time=state.time + 1,
            locations=state.locations,
            times=state.times,
            key=state.key,
            event=next_event,
        )
        done = self.is_terminal(next_state, params)
        reward = 0.0  # Should maybe have some unfulfill cost
        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            jnp.array(reward, dtype=float),
            done,
            {"discount": self.discount(state, params)},
        )

    def step_env_dispatch(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        t_pickup, _ = self.get_pickup_and_dropoff_times(state, action, params)
        next_state = self.dispatch_and_update_state(state, action, params)
        done = self.is_terminal(next_state, params)
        reward = -(t_pickup - state.event.t)  # Minimize rider wait times

        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            jnp.array(reward, dtype=float),
            done,
            {"discount": self.discount(state, params)},
        )

    @staticmethod
    def get_pickup_and_dropoff_times(
        state: EnvState, car_id: int, params: EnvParams
    ) -> Tuple[int, int]:
        """
        Compute the pickup and dropoff times for dispatching a `car_id`
        to serve `state.event`.
        """
        event = state.event
        t_pickup = (
            jnp.maximum(event.t, state.times[car_id])
            + params.distances[state.locations[car_id], event.src]
        )
        t_dropoff = t_pickup + params.distances[event.src, event.dest]
        return t_pickup, t_dropoff

    def dispatch_and_update_state(
        self, state: EnvState, car_id: int, params: EnvParams
    ) -> EnvState:
        event = state.event
        t_pickup, t_dropoff = self.get_pickup_and_dropoff_times(
            state, car_id, params
        )
        # is_accept = params.choice_model(choice_key, event, price, etd - event.t)
        new_locations = state.locations.at[car_id].set(event.dest)
        new_times = state.times.at[car_id].set(t_dropoff)
        next_event = get_nth_event(params.events, state.time + 1)
        next_state = EnvState(
            time=state.time + 1,
            locations=new_locations,
            times=new_times,
            key=state.key,
            event=next_event,
        )
        return next_state

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_reset = jax.random.split(key)
        state = EnvState(
            time=0,
            # Random init locations
            locations=jax.random.choice(
                key_reset, jnp.arange(self.n_nodes), (self.n_cars,)
            ),
            times=jnp.zeros(self.n_cars, dtype=int),  # Empty cars
            key=key,
            event=get_nth_event(params.events, 0),
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        return jnp.concatenate(
            [
                jnp.reshape(state.event.t, (1,)),
                jnp.reshape(state.event.src, (1,)),
                jnp.reshape(state.event.dest, (1,)),
                state.locations,
                state.times,
            ]
        )

    def is_terminal(self, state: EnvState, params=None) -> jnp.ndarray:
        """Check whether state is terminal."""
        return state.time >= self.n_events

    @property
    def name(self) -> str:
        """Environment name."""
        return "RideshareDispatch-v0"

    @property
    def num_actions(self) -> int:
        return self.n_cars

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(params.n_cars)

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        """Observation space of the environment."""
        return spaces.Box(0, jnp.inf, (3 + 2 * self.n_cars), jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "locations": spaces.Box(
                    0, params.n_nodes, (self.n_cars,), jnp.int32
                ),
                "times": spaces.Box(
                    0, params.events.t[-1], (self.n_cars,), jnp.int32
                ),
                "event": spaces.Dict(
                    {
                        "t": spaces.Box(
                            0, params.events.t[-1], (1,), jnp.int32
                        ),
                        "src": spaces.Box(0, self.n_nodes, (1,), jnp.int32),
                        "dest": spaces.Box(0, self.n_nodes, (1,), jnp.int32),
                    }
                ),
            }
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            events=RideshareEvent(
                jnp.arange(self.n_events),
                jax.random.choice(
                    jax.random.PRNGKey(0),
                    jnp.arange(self.n_nodes),
                    (self.n_events,),
                ),
                jax.random.choice(
                    jax.random.PRNGKey(1),
                    jnp.arange(self.n_nodes),
                    (self.n_events,),
                ),
            ),
            distances=jax.random.normal(
                jax.random.PRNGKey(2), (self.n_nodes, self.n_nodes)
            ),
            n_cars=self.n_cars,
        )


class RidesharePricing(RideshareDispatch):
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ):
        etas = (
            jnp.maximum(state.event.t, state.times)
            + params.distances[state.locations, state.event.src]
            - state.event.t
        )
        etds = params.distances[state.event.src, state.event.dest] + etas
        min_etd = jnp.min(etds)
        # Assume greedy dispatch
        best_car = jnp.argmin(etds)
        eta = etas[best_car]

        # Simple choice model: based on price per unit time,
        # and estimated time to pickup
        logit = (
            params.w_price * action  # / min_etd
            + params.w_intercept
            + params.w_eta * eta
        )
        p_accept = jnp.exp(logit) / (1 + jnp.exp(logit))
        accept_key, step_key = jax.random.split(key)
        accept = jax.random.bernoulli(accept_key, p_accept)

        reward = jax.lax.cond(
            accept,
            lambda: action,
            lambda: 0.0,
        )

        # Greedy dispatch
        next_obs, next_state, _, done, info = jax.lax.cond(
            accept,
            lambda: super(RidesharePricing, self).step_env(
                key, state, best_car, params.dispatch_env_params
            ),
            lambda: super(RidesharePricing, self).step_env(
                key, state, -1, params.dispatch_env_params
            ),  # unfulfilled
        )

        new_info = {
            "accept": accept,
            "p_accept": p_accept,
            "etd": min_etd,
            "eta": eta,
            **info,
        }

        return next_obs, next_state, reward, done, new_info

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        return super().reset_env(key, params.dispatch_env_params)


class ManhattanRideshareDispatch(RideshareDispatch):
    def __init__(self, n_cars=10000, n_events=100000):
        super().__init__(n_cars=n_cars, n_nodes=4333, n_events=n_events)

    @property
    def name(self) -> str:
        """Environment name."""
        return "RideshareDispatch-v0"

    @property
    def default_params(self) -> EnvParams:
        root = "https://github.com/atzheng/nyc-taxi-simulator-data/releases/download/initial-release"

        events_fname = pooch.retrieve(
            f"{root}/manhattan-trips.parquet",
            known_hash="md5:653f0d7d28348a3e998fdb38ef00ef47",
        )
        raw_events = pd.read_parquet(events_fname).head(self.n_events)
        distance_matrix_fname = pooch.retrieve(
            f"{root}/manhattan-distances.npy",
            known_hash="md5:95fda63cbed95bdb094f3b76baa7c7b4",
        )
        distances_np = np.load(distance_matrix_fname)
        # distances_np[distances_np == 0] = np.inf
        # distances_np = distances_np * (1 - np.eye(distances_np.shape[0]))
        distances = jnp.asarray(np.round(distances_np), dtype=int)

        events = RideshareEvent(
            jnp.asarray(raw_events["t"].values - raw_events["t"].values.min()),
            jnp.asarray(raw_events["pickup_idx"].values),
            jnp.asarray(raw_events["dropoff_idx"].values),
        )
        return EnvParams(events=events, distances=distances, n_cars=self.n_cars)


class ManhattanRidesharePricing(RidesharePricing):
    def __init__(self, n_cars=10000, n_events=100000):
        super().__init__(n_cars=n_cars, n_nodes=4333, n_events=n_events)

    @property
    def name(self) -> str:
        """Environment name."""
        return "RidesharePricing-v0"

    @property
    def default_params(self) -> PricingEnvParams:
        root = "https://github.com/atzheng/nyc-taxi-simulator-data/releases/download/initial-release"

        events_fname = pooch.retrieve(
            f"{root}/manhattan-trips.parquet",
            known_hash="md5:653f0d7d28348a3e998fdb38ef00ef47",
        )
        raw_events = (
            pd.read_parquet(events_fname).sort_values("t").head(self.n_events)
        )

        # raw_events = pd.read_parquet(events_fname).sort_values("t")
        # # Drop very early years — keep only modern data
        # raw_events = raw_events[raw_events["t"] > 1600000000].head(self.n_events)


        # # 🔍 Add print statements here to inspect timestamps
        # print("Raw timestamps (first 10):", raw_events["t"].values[:10])
        # print("Timestamps after diff:", np.diff(raw_events["t"].values[:10]))

        distance_matrix_fname = pooch.retrieve(
            f"{root}/manhattan-distances.npy",
            known_hash="md5:95fda63cbed95bdb094f3b76baa7c7b4",
        )
        distances_np = np.load(distance_matrix_fname)
        distances = jnp.asarray(np.round(distances_np), dtype=int)

        events = RideshareEvent(
            jnp.asarray(raw_events["t"].values - raw_events["t"].values.min()),
            jnp.asarray(raw_events["pickup_idx"].values),
            jnp.asarray(raw_events["dropoff_idx"].values),
        )
        return PricingEnvParams(
            dispatch_env_params=EnvParams(
                events=events, distances=distances, n_cars=self.n_cars
            ),
            w_price=-1.0,
            w_intercept=1.0,
            w_eta=-1.0,
        )


@struct.dataclass
class GreedyPolicy(Policy):
    """
    A simple greedy policy that selects the car with the lowest
    estimated time of arrival (ETA) to the pickup location.
    """

    n_cars: int
    temperature: float

    @partial(jax.jit, static_argnums=(0,))
    def apply(
        self,
        env_params: EnvParams,
        nn_params: Dict,
        obs: Integer[Array, "o_dim"],
        rng: chex.PRNGKey,
    ):
        event, locations, times = obs_to_state(self.n_cars, obs)
        rng, cost_rng = jax.random.split(rng)
        rewards = (
            -self.get_costs(
                env_params, cost_rng, event, locations, times, nn_params
            )
            # Don't dispatch unreachable nodes
            - (env_params.distances[locations, event.src] < 0) * jnp.inf
        )

        action = jax.random.choice(
            rng,
            jnp.arange(self.n_cars),
            p=jnp.exp((rewards - jnp.max(rewards)) / self.temperature),
        )
        return action, {}

    def get_costs(
        self,
        env_params: EnvParams,
        rng: chex.PRNGKey,
        event: RideshareEvent,
        locations: Integer[Array, "n_cars"],
        times: Integer[Array, "n_cars"],
        params: EnvParams,
    ):
        # jax.debug.print(f"{event.t}")
        # jax.debug.print(f"{times}")
        etas = (
            jnp.maximum(event.t, times)
            + env_params.distances[locations, event.src]
        )
        return etas

    def init(self, env_params, *args, **kwargs):
        return env_params


@struct.dataclass
class SimplePricingPolicy(Policy):
    n_cars: int
    price_per_distance: float

    def apply(
        self,
        env_params: EnvParams,
        nn_params: Dict,
        obs: Integer[Array, "o_dim"],
        rng: chex.PRNGKey,
    ):
        event, locations, times = obs_to_state(self.n_cars, obs)
        params = env_params.dispatch_env_params
        # Charge on trip time

        # # Greedy ETD
        # min_etd = jnp.min(
        #     jnp.maximum(event.t, times)
        #     + params.distances[locations, event.src]
        #     + params.distances[event.src, event.dest]
        # )
        return (
            self.price_per_distance * params.distances[event.src, event.dest],
            {},
        )


@struct.dataclass
class ValueGreedyPolicy(GreedyPolicy):
    """
    Uses a value function approximator to compute long term costs,
    then selects the car with the lowest cost.
    """

    nn: nn.Module
    gamma: float

    def obs_to_post_states(
        self, obs: Integer[Array, "o_dim"], env_params: EnvParams
    ):
        event, locations, times = obs_to_state(self.n_cars, obs)
        post_time_deltas = jnp.maximum(
            0,
            jnp.maximum(event.t, times)
            + env_params.distances[locations, event.src]
            + env_params.distances[event.src, event.dest]
            - event.t,
        )
        # post_states = jnp.vstack(
        #     [locations, times + post_time_deltas]
        # ).transpose()
        post_states = jnp.vstack(
            [locations, times + post_time_deltas]
        ).transpose()[0]
        return post_states

    def get_costs(
        self,
        env_params: EnvParams,
        rng: chex.PRNGKey,
        event: RideshareEvent,
        locations: Integer[Array, "n_cars"],
        times: Integer[Array, "n_cars"],
        nn_params: Dict,
    ):
        post_time_deltas = jnp.maximum(
            0,
            jnp.maximum(event.t, times)
            + env_params.distances[locations, event.src]
            + env_params.distances[event.src, event.dest]
            - event.t,
        )
        post_states = jnp.vstack(
            [locations, times + post_time_deltas]
        ).transpose()
        # TODO Should use first-order approximation for post-values
        post_values = self.nn.apply(nn_params, post_states[0]).reshape(-1)
        costs = super().get_costs(
            env_params, rng, event, locations, times, nn_params
        )
        return costs - self.gamma * post_values

    def init(self, env_params, rng, obs):
        return self.nn.init(rng, self.obs_to_post_states(obs, env_params))


class RideshareValueNetwork(MLP):
    @classmethod
    def from_env(cls, env, env_params, **kwargs):
        return cls(num_output_units=1, **kwargs)


if __name__ == "__main__":
    n_events = 100
    key = jax.random.PRNGKey(0)
    env = ManhattanRidesharePricing(n_cars=10000, n_events=n_events)
    env_params = env.default_params
    print(env_params)
    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.1)
    obs, state = env.reset(key, env_params)
    action, action_info = A.apply(env_params, dict(), obs, key)
    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)
