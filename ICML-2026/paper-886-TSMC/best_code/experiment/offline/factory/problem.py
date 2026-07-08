"""This module implements the factory for environments and dependent functions


"""
from typing import Any, Callable
from typing_extensions import Self

import jax.numpy as jnp

import jit_env

from experiment import env
from experiment.src.types import MetricData

from ..utils import unpack_nested


def prefix_test_metrics(
        f: Callable[[int, dict[str, Any]], MetricData]
) -> Callable[[int, dict[str, Any]], MetricData]:
    # Wrapper for common naming prefix (to prevent typos)
    def metric_fun(step: int, trajectory: dict[str, Any]) -> MetricData:
        result = f(step, trajectory)
        return {f'eval_data/{k}': v for k, v in result.items()}

    return metric_fun


def prefix_train_metrics(
        f: Callable[[int, dict[str, Any]], MetricData]
) -> Callable[[int, dict[str, Any]], MetricData]:
    # Wrapper for common naming prefix (to prevent typos)
    def metric_fun(step: int, trajectory: dict[str, Any]) -> MetricData:
        result = f(step, trajectory)
        return {f'train_data/{k}': v for k, v in result.items()}

    return metric_fun


@prefix_test_metrics
def test_metrics_generic(
        step: int,
        output: dict[str, Any]
) -> dict[str, Any]:
    returns = output['returns']
    lengths = output['episode_lengths']

    prefix_fun = [('avg', jnp.mean), ('var', jnp.var),
                  ('min', jnp.min), ('max', jnp.max)]
    metrics = {
        f'{prefix}_episode_return': f(returns)
        for prefix, f in prefix_fun
    } | {
        f'{prefix}_episode_length': f(lengths)
        for prefix, f in prefix_fun
    } | {'num_interactions': output['data_size'] * step}

    return metrics


@prefix_train_metrics
def training_metrics_generic(
        step: int,
        timesteps: dict[str, Any]
) -> dict[str, Any]:
    # Compute metrics from the most recent batch of pre-processed data
    steps = timesteps

    # averaged over batch
    total_reward = jnp.mean(steps.reward.sum(axis=-1))
    num_terminals = jnp.mean(jnp.isclose(steps.discount, 0.0).sum(axis=-1))

    # Counts number of (auto)-resets/ terminations
    avg_reward = jnp.mean(
        steps.reward.sum(axis=-1) /
        (1 + (steps.step_type != jit_env.StepType.MID).sum(axis=-1)),
    )

    data_size = steps.reward.shape[0] * steps.reward.shape[1]
    metrics = {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'num_terminals': num_terminals,
        'num_interactions': data_size * step,

    }

    return metrics


@prefix_train_metrics
def training_metrics_pgx(
        step: int,
        timesteps: dict[str, Any]
) -> dict[str, Any]:
    # Compute metrics from the most recent batch of pre-processed data
    steps = timesteps

    # averaged over batch
    if len(steps.reward.shape) > 2:
        # Boardgames has 2D rewards
        total_reward_p1 = jnp.mean(steps.reward.sum(axis=-1)[..., 0])
        total_reward_p2 = jnp.mean(steps.reward.sum(axis=-1)[..., 1])
    else:
        total_reward_p1 = jnp.mean(steps.reward.sum(axis=-1))
        total_reward_p2 = 0

    data_size = steps.reward.shape[0] * steps.reward.shape[1]
    metrics = {
        'total_reward_p1': total_reward_p1,
        'total_reward_p2': total_reward_p2,
        'num_interactions': data_size * step
    }

    return metrics


@prefix_train_metrics
def training_metrics_gridworld(
        step: int,
        trajectory: dict[str, Any]
) -> dict[str, Any]:
    # Compute metrics from the most recent batch of pre-processed data
    steps = trajectory['steps']

    # averaged over batch
    total_reward = jnp.mean(steps.reward.sum(axis=-1))
    total_regret = jnp.mean(steps.extras['regret'].sum(axis=-1))
    num_terminals = jnp.mean(jnp.isclose(steps.discount, 0.0).sum(axis=-1))

    # Counts number of (auto)-resets/ terminations
    avg_reward = jnp.mean(
        steps.reward.sum(axis=-1) /
        (1 + (steps.step_type != jit_env.StepType.MID).sum(axis=-1)),
    )

    data_size = steps.reward.shape[0] * steps.reward.shape[1]
    metrics = {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'num_interactions': data_size * step,
        'num_terminals': num_terminals,
        'regret': total_regret
    }

    return metrics


class ProblemBuilder:
    """Builder for the environments and the training-metrics function.

    We instantiate multiple copies for the same environment to be used
    differently by the agent, trainer, and evaluator.

    This is to separate different wrappers and utilities from each component
    of a reinforcement learning experiment. For example, our SMC or MCTS
    planner should not autoreset environments when ending up in terminal
    states. But when vectorizing environments for data generation, we would
    like to add this to ensure that all data shapes are homogenous.

    Finally, for the evaluation environment, we can extend this in many
    different ways to test offline performance.

    We also build the function for collecting training-metrics here, since
    environments over different implementation modules generate different
    data. This builder then has convenient context on how to format this.
    """

    def __init__(self):
        # Keep copies of the same environment to be used differently
        self.name: str | None = None
        self.datagen_env: jit_env.Environment | None = None
        self.planner_env: jit_env.Environment | None = None
        self.eval_env: jit_env.Environment | None = None

        self.train_metric_fun: Callable[
                             [int, dict[str, Any]], MetricData
                         ] | None = None

        self.test_metric_fun: Callable[
                             [int, dict[str, Any]], MetricData
                         ] | None = None

    def format_eval_env(self, max_length: int) -> jit_env.Environment:

        if self.eval_env is None:
            raise RuntimeError("Call `ProblemBuilder.build` first!")

        eval_env = env.TimeoutWrapper(self.eval_env, max_length, True)
        eval_env = env.AddObservationToState(eval_env)

        return eval_env

    def _setup_square_grid(self, name: str, config: dict[str, Any]):
        self.datagen_env = env.SquareGrid(**config)
        self.planner_env = env.SquareGrid(**config)
        self.eval_env = env.SquareGrid(**config)

        self.train_metric_fun = training_metrics_gridworld
        self.test_metric_fun = test_metrics_generic

    def _setup_brax(self, name: str, config: dict[str, Any]):
        from brax import envs as brax_envs

        if name not in brax_envs._envs:
            raise NotImplementedError(
                f"Brax does not support option {name}. "
                f"Choose from: {brax_envs._envs}"
            )

        self.datagen_env = env.BraxWrapper(brax_envs.create(
            name, **config,
            episode_length=None, auto_reset=False  # type: ignore
        ))
        self.planner_env = env.BraxWrapper(brax_envs.create(
            name, **config,
            episode_length=None, auto_reset=False  # type: ignore
        ))
        self.eval_env = env.BraxWrapper(brax_envs.create(
            name, **config,
            episode_length=None, auto_reset=False  # type: ignore
        ))

        self.train_metric_fun = training_metrics_generic
        self.test_metric_fun = test_metrics_generic

    def _setup_jumanji(self, name: str, config: dict[str, Any]):
        import jumanji
        self.datagen_env = env.JumanjiWrapper(jumanji.make(name, **config))
        self.planner_env = env.JumanjiWrapper(jumanji.make(name, **config))
        self.eval_env = env.JumanjiWrapper(jumanji.make(name, **config))

        self.train_metric_fun = training_metrics_generic
        self.test_metric_fun = test_metrics_generic

    def _setup_pgx(self, name: str, config: dict[str, Any]):
        import pgx
        self.datagen_env = env.PGXWrapper(pgx.make(name))  # type: ignore
        self.planner_env = env.PGXWrapper(pgx.make(name))  # type: ignore
        self.eval_env = env.PGXWrapper(pgx.make(name))  # type: ignore

        self.train_metric_fun = training_metrics_pgx
        self.test_metric_fun = test_metrics_generic

    def build(self, config: dict[str, Any]) -> Self:
        env_kwargs = unpack_nested(config, 'kwargs')

        self.name = unpack_nested(config, 'name')
        match self.name.split(' '):
            case ['SquareGrid']: self._setup_square_grid('', env_kwargs)
            case ['brax', s]: self._setup_brax(s, env_kwargs)
            case ['jumanji', s]: self._setup_jumanji(s, env_kwargs)
            case ['pgx', s]: self._setup_pgx(s, env_kwargs)
            case _: raise ValueError(f'Unsupported env config: {config}')

        self.datagen_env, self.planner_env = env.make_smz_compat(
            self.datagen_env, self.planner_env,
            truncation_length=unpack_nested(config, 'truncation_length')
        )

        return self
