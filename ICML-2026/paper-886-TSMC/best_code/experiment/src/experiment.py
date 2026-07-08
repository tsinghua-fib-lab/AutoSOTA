from typing import Callable, Any

import os
import pickle

import jax
import optax
import flashbax as fbx

from jaxtyping import PRNGKeyArray

import tqdm

import jit_env

from .datagen import EnvironmentDataGenerator
from .types import Learner, Logger, MetricData, Identity


def snapshot(step: int, variables, eval_metrics: dict[str, Any] | None) -> bool:
    path = Identity().make_path()

    out_folder = os.path.join(path, 'models')

    try:
        os.makedirs(out_folder, exist_ok=True)
    except PermissionError:
        print('Permission error! Cannot creat folder for model snapshots')
        return False

    try:
        with open(os.path.join(out_folder, f'model_{step}.out'), 'wb') as f:
            pickle.dump({'variables': variables, 'metrics': eval_metrics}, f)
    except Exception as e:
        print(e)
        return False

    return True


def run_experiment[Action, State, Observation](
        key: PRNGKeyArray,
        data_gen: EnvironmentDataGenerator[
            optax.Params, Any, State, Observation, Action
        ],
        buffer: fbx.trajectory_buffer.TrajectoryBuffer,

        learner: Learner[optax.OptState, optax.Params, Any],
        preprocess_fun: Callable[[
            jit_env.TimeStep[Observation, float, float, int],
            Action,
            Any
        ], dict[str, Any]],
        evaluate_fun: Callable[[int, int, PRNGKeyArray, optax.Params], MetricData],
        metric_fun: Callable[[int, dict[str, Any]], MetricData],
        logger: Logger,
        *,
        num_iterations: int,
        eval_period: int,
        snapshot_period: int | None = None,
        eager_eval: bool = False,
        start: int = 0,
        use_pbar: bool = True,
        _raise: bool = False
) -> tuple[PRNGKeyArray, int, tuple[optax.OptState, optax.Params]]:
    """Do not jit! Do not vmap! Only returns variables needed for resuming

    Runs a synchronous loop:
     1) collect-data with agent
     2) preprocess this data
     3) update agent parameters with current storage
    to do an offline reinforcement learning experiment.
    """
    if snapshot_period is not None:
        if not ((snapshot_period >= eval_period) and
                (snapshot_period % eval_period == 0)):
            print('Warning! snapshot period is not adjusted '
                  'for evaluation period')

    # Initialize experiment state
    carry, key_init_leaner = jax.random.split(key)
    learner_state, params = learner.init(key_init_leaner)
    generator_state = buffer_state = None
    evaluate_queue: dict[int, optax.Params] = {}

    # Modify on device data in-place if possible (prevent copying of data)
    buffer_update_fun = jax.jit(buffer.add, donate_argnums=0)
    param_update_fun = jax.jit(learner.update, donate_argnums=(0, 1))
    param_update_fun_eval = jax.jit(learner.update, donate_argnums=0)
    param_update_fun = param_update_fun_eval  # TODO fix, I have no clue tho

    pbar = tqdm.trange(
        start, start + num_iterations, desc='train'
    ) if use_pbar else range(start, start + num_iterations)

    for i in pbar:
        carry, key_data, key_eval, key_update = jax.random.split(carry, num=4)

        # Generate data
        generator_state, (steps, actions, policy_data) = data_gen.sample_data(
            key_data, params, generator_state
        )
        trajectory_batch = preprocess_fun(steps, actions, policy_data)

        # Add to buffer
        if buffer_state is None:
            # Note; Buffer must be initialized with unbatched data
            buffer_state = buffer.init(
                jax.tree_util.tree_map(lambda x: x[0, 0], trajectory_batch)
            )

        buffer_state = buffer_update_fun(buffer_state, trajectory_batch)

        # Update agent parameters and flag periodically for offline evaluation
        eval_flagged = i % eval_period == 0
        if eval_flagged:
            evaluate_queue[i] = params

        eval_metrics = {}
        if eval_flagged and eager_eval:
            # If specified evaluate iterates immediately
            eval_metrics = evaluate_fun(i, steps.reward.size, key_eval, evaluate_queue.pop(i))

        # Save model weights if necessary
        if snapshot_period is not None:
            if i % snapshot_period == 0 or i == start + num_iterations - 1:
                pbar.write(
                    f'[Step {i}] Saving model snapshot...'
                )

                _dct = eval_metrics if eval_flagged and eager_eval else None
                success = snapshot(i, params, _dct)

                if _raise and (not success):
                    raise RuntimeError("Model snapshotting failed! Exiting")

        can_sample = buffer.can_sample(buffer_state)
        learner_metrics = {}
        if eval_flagged and can_sample and (not eager_eval):
            # Do not donate `params` when doing offline evaluation
            (learner_state, params), learner_metrics = param_update_fun_eval(
                learner_state, params, key_update, buffer_state
            )
        elif eval_flagged and (not can_sample):
            # Ensure buffer is not overridden in the next loop iteration.
            (learner_state, params) = jax.tree_util.tree_map(jax.numpy.copy,
                                                 (learner_state, params))
        elif can_sample:
            # Donates learner state and params
            (learner_state, params), learner_metrics = param_update_fun(
                learner_state, params, key_update, buffer_state
            )

        # Track progress
        data_metrics: dict[str, Any] = metric_fun(i, steps)
        logger.log(eval_metrics | learner_metrics | data_metrics | {'step': i})

    # Add the last parameter-set if specified.
    if num_iterations % eval_period == 0:
        evaluate_queue[num_iterations] = params

    # Run evaluation suite  (if non-empty)
    iterator = tqdm.tqdm(
        evaluate_queue.items(), desc='eval'
    ) if use_pbar else evaluate_queue.items()

    for it, p in iterator:
        carry, key = jax.random.split(carry)

        metrics = evaluate_fun(it, steps.reward.size, key, p)
        logger.log(metrics | {'step': it})

    return carry, num_iterations, (learner_state, params)
