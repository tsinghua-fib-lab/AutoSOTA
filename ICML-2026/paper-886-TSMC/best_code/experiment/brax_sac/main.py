"""RL training with an environment running entirely on an accelerator.

Simplified fork of (accessed August 2024):
https://github.com/google/brax/blob/main/brax/training/learner.py
"""

import functools

from absl import app
from absl import flags

from brax import envs
from brax.io import metrics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac


FLAGS = flags.FLAGS

flags.DEFINE_enum('learner', 'sac', ['ppo', 'sac'],
                  'Which algorithm to run.')
flags.DEFINE_string('env', 'ant', 'Name of environment to train.')

flags.DEFINE_enum(
    'backend',
    'spring',
    ['spring', 'generalized', 'positional'],
    'The physics backend to use.',
)
flags.DEFINE_bool('legacy_spring', False, 'Brax v1 backend.')
flags.DEFINE_integer('total_env_steps', 10_000_000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('num_evals', 200, 'How many times to run an eval.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_envs', 64, 'Number of envs to run in parallel.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('unroll_length', 30, 'Unroll length.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('num_minibatches', 1, 'Number')
flags.DEFINE_integer(
    'num_updates_per_batch', 1,
    'Number of times to reuse each transition for gradient '
    'computation.')
flags.DEFINE_float('reward_scaling', 10.0, 'Reward scale.')
flags.DEFINE_float('entropy_cost', 3e-4, 'Entropy cost.')
flags.DEFINE_integer('episode_length', 1000, 'Episode length.')
flags.DEFINE_float('discounting', 0.99, 'Discounting.')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_float('max_gradient_norm', 1e9,
                   'Maximal norm of a gradient update.')
flags.DEFINE_string('logdir', '', 'Logdir.')
flags.DEFINE_bool('normalize_observations', True,
                  'Whether to apply observation normalization.')
flags.DEFINE_integer(
    'max_devices_per_host', None,
    'Maximum number of devices to use per host. If None, '
    'defaults to use as much as it can.')
flags.DEFINE_integer('num_videos', 0,
                     'Number of videos to record after training.')
flags.DEFINE_integer('num_trajectories_npy', 0,
                     'Number of rollouts to write to disk as raw QP states.')

# SAC hps.
flags.DEFINE_integer('min_replay_size', 8192,
                     'Minimal replay buffer size before the training starts.')
flags.DEFINE_integer('max_replay_size', 1_048_576,
                     'Maximal replay buffer size.')
flags.DEFINE_integer(
    'grad_updates_per_step', 1,
    'How many SAC gradient updates to run per one step in the '
    'environment.')

# PPO hps.
flags.DEFINE_float('gae_lambda', .95, 'General advantage estimation lambda.')
flags.DEFINE_float('clipping_epsilon', .3, 'Policy loss clipping epsilon.')
flags.DEFINE_integer('num_resets_per_eval', 10, 'Number of resets per eval.')


def main(unused_argv):
    get_environment = functools.partial(
        envs.get_environment, backend=FLAGS.backend
    )

    with metrics.Writer(f'{FLAGS.logdir}/{FLAGS.env}_{FLAGS.seed}') as writer:
        writer.write_hparams({
            'num_evals': FLAGS.num_evals,
            'num_envs': FLAGS.num_envs,
            'total_env_steps': FLAGS.total_env_steps
        })
        if FLAGS.learner == 'sac':
            make_policy, params, _ = sac.train(
                environment=get_environment(FLAGS.env),
                num_envs=FLAGS.num_envs,
                action_repeat=FLAGS.action_repeat,
                normalize_observations=FLAGS.normalize_observations,
                num_timesteps=FLAGS.total_env_steps,
                num_evals=FLAGS.num_evals,
                batch_size=FLAGS.batch_size,
                min_replay_size=FLAGS.min_replay_size,
                max_replay_size=FLAGS.max_replay_size,
                learning_rate=FLAGS.learning_rate,
                discounting=FLAGS.discounting,
                max_devices_per_host=FLAGS.max_devices_per_host,
                seed=FLAGS.seed,
                reward_scaling=FLAGS.reward_scaling,
                grad_updates_per_step=FLAGS.grad_updates_per_step,
                episode_length=FLAGS.episode_length,
                progress_fn=writer.write_scalars)
        if FLAGS.learner == 'ppo':
            make_policy, params, _ = ppo.train(
                environment=get_environment(FLAGS.env),
                num_timesteps=FLAGS.total_env_steps,
                episode_length=FLAGS.episode_length,
                action_repeat=FLAGS.action_repeat,
                num_envs=FLAGS.num_envs,
                max_devices_per_host=FLAGS.max_devices_per_host,
                learning_rate=FLAGS.learning_rate,
                entropy_cost=FLAGS.entropy_cost,
                discounting=FLAGS.discounting,
                seed=FLAGS.seed,
                unroll_length=FLAGS.unroll_length,
                batch_size=FLAGS.batch_size,
                num_minibatches=FLAGS.num_minibatches,
                normalize_observations=FLAGS.normalize_observations,
                num_updates_per_batch=FLAGS.num_updates_per_batch,
                num_evals=FLAGS.num_evals,
                reward_scaling=FLAGS.reward_scaling,
                gae_lambda=FLAGS.gae_lambda,
                clipping_epsilon=FLAGS.clipping_epsilon,
                num_resets_per_eval=FLAGS.num_resets_per_eval,
                progress_fn=writer.write_scalars,
            )


if __name__ == '__main__':
    app.run(main)
