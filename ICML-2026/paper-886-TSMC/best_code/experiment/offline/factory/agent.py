"""This module implements the factory for the agent and its components


"""
from typing import Any
from typing_extensions import Self

import optax

from flax import traverse_util as ftu

import jit_env

from smz import smc
from smz.planning.impl import (
    CompositeAccumulator, RetraceAccumulator,
    LogweightAccumulator, DSMCAccumulator
)

import numpy as np

from experiment import models
from experiment.src.types import Policy, Learner

from ..impl import (
    LinenELBOTarget, LinenTRPIProposal, JitEnvSMCTransition,
    JitEnvSMCTransitionWithPRNGKey, # SMC-components
    MCTXComponents,
    SMCPolicy, MCTXPolicy, PPOPolicy,
    EMLearner, EMLearnerHyperparameters,
    PPOLearner, PPOLearnerHyperparameters,
)
from ..utils import unpack_nested
from smz.sh_smcts import SHSMC


def make_optimizer(config: dict[str, Any]) -> optax.GradientTransformation:
    """Factory function to parse a dict-config into an optax optimizer"""
    decay = optax.identity()

    name = unpack_nested(config, "name")
    if name == optax.adamw.__name__:
        optimizer = optax.adamw(
            learning_rate=unpack_nested(config, "learning_rate"),
            weight_decay=unpack_nested(config, "l2_weight_decay"),
        )

    else:
        optimizer = getattr(optax, name)(
            unpack_nested(config, "learning_rate")
        )

        l2 = unpack_nested(config, "l2_weight_decay")
        if l2 > 0.0:
            decay = optax.add_decayed_weights(l2)

    # Gradient transform to zero out subcontainers not in the FINETUNABLE scope
    finetune_mask = optax.identity()
    if unpack_nested(config, 'subset_of_weights'):
        mask_fn = lambda param: ftu.path_aware_map(
            lambda path, x: models.blocks.FINETUNABLE_PREFIX not in path, param
        )
        finetune_mask = optax.masked(optax.set_to_zero(), mask_fn)

    return optax.chain(
        optax.clip(unpack_nested(config, "max_grad")),
        optax.clip_by_global_norm(
            unpack_nested(config, "max_grad_norm")),
        optimizer,
        decay,
        finetune_mask  # Note; zero-mask always at the end
    )


class ModelBuilder:

    def __init__(self):
        self.model: models.JointModel | None = None

    def _setup_square_grid(
            self,
            name: str,
            config: dict[str, Any],
            io_spec: tuple[jit_env.specs.Spec, jit_env.specs.Spec]
    ):
        nets = models.networks.square_grid

        # Heuristic check to see whether the observation is one-hot-encoded.
        use_cnn = io_spec[0].generate_value().shape[-1] != 2
        size = io_spec[1].maximum + 1  # type: ignore

        self.model = models.JointModel(
            policy=nets.PolicyNetwork(output_size=size, use_cnn=use_cnn),
            value=nets.ValueNetwork(use_cnn=use_cnn),
            q_value=nets.QValueNetwork(output_size=size, use_cnn=use_cnn)
        )

    def _setup_brax(
            self,
            name: str,
            config: dict[str, Any],
            io_spec: tuple[jit_env.specs.Spec, jit_env.specs.Spec]
    ):
        nets = models.networks.brax

        size = sum(io_spec[1].shape)  # type: ignore

        self.model = models.JointModel(
            policy=nets.PolicyNetwork(
                output_size=size,
                option='tanh',
                mlp_kwargs=unpack_nested(config, 'mlp_kwargs'),
                bounds=(-1.0, 1.0)
            ),
            value=nets.ValueNetwork(
                mlp_kwargs=unpack_nested(config, 'mlp_kwargs'),
                value_transform=unpack_nested(config, "value_transform")
            ),
            q_value=nets.QValueNetwork(
                mlp_kwargs=unpack_nested(config, 'mlp_kwargs'),
                value_transform=unpack_nested(config, "value_transform")
            )
        )

    def _setup_jumanji(
            self,
            name: str,
            config: dict[str, Any],
            io_spec: tuple[jit_env.specs.Spec, jit_env.specs.Spec]
    ):
        if 'pacman' in name.lower():
            nets = models.networks.pacman
        elif 'snake' in name.lower():
            nets = models.networks.snake
        elif 'minesweeper' in name.lower():
            nets = models.networks.minesweeper
        elif 'rubiks' in name.lower():
            nets = models.networks.rubiks_cube
            num_values = io_spec[1].num_values  # type: ignore

            self.model = models.JointModel(
                policy=nets.PolicyNetwork(
                    mlp_kwargs=unpack_nested(config, "mlp_kwargs"),
                    output_size=np.prod(num_values),
                    output_sizes=np.asarray(num_values)
                ),
                value=nets.ValueNetwork(
                    mlp_kwargs=unpack_nested(config, "mlp_kwargs"),
                    value_transform=unpack_nested(config, "value_transform")
                ),
                q_value=nets.QValueNetwork(
                    mlp_kwargs=unpack_nested(config, "mlp_kwargs"),
                    value_transform=unpack_nested(config, "value_transform"),
                    output_size=np.prod(num_values),
                    output_sizes=np.asarray(num_values)
                )
            )
            return
        else:
            raise NotImplementedError(
                f"No networks implemented for Jumanji {name}"
            )

        self.model = models.JointModel(
            policy=nets.PolicyNetwork(
                conv_kwargs=unpack_nested(config, "conv_kwargs"),
                out_kwargs=unpack_nested(config, "out_kwargs")
            ),
            value=nets.ValueNetwork(
                conv_kwargs=unpack_nested(config, "conv_kwargs"),
                out_kwargs=unpack_nested(config, "out_kwargs"),
                value_transform=unpack_nested(config, "value_transform")
            ),
            q_value=nets.QValueNetwork(
                conv_kwargs=unpack_nested(config, "conv_kwargs"),
                out_kwargs=unpack_nested(config, "out_kwargs"),
                value_transform=unpack_nested(config, "value_transform")
            )
        )

    def _setup_pgx(
            self,
            name: str,
            config: dict[str, Any],
            io_spec: tuple[jit_env.specs.Spec, jit_env.specs.Spec]
    ):
        raise NotImplementedError('No networks implemented for PGX')

    def build(
            self,
            name: str,
            config: dict[str, Any],
            io_spec: tuple[jit_env.specs.Spec, jit_env.specs.Spec]
    ) -> Self:

        match name.split(' '):  # Map Environment name to compatible networks
            case ['SquareGrid']: self._setup_square_grid(name, config, io_spec)
            case ['brax', _]: self._setup_brax(name, config, io_spec)
            case ['jumanji', _]: self._setup_jumanji(name, config, io_spec)
            case ['pgx', _]: self._setup_pgx(name, config, io_spec)
            case _: raise ValueError(
                f'No supported networks/ models for env: {name}'
            )

        if self.model is None:
            raise RuntimeError(f"Model creation failed for {name}: {config}")

        return self


class AgentBuilder:

    def __init__(self):
        self.policy: Policy | None = None
        self.testing_baseline_smc_policy: Policy | None = None
        self.testing_smcts_policy: Policy | None = None

    def _setup_ppo(
            self,
            config: dict[str, Any],
            model: models.JointModel,
            env: jit_env.Environment
    ):
        if hasattr(model.policy, '_tanh_logprob_jacdet_correction'):
            model.policy._tanh_logprob_jacdet_correction[0] = False  # type: ignore

        self.policy = PPOPolicy(
            value_model=model.value,
            policy_model=model.policy,
            stochastic_eval=unpack_nested(config, 'stochastic_eval'),
        )

    def _setup_mctx(
            self,
            config: dict[str, Any],
            model: models.JointModel,
            env: jit_env.Environment
    ):

        components = MCTXComponents(
            value_model=model.value,
            policy_model=model.policy,
            transition=JitEnvSMCTransition(env),
            bootstrap=unpack_nested(config, 'bootstrap'),
        )

        self.policy = MCTXPolicy(
            components=components,
            budget=unpack_nested(config, "budget"),
            max_depth=unpack_nested(config, "max_depth"),
            max_breadth_root=unpack_nested(config, "max_breadth_root"),
            stochastic_eval=unpack_nested(config, 'stochastic_eval'),
        )

    def _setup_smc(
            self,
            config: dict[str, Any],
            model: models.JointModel,
            env: jit_env.Environment
    ):
        # Decompose config
        smc_cfg = config['smc']

        # Create the base SMC method
        proposal = LinenTRPIProposal(
            value_model=model.q_value,
            policy_model=model.policy,
            **unpack_nested(smc_cfg, "proposal_kwargs")
        )
        target = LinenELBOTarget(
            model.value, **unpack_nested(smc_cfg, "target_kwargs")
        )

        accumulator = CompositeAccumulator(
            ('logweight', LogweightAccumulator(False)),  # type: ignore
            ('retrace', RetraceAccumulator(  # type: ignore
                gamma=config['credit_assignment']['discount'],
                td_lambda=config['credit_assignment']['td_lambda'],
                return_data=False
            )),
            ('dsmc', DSMCAccumulator(  # type: ignore
                gamma=config['credit_assignment']['discount'],
                td_lambda=config['credit_assignment']['td_lambda'],
                return_data=False
            )),
            return_data=True
        )

        resample_fun = smc.multinomial_resampling
        if 'resampling_method' in smc_cfg:
            resample_method = unpack_nested(smc_cfg, 'resampling_method')
            if resample_method == 'multinomial':
                resample_fun = smc.multinomial_resampling
            elif resample_method == 'deterministic':
                resample_fun = smc.deterministic_resampling
            else:
                raise NotImplementedError(resample_method)

        planner = smc.SMC(
            proposal=proposal,
            transition=JitEnvSMCTransitionWithPRNGKey(env),
            target=target,  # type: ignore
            statistic_fun=accumulator,  # type: ignore
            **unpack_nested(smc_cfg, "kwargs"),
            resampling_method=resample_fun
        )

        if 'sh_smcts' in smc_cfg and unpack_nested(config, 'option') == 'dsmc':
            root_algorithm_type = unpack_nested(smc_cfg, 'sh_smcts')['root_planner']
        else:
            root_algorithm_type = "smc"

        if 'sh_smcts' in smc_cfg and root_algorithm_type == 'sh_smcts':
            self.policy = SHSMC(
                planner,
                num_actions_to_search=unpack_nested(smc_cfg, 'sh_smcts')['num_actions_to_search'],
                discount=config['credit_assignment']['discount'],
                td_lambda=config['credit_assignment']['td_lambda'],
                option=unpack_nested(config, 'option'),
                value_mixing=unpack_nested(config, 'value_mixing'),
                stochastic_eval=unpack_nested(config, 'stochastic_eval'),
                use_completed_q_values=unpack_nested(smc_cfg, 'sh_smcts')['use_completed_q_values'],
                use_q_transform=unpack_nested(smc_cfg, 'sh_smcts')['use_q_transform'],
            )
        else:
            # Construct Policy class given the planner
            self.policy = SMCPolicy(
                planner,
                discount=config['credit_assignment']['discount'],
                td_lambda=config['credit_assignment']['td_lambda'],
                option=unpack_nested(config, 'option'),
                value_mixing=unpack_nested(config, 'value_mixing'),
                stochastic_eval=unpack_nested(config, 'stochastic_eval'),
            )

        # If we want variance ablations, we instantiate all three policies
        if 'sh_smcts' in smc_cfg and unpack_nested(smc_cfg, 'sh_smcts')['variance_ablations']:
            # Instantiate a new planner, that is SMC baseline.
            self.testing_baseline_smc_policy = SMCPolicy(
                planner,
                discount=config['credit_assignment']['discount'],
                td_lambda=config['credit_assignment']['td_lambda'],
                option='dirac',
                value_mixing=unpack_nested(config, 'value_mixing'),
                stochastic_eval=unpack_nested(config, 'stochastic_eval'),
            )

            self.testing_smcts_policy = SMCPolicy(
                planner,
                discount=config['credit_assignment']['discount'],
                td_lambda=config['credit_assignment']['td_lambda'],
                option='dsmc',
                value_mixing=unpack_nested(config, 'value_mixing'),
                stochastic_eval=unpack_nested(config, 'stochastic_eval'),
            )

    def build(
            self,
            config: dict[str, Any],
            model: models.JointModel,
            env: jit_env.Environment
    ) -> Self:
        method = unpack_nested(config, "method")
        params = config['params']

        match method:
            case 'smc': self._setup_smc(params, model, env)
            case 'mctx' | 'mcts': self._setup_mctx(params, model, env)
            case 'ppo': self._setup_ppo(params, model, env)
            case _: raise ValueError(f'Unsupported agent method: {method}')

        if self.policy is None:
            raise RuntimeError(
                f"Policy creation failed for {method}: {config}"
            )

        return self


class LearnerBuilder:

    def __init__(self):
        self.learner: Learner | None = None

    def _setup_ppo(
            self,
            config: dict[str, Any],
            optimizer: optax.GradientTransformation,
            model: models.JointModel,
            dummy_ins
    ):
        self.learner = PPOLearner(
            optimizer, model, dummy_ins,
            hyperparameters=PPOLearnerHyperparameters(
                **unpack_nested(config, 'credit_assignment'),
                **unpack_nested(config, 'loss'),
                **unpack_nested(config, 'gradient_descent')
            )
        )

    def _setup_em(
            self,
            config: dict[str, Any],
            optimizer: optax.GradientTransformation,
            model: models.JointModel,
            dummy_ins
    ):
        self.learner = EMLearner(
            optimizer, model, dummy_ins,
            hyperparameters=EMLearnerHyperparameters(
                **unpack_nested(config, 'replay_buffer'),
                **unpack_nested(config, 'credit_assignment'),
                **unpack_nested(config, 'loss'),
                **unpack_nested(config, 'gradient_descent')
            )
        )

    def build(
            self,
            name: str,
            config: dict[str, Any],
            optimizer: optax.GradientTransformation,
            model: models.JointModel,
            dummy_ins
    ) -> Self:

        match name:
            case 'smc' | 'mctx' | 'mcts': self._setup_em(
                config, optimizer, model, dummy_ins
            )
            case 'ppo': self._setup_ppo(config, optimizer, model, dummy_ins)
            case _: raise ValueError(f'Unsupported learner for: {name}')

        if self.learner is None:
            raise RuntimeError(
                f"Learner creation failed for {name}: {config}"
            )

        return self
