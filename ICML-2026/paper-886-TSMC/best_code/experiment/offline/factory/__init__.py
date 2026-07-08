"""Factory module to parse a complex dict config into a runnable experiment

"""
from .problem import ProblemBuilder
from .agent import ModelBuilder, AgentBuilder, LearnerBuilder, make_optimizer
from .experiment import CompiledExperiment

from experiment.src.datagen import EnvironmentDataGenerator, SimpleEvaluator
from experiment.src.utils import OptionalWANDBLogger

from ..utils import unpack_nested


def from_config(config_dict: dict, service: None) -> CompiledExperiment:

    # Create the problem namespace
    task = ProblemBuilder().build(config_dict['environment'])

    # Create the agent
    model_builder = ModelBuilder().build(
        task.name, config_dict['model'],
        io_spec=(
            task.datagen_env.observation_spec(),
            task.datagen_env.action_spec()
        )
    )
    agent_builder = AgentBuilder().build(
        config_dict['policy'], model_builder.model, task.planner_env
    )

    # Create the learner (consumer)
    optimizer = make_optimizer(config_dict["optimizer"])

    dummy_obs = task.datagen_env.observation_spec().generate_value()
    dummy_action = task.datagen_env.action_spec().generate_value()

    learner = LearnerBuilder().build(
        name=unpack_nested(config_dict, 'policy', 'method'),
        config=config_dict['learner'],
        optimizer=optimizer,
        model=model_builder.model,
        dummy_ins=(dummy_obs, dummy_action)
    ).learner

    # Create the data-sources for training and evaluation (producers)
    generator = EnvironmentDataGenerator(
        task.datagen_env,
        agent_builder.policy,  # type: ignore
        batch_size=unpack_nested(
            config_dict, "data_generator", "batch_size"
        ),
        length=unpack_nested(
            config_dict, "data_generator", "length"
        ),
        as_pomdp=False
    )
    evaluator = SimpleEvaluator(
        task.format_eval_env(
            max_length=unpack_nested(config_dict, 'evaluation', 'timeout')
        ),
        agent_builder.policy,  # type: ignore
        testing_smcts_policy=agent_builder.testing_smcts_policy,  # type: ignore
        testing_smc_policy=agent_builder.testing_baseline_smc_policy,  # type: ignore
        batch_size=unpack_nested(config_dict, "evaluation", "batch_size"),
        as_pomdp=False,
        metric_fun=task.test_metric_fun,
        fixed_seed=unpack_nested(config_dict, 'evaluation', 'fixed_seed')
    )
    evaluator.training_datasize = generator.batch_size * generator.length

    # Unpack base experiment configuration
    seed = unpack_nested(config_dict, "rng", "seed")
    num_iter = unpack_nested(config_dict, "experiment", "max_iterations")
    eager_eval = unpack_nested(config_dict, "experiment", "eager_eval")
    p = unpack_nested(config_dict, "experiment", "eval_period")
    mp = unpack_nested(config_dict, "experiment", "snapshot_period")

    # Create the experiment to be executed (when calling `.run()`)
    experiment_fun = CompiledExperiment(
        seed=seed,
        datagen=generator,
        learner=learner,
        preprocessor=learner.preprocess_data,  # type: ignore
        logging_service=OptionalWANDBLogger(service),
        metric_fun=task.train_metric_fun,
        evaluator=evaluator,
        num_iterations=num_iter,
        eager_eval=eager_eval,
        evaluation_period=p,
        snapshot_period=mp
    )

    return experiment_fun
