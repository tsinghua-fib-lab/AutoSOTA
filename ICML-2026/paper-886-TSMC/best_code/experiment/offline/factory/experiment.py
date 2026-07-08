from typing import Callable, Any

import jax

import flashbax as fbx

from jaxtyping import PRNGKeyArray

from experiment.src.types import Logger, MetricData, Learner
from experiment.src.datagen import EnvironmentDataGenerator
from experiment.src.experiment import run_experiment


class CompiledExperiment:

    def __init__(
            self,
            seed: int,
            datagen: EnvironmentDataGenerator,
            learner: Learner,
            preprocessor: Callable[..., dict[str, jax.Array]],
            metric_fun: Callable[[int, dict[str, Any]], MetricData],
            evaluator: Callable[[int, PRNGKeyArray, ...], MetricData],
            logging_service: Logger,
            num_iterations: int,
            eager_eval: bool,
            evaluation_period: int,
            snapshot_period: int
    ):
        self.seed = seed

        self.datagen = datagen
        self.learner = learner

        self.preprocessor = preprocessor
        self.metric_fun = metric_fun
        self.evaluator = evaluator
        self.logging_service = logging_service

        self.num_iterations = num_iterations
        self.eager_eval = eager_eval
        self.evaluation_period = evaluation_period
        self.snapshot_period = snapshot_period

    def run(self):
        if self.learner.param.prioritized:
            buffer = fbx.make_prioritised_flat_buffer(
                max_length=(
                        self.learner.param.max_age_buffer *
                        self.datagen.batch_size *
                        self.datagen.length
                ),
                min_length=max(
                    self.learner.param.min_length_buffer, self.datagen.length
                ),
                sample_batch_size=self.learner.param.batch_size,
                add_sequences=True,
                add_batch_size=self.datagen.batch_size,
                priority_exponent=self.learner.param.priority_exponent
            )
        else:
            buffer = fbx.make_flat_buffer(
                max_length=(
                    self.learner.param.max_age_buffer *
                    self.datagen.batch_size *
                    self.datagen.length
                ),
                min_length=max(
                    self.learner.param.min_length_buffer, self.datagen.length
                ),
                sample_batch_size=self.learner.param.batch_size,
                add_sequences=True,
                add_batch_size=self.datagen.batch_size
            )

        if hasattr(self.learner, 'set_buffer'):
            # Share reference to buffer logic (e.g., for prioritized sampling)
            self.learner.set_buffer(buffer)

        run_experiment(
            key=jax.random.key(self.seed),
            data_gen=self.datagen,
            buffer=buffer,
            learner=self.learner,
            preprocess_fun=jax.jit(self.preprocessor),
            evaluate_fun=self.evaluator,
            metric_fun=self.metric_fun,
            logger=self.logging_service,
            # Keyword only
            num_iterations=self.num_iterations,
            eval_period=self.evaluation_period,
            snapshot_period=self.snapshot_period,
            eager_eval=self.eager_eval,
            start=0,
            use_pbar=True,
        )
