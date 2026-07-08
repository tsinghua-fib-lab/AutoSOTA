"""Black-box optimizer training with an outer black-box optimizer."""

import json

import torch
from tqdm import tqdm

from meta_trainers.base_trainer import BaseTrainer
from optimizers import (
    BBORunner,
    CMAES,
    DE,
    GA,
    GradBasedLES,
    GradFreeLES,
    JADE,
    LDE,
    LGA,
    LSHADE,
    POM,
    PSO,
    SHADE,
)
from tasks.problem import MetaTask


OPTIMIZER_REGISTRY = {
    "POM": POM,
    "GradBasedLES": GradBasedLES,
    "GradFreeLES": GradFreeLES,
    "LGA": LGA,
    "LDE": LDE,
    "DE": DE,
    "JADE": JADE,
    "SHADE": SHADE,
    "LSHADE": LSHADE,
    "GA": GA,
    "PSO": PSO,
    "CMAES": CMAES,
}


def _build_optimizer(spec):
    name = spec["name"]
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")
    with open(spec["config"], "r", encoding="utf-8") as fp:
        config = json.load(fp)
    return OPTIMIZER_REGISTRY[name](config=config)


class BBOTrainer(BaseTrainer):
    """Train an inner optimizer by optimizing its parameters with an outer optimizer."""

    def __init__(self, config):
        super().__init__(config)

        self.meta_batchsize = config["meta_batchsize"]
        self.meta_popsize = config["meta_popsize"]
        self.meta_problemdim = config["meta_problemdim"]

        self.meta_opt = _build_optimizer(config["meta_opt"])
        self.outer_runner = BBORunner(algo=self.meta_opt)

        self.inner_opt = _build_optimizer(config["inner_opt"])
        self.inner_runner = BBORunner(algo=self.inner_opt)

        self._print_config()

    def _print_config(self):
        ext_configs = {
            "Outer optimizer name": self.meta_opt.name,
            "Outer population shape": {
                (self.batchsize, self.popsize, self.problemdim)
            },
            "Inner optimizer name": self.inner_opt.name,
        }
        super()._print_config(ext_configs=ext_configs)

    def train(self):
        epoch_bar = tqdm(range(int(self.n_epochs)), ncols=100)
        meta_task = MetaTask(
            repaire=True,
            dim=self.meta_problemdim,
            inner_algo=self.inner_runner,
            inner_generations=self.n_generations,
            inner_funcs=self.training_set,
            inner_popshape=(self.batchsize, self.popsize, self.problemdim),
        )
        param_pop = meta_task.init_population(
            popshape=(self.meta_batchsize, self.meta_popsize, self.meta_problemdim)
        )
        param_pop, _ = meta_task.calfitness(param_pop)
        start_fit = param_pop[..., 0].view(-1).min()
        best_fit = torch.inf

        for epoch in epoch_bar:
            out = self.outer_runner.step(param_pop, meta_task)
            param_pop = out[0] if isinstance(out, tuple) else out
            cur_fit = param_pop[..., 0].view(-1).min()
            improvement = cur_fit - start_fit
            start_fit = cur_fit

            self.logger.add_scalar("fitness/epoch", cur_fit, epoch)
            self.logger.add_scalar("improvement/epoch", improvement, epoch)

            if epoch > self.n_epochs // 2:
                ckpt = param_pop[..., 1:]
                if cur_fit < best_fit:
                    best_fit = cur_fit
                    torch.save(
                        ckpt,
                        f"{self.ckpt_saving_path}/{self.expname}_better_{epoch}.pth",
                    )
                elif epoch == self.n_epochs - 1:
                    torch.save(
                        ckpt,
                        f"{self.ckpt_saving_path}/{self.expname}_{epoch}.pth",
                    )
