"""
Training script for General EFC on arbitrary architectures.

Self-contained: includes its own CL trainer so that the general/ folder
does not depend on or modify any code in the main codebase.

Usage:
    python -m general.train --config general/configs/cifar10_resnet18.yaml
    python -m general.train --config general/configs/tinyimagenet_resnet18.yaml
"""

import argparse
import copy
import logging
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from general.network import GeneralEFCNetwork
from general.resnet import build_resnet18_blocks
from general.dataloader import EndToEndCLDataloader

logger = logging.getLogger(__name__)


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="General-architecture EFC training")

    # config file
    p.add_argument("--config", type=str, default=None)

    # architecture
    p.add_argument("--arch", type=str, default="resnet18")

    # training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--optimizer", type=str, default="Adam")
    p.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    p.add_argument("--loss_fn", type=str, default="ce")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # EFC hyper-parameters
    p.add_argument("--beta_efc", type=float, default=0.1)
    p.add_argument("--target_lr", type=float, default=0.02)
    p.add_argument("--dt_di", type=float, default=0.02)
    p.add_argument("--time_constant_ratio", type=float, default=0.2)
    p.add_argument("--k_p", type=float, default=2.0)
    p.add_argument("--tmax_di", type=int, default=300)
    p.add_argument("--eps", type=float, default=1e-4)

    # continual-learning setting
    p.add_argument("--setting", type=str, default="ClassILCIFAR10")
    p.add_argument("--dataset", type=str, default="CIFAR10")
    p.add_argument("--num_tasks", type=int, default=5)
    p.add_argument("--classes_per_task", type=int, default=2)

    # environment
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="./outputs/general")
    p.add_argument("--run_name", type=str, default="general_efc")
    p.add_argument("--save", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--peak", action="store_true")

    # first pass: load config file defaults
    initial, _ = p.parse_known_args()
    if initial.config:
        p.set_defaults(**OmegaConf.to_container(OmegaConf.load(initial.config)))

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"Ignoring unknown args: {unknown}")
    return args


# ======================================================================
# Model builder
# ======================================================================

def build_model(config):
    total_classes = config.num_tasks * config.classes_per_task
    small_input = config.dataset == "CIFAR10"
    blocks = build_resnet18_blocks(
        total_classes, in_channels=3, small_input=small_input
    )
    return GeneralEFCNetwork(blocks, config, name=f"GeneralEFC_{config.arch}")


# ======================================================================
# Continual-learning trainer
# ======================================================================

def _is_taskil(setting: str) -> bool:
    return "taskil" in setting.lower()


class GeneralCLTrainer:
    """
    Self-contained CL trainer for the general EFC network.

    Handles:
      - Least-square classifier initialisation per task
      - Per-task training with DI backward
      - Fisher computation between tasks
      - Task-IL / Class-IL evaluation
    """

    def __init__(self, model, tasks_dataloaders, config):
        self.model = model
        self.tasks = tasks_dataloaders
        self.config = config
        self.device = torch.device(config.device)

        self.model.to(self.device)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        os.makedirs(config.output_dir, exist_ok=True)
        self._setup_logging()

    # ---- logging ----

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    os.path.join(self.config.output_dir, "training.log"),
                    mode="w",
                ),
            ],
        )

    # ---- optimizer / scheduler ----

    def _reset_optimizer(self):
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.lr
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config.lr, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        if self.config.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        else:
            self.scheduler = None

    # ---- least-square classifier init ----

    def _least_square_init(self, dataloader, task_id):
        """
        Initialise the classifier head for a new task via closed-form
        least squares, using features from the penultimate block.
        """
        self.model.eval()
        cpt = self.config.classes_per_task
        start = task_id * cpt
        end = (task_id + 1) * cpt
        is_taskil = _is_taskil(self.config.setting)

        features_list, labels_list = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                # Forward through all blocks except classifier
                for block in self.model.blocks[:-1]:
                    x = block(x)
                features_list.append(x)
                labels_list.append(y.argmax(dim=1))

        features = torch.cat(features_list)
        labels = torch.cat(labels_list)
        N, d = features.shape

        features_ext = torch.cat(
            [features, torch.ones(N, 1, device=self.device)], dim=1
        )

        if is_taskil:
            # Task-IL: labels are already task-local (0 .. cpt-1)
            targets = torch.zeros(N, cpt, device=self.device)
            for i, lab in enumerate(labels):
                targets[i, lab] = 1.0
            Z, Y = features_ext, targets
        else:
            # Class-IL: labels are global; focus on current task's classes
            targets = torch.zeros(N, cpt, device=self.device)
            for i, lab in enumerate(labels):
                if start <= lab < end:
                    targets[i, lab - start] = 1.0
            mask = (labels >= start) & (labels < end)
            Z, Y = features_ext[mask], targets[mask]

        reg = 1e-4 * Z.shape[0] * torch.eye(d + 1, device=self.device)
        W_ls = torch.linalg.solve(Z.T @ Z + reg, Z.T @ Y)

        classifier = self.model.blocks[-1]  # nn.Linear
        with torch.no_grad():
            for c_idx in range(cpt):
                g = start + c_idx
                classifier.weight[g] = W_ls[:d, c_idx]
                classifier.bias[g] = W_ls[d, c_idx]

        logger.info(
            f"  LS init for task {task_id}: classes [{start}, {end})"
        )

    # ---- evaluation ----

    @torch.no_grad()
    def _evaluate(self, current_task_id):
        """Evaluate accuracy on all seen tasks."""
        self.model.eval()
        is_til = _is_taskil(self.config.setting)
        results = {}

        for t in range(current_task_id + 1):
            if is_til:
                self.model.task_id = t

            correct, total = 0, 0
            _, test_loader = self.tasks[t]
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                correct += (y_hat.argmax(1) == y.argmax(1)).sum().item()
                total += x.size(0)
            results[f"task_{t}"] = 100.0 * correct / total if total > 0 else 0.0

        results["average"] = (
            sum(results[f"task_{t}"] for t in range(current_task_id + 1))
            / (current_task_id + 1)
        )
        return results

    # ---- main training loop ----

    def train(self):
        wandb = None
        if self.config.use_wandb:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project="general-efc",
                name=self.config.run_name,
                config=OmegaConf.to_container(self.config),
            )

        global_step = 0

        for task_id, (train_loader, test_loader) in enumerate(self.tasks):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  Task {task_id + 1} / {len(self.tasks)}")
            logger.info(f"{'=' * 60}")

            self.model.task_id = task_id

            # LS init for tasks > 0
            if task_id > 0:
                self._least_square_init(train_loader, task_id)
                self.model._first_task = False

            # Fresh optimizer per task
            self._reset_optimizer()

            # Peak-model tracking (optional)
            best_acc = -1.0
            best_state = None

            for epoch in range(1, self.config.epochs + 1):
                # ---- train ----
                self.model.train()
                epoch_loss = 0.0

                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.model(x)
                    loss = self.model.calculate_loss(y_hat, y)

                    self.optimizer.zero_grad()
                    self.model.backward(y)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    global_step += 1

                epoch_loss /= len(train_loader)
                if self.scheduler:
                    self.scheduler.step()

                # ---- evaluate ----
                results = self._evaluate(task_id)

                logger.info(
                    f"  Epoch {epoch:>3d}/{self.config.epochs} | "
                    f"Loss {epoch_loss:.4f} | "
                    f"Avg Acc {results['average']:.2f}%"
                )

                if wandb is not None:
                    log_dict = {
                        "task": task_id,
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": epoch_loss,
                        "avg_accuracy": results["average"],
                    }
                    for t in range(task_id + 1):
                        log_dict[f"acc_task_{t}"] = results[f"task_{t}"]
                    wandb.log(log_dict, step=global_step)

                # Peak tracking
                if self.config.peak and task_id > 0:
                    if results["average"] > best_acc:
                        best_acc = results["average"]
                        best_state = copy.deepcopy(self.model.state_dict())

            # Restore peak model if applicable
            if self.config.peak and task_id > 0 and best_state is not None:
                self.model.load_state_dict(best_state)
                logger.info(f"  Restored peak model (acc={best_acc:.2f}%)")

            # ---- end of task: Fisher ----
            logger.info(f"  Computing Fisher for task {task_id} ...")
            self.model.complete_task(train_loader)

            final = self._evaluate(task_id)
            per_task = " | ".join(
                f"T{t}={final[f'task_{t}']:.1f}%"
                for t in range(task_id + 1)
            )
            logger.info(f"  End-of-task results: {per_task}")
            logger.info(f"  Average accuracy: {final['average']:.2f}%")

        if self.config.save:
            path = os.path.join(self.config.output_dir, "model_final.pt")
            torch.save(self.model.state_dict(), path)
            logger.info(f"  Saved model to {path}")

        if wandb is not None:
            wandb.finish()


# ======================================================================
# Entry point
# ======================================================================

def main():
    args = parse_args()
    config = OmegaConf.create(vars(args))

    print("=" * 60)
    print("  General EFC — Arbitrary-Architecture Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))

    # Build model
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.name}  ({n_params:,} parameters)")

    # Load data
    dl = EndToEndCLDataloader(config, config.dataset)
    tasks = dl.get_all_tasks_dataloaders()

    # Train
    trainer = GeneralCLTrainer(model, tasks, config)
    trainer.train()


if __name__ == "__main__":
    main()
