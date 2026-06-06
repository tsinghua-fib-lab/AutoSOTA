import time
from typing import Any, Dict, Tuple, Union, Callable, Optional
import jax
from torch.utils.data import DataLoader
import jax.numpy as jnp
import optax
from flax import linen as nn
import wandb
from flax.training import train_state, checkpoints, common_utils
from jax.tree_util import tree_leaves
import numpy as np
import os

from src.utils.dp import theta_epsilon, clamp_matrix
from src.utils.utils import logger

from functools import partial

from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant


class Trainer:
    optimizer: optax.GradientTransformation
    lr_scheduler: optax.Schedule
    model: nn.Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader
    forget_dataloader: DataLoader
    retain_dataloader: DataLoader
    train_epochs: int
    unlearn_epochs: int
    num_classes: int
    train_weight_decay: np.float32
    unlearn_weight_decay: np.float32
    config: Dict[str, Any]
    ckpt_path: str
    use_pretrained: Union[str, bool]
    criterion: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    run_dir: str
    dp_sgd_train: bool = False

    unlearn_delta: np.float32
    unlearn_current_delta: np.float32
    unlearn_epsilon: np.float32
    unlearn_epsilon_renyi: np.float32

    unlearn_init_model_clip: np.float32
    unlearn_init_sigma: np.float32
    unlearn_grad_clip: Union[np.float32, bool]
    model_dimension: np.int32
    sigma: np.float32
    model_clip: np.float32
    noise_addition_after_projection: bool
    noise_addition_before_projection: bool
    stop_unlearning: bool = False
    pabi_steps: int
    post_unlearn_clip: np.float32

    pabi_intermediate_rho_t = jnp.array([], dtype=jnp.float32)
    pabi_intermediate_s_t = jnp.array([], dtype=jnp.float32)
    pabi_intermediate_sigma_t = jnp.array([], dtype=jnp.float32)
    pabi_intermediate_eps_renyi = []
    pabi_calc_intermediate_eps_renyi: bool = False

    logging_state_step_adjustment = 0

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        forget_dataloader: DataLoader,
        retain_dataloader: DataLoader,
        key: jax.Array,
        use_pretrained: Union[str, bool] = False,
        run_dir: str = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.forget_dataloader = forget_dataloader
        self.retain_dataloader = retain_dataloader
        self.train_epochs = config["training"]["epochs"]
        self.dp_sgd_train = config["training"].get("dp_sgd", False)
        self.unlearn_epochs = config["unlearning"]["epochs"]
        self.config = config
        self.ckpt_path = config["checkpoint_path"]
        self.use_pretrained = use_pretrained
        self.train_steps_per_epoch = len(train_dataloader)
        self.unlearn_steps_per_epoch = len(retain_dataloader)
        self.num_classes = config["model"]["n_classes"]
        self.train_weight_decay = config["training"]["weight_decay"]
        self.unlearn_weight_decay = config["unlearning"]["weight_decay"]
        self.key = key  # AK: what is key? # AJ: used for JAX PRNG
        self.criterion = optax.softmax_cross_entropy
        self.unlearn_init_model_clip = config["unlearning"]["init_model_clip"]
        self.unlearn_init_sigma = config["unlearning"]["init_sigma"]
        self.unlearn_delta = config["unlearning"]["delta"]
        self.run_dir = run_dir
        self.post_unlearn_clip = self.config["post_unlearning"]["post_unlearn_clip"]
        self.post_unlearn_weight_decay = self.config["post_unlearning"]["weight_decay"]
        self.fine_grained_validation = self.config["unlearning"].get(
            "fine_grained_validation", False
        )
        if self.dp_sgd_train:
            self.train_dataset_size = len(self.train_dataloader.dataset)
            self.batch_size = self.config["dataset"]["batch_size"]
            dp_cfg = self.config["training"]["dp_sgd"]
            self.l2_norm_clip = dp_cfg["l2_norm_clip"]
            target_epsilon = dp_cfg["target_epsilon"]
            target_delta = dp_cfg["target_delta"]
            self.train_epsilon = target_epsilon / len(self.forget_dataloader.dataset)
            self.train_delta = np.exp(
                np.log(target_delta)
                + ((self.train_epsilon) - 1)
                - ((self.train_epsilon * len(self.forget_dataloader.dataset)) - 1)
            )
            wandb.log(
                {
                    "train_epsilon": self.train_epsilon,
                    "train_delta": self.train_delta,
                }
            )
            print(
                {
                    "train_epsilon": self.train_epsilon,
                    "train_delta": self.train_delta,
                }
            )
            self.dp_sgd_seed = dp_cfg["seed"]
            self.noise_multiplier = self._compute_noise_multiplier()
            logger.info(
                f"{self.l2_norm_clip=} {target_epsilon=} {target_delta=} {self.train_epsilon=} {self.train_delta=} {self.noise_multiplier=}"
            )

        if self.config["unlearning"]["algorithm"] == "iteration":
            algo_config = config["unlearning"]["iteration"]
            self.unlearn_epsilon_renyi = algo_config["epsilon_renyi_target"]
            self.unlearn_grad_clip = algo_config["grad_clip"]
            self.pabi_calc_intermediate_eps_renyi = algo_config[
                "calc_intermediate_eps_renyi"
            ]
        elif self.config["unlearning"]["algorithm"] == "contractive_coefficients":
            algo_config = config["unlearning"]["contractive_coefficients"]
            self.unlearn_epsilon = algo_config["epsilon_target"]
            self.sigma = algo_config["sigma"]
            self.model_clip = algo_config["model_clip"]
            self.unlearn_grad_clip = algo_config["grad_clip"]
            self.noise_addition_after_projection = algo_config[
                "noise_addition_after_projection"
            ]
            self.noise_addition_before_projection = algo_config[
                "noise_addition_before_projection"
            ]
            assert (
                self.noise_addition_before_projection
                or self.noise_addition_after_projection
            )
            assert (
                self.unlearn_grad_clip and self.noise_addition_before_projection
            ) or (not self.unlearn_grad_clip and self.noise_addition_after_projection)
        elif self.config["unlearning"]["algorithm"] == "dp-baseline":
            algo_config = config["unlearning"]["dp-baseline"]
            self.unlearn_epsilon = algo_config["epsilon_target"]
        elif self.config["unlearning"]["algorithm"] == "retrain":
            self.unlearn_epsilon = 0
            self.start_from_all_zeros = self.config["unlearning"]["retrain"].get(
                "start_from_all_zeros", False
            )
            self.clip_and_noise = self.config["unlearning"]["retrain"].get(
                "clip_and_noise", False
            )
        elif self.config["unlearning"]["algorithm"] == "dp-sgd":
            # Group Privacy
            self.unlearn_epsilon = (
                len(self.forget_dataloader.dataset) * self.train_epsilon
            )
            self.unlearn_delta = np.exp(
                np.log(self.train_delta)
                + ((self.train_epsilon * len(self.forget_dataloader.dataset)) - 1)
                - ((self.train_epsilon) - 1)
            )
        else:
            raise ValueError("Invalid unlearning algorithm")

    def _compute_noise_multiplier(self) -> float:
        sampling_rate = self.batch_size / self.train_dataset_size
        num_steps = int(self.train_epochs * self.train_steps_per_epoch)
        orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 10240, 50))
        low, high, tolerance = 0.01, 1e5, 0.0001
        sigma = None
        epsilon = np.inf
        while abs(epsilon - self.train_epsilon) > tolerance:
            mid = (low + high) / 2
            
            accountant = rdp_privacy_accountant.RdpAccountant(orders)
            accountant.compose(
                dp_event.PoissonSampledDpEvent(
                    sampling_rate, dp_event.GaussianDpEvent(mid)
                ),
                num_steps,
            )
            epsilon = accountant.get_epsilon(self.train_delta)
            logger.info(f"checking {mid=} {epsilon=}")
            if epsilon <= self.train_epsilon:
                high = mid
                sigma = mid
            else:
                low = mid
            new_mid = (low + high) / 2
            if abs(new_mid - mid) < 1e-5:
                break

        if sigma is None:
            raise ValueError("No valid noise multiplier found")

        accountant = rdp_privacy_accountant.RdpAccountant(orders)
        accountant.compose(
            dp_event.PoissonSampledDpEvent(
                sampling_rate, dp_event.GaussianDpEvent(sigma)
            ),
            num_steps,
        )
        final_epsilon = accountant.get_epsilon(self.train_delta)
        logger.info(
            f"DP-SGD: σ={sigma}, Achieved ε={final_epsilon} (δ={self.train_delta})"
        )
        wandb.log({"dp_sigma": sigma})
        return sigma

    def train_setup(self) -> train_state.TrainState:
        self.set_lr_scheduler()
        self.set_optimizer()
        self.key, subkey = jax.random.split(self.key)

        variables = self.model.init(
            {"params": subkey}, jnp.ones((1, *self.config["dataset"]["input_shape"]))
        )
        params = variables["params"]
        self.model_dimension = sum(x.size for x in tree_leaves(params))
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self.optimizer
        )

    def set_optimizer(self, phase="training"):
        base_optim_name = self.config[phase]["optim"]
        if base_optim_name == "SGD":
            base_optimizer = optax.sgd(
                learning_rate=self.lr_schedule,
                momentum=self.config[phase].get("momentum", 0),
                nesterov=self.config[phase].get("nesterov", False),
            )
            optimizer_chain = [base_optimizer]
            if self.train_weight_decay > 0 and phase == "training":
                optimizer_chain.insert(
                    0, optax.add_decayed_weights(self.train_weight_decay)
                )

        elif base_optim_name == "RMSprop":
            base_optimizer = optax.rmsprop(learning_rate=self.lr_schedule)
            optimizer_chain = [base_optimizer]
            if self.train_weight_decay > 0 and phase == "training":
                optimizer_chain.insert(
                    0, optax.add_decayed_weights(self.train_weight_decay)
                )

        elif base_optim_name == "Adam":
            base_optimizer = optax.adamw(
                learning_rate=self.lr_schedule,
                weight_decay=self.train_weight_decay if phase == "training" else 0,
            )
            optimizer_chain = [base_optimizer]

        else:
            raise ValueError(f"Invalid optimizer: {base_optim_name}")

        if self.dp_sgd_train and phase == "training":
            logger.info("Setting up DP-SGD optimizer...")
            dp_aggregate = optax.contrib.differentially_private_aggregate(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                seed=self.dp_sgd_seed,
            )
            optimizer_chain.insert(0, dp_aggregate)

        self.optimizer = optax.chain(*optimizer_chain)

    def set_lr_scheduler(self, phase="training"):
        if self.config[phase]["lr_schedule"] == "constant":
            self.lr_schedule = optax.constant_schedule(self.config[phase]["max_lr"])
        elif self.config[phase]["lr_schedule"] == "cos":
            self.lr_schedule = optax.cosine_decay_schedule(
                init_value=self.config[phase]["max_lr"],
                decay_steps=max(self.train_epochs, 1)
                if phase == "training"
                else self.unlearn_epochs,
            )
        elif self.config[phase]["lr_schedule"] == "onecycle":
            if phase == "training":
                steps = max(self.train_epochs, 1) * self.train_steps_per_epoch
            else:
                steps = self.unlearn_epochs * self.unlearn_steps_per_epoch
            self.lr_schedule = optax.linear_onecycle_schedule(
                steps,
                self.config[phase]["max_lr"],
            )
        else:
            raise ValueError("Invalid learning rate schedule")

    @partial(jax.jit, static_argnums=(0,))
    def compute_loss(self, logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        one_hot_labels = common_utils.onehot(labels, num_classes=self.num_classes)
        xentropy = self.criterion(logits, one_hot_labels)
        return jnp.mean(xentropy)

    @partial(jax.jit, static_argnums=(0,))
    def dp_train_step(
        self, state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
        images, labels = batch

        def single_example_loss_fn(params, image, label):
            image_batch = jnp.expand_dims(image, axis=0)
            label_batch = jnp.expand_dims(label, axis=0)

            logits = state.apply_fn(
                {"params": params},
                image_batch,
                train=True,
            )
            loss = self.compute_loss(logits, label_batch)
            return loss, logits

        grad_fn_vmap = jax.vmap(
            jax.value_and_grad(single_example_loss_fn, has_aux=True),
            in_axes=(None, 0, 0),
            out_axes=0,
        )
        (losses, logits_batch), grads = grad_fn_vmap(state.params, images, labels)

        loss = jnp.mean(losses)
        accuracy = jnp.mean(jnp.argmax(jnp.squeeze(logits_batch), -1) == labels)
        new_state = state.apply_gradients(grads=grads)
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return new_state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self, state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
        images, labels = batch

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                images,
                train=True,
            )
            loss = self.compute_loss(logits, labels)
            # weight_penalty_params = jax.tree_util.tree_leaves(params)
            # # l_inf = jnp.max(jnp.array([jnp.max(jnp.abs(x)) for x in weight_penalty_params]))
            # l_inf = 0
            # weight_penalty_params = jax.tree_util.tree_leaves(params)
            # weight_l2 = jnp.sum(
            #     jnp.array([jnp.sum(x**2) for x in weight_penalty_params])
            # )
            # weight_penalty = self.train_weight_decay * 0.5 * weight_l2 + l_inf
            weight_penalty = 0  # handled by optax.add_decayed_weights
            total_loss = loss + weight_penalty
            return total_loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        # clipped_grads = self.clip_params(grads, 1)
        clipped_grads = grads
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        new_state = state.apply_gradients(
            grads=clipped_grads,
        )
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return new_state, metrics

    def train_one_epoch(self, state, epoch):
        train_metrics = []
        train_func = self.dp_train_step if self.dp_sgd_train else self.train_step

        for batch in self.train_dataloader:
            state, metrics = train_func(state, batch)
            train_metrics.append(metrics)

        train_metrics = common_utils.stack_forest(train_metrics)

        summary = {
            f"train_{k}": v
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), train_metrics
            ).items()
        }
        summary["epoch"] = epoch

        wandb.log(summary)
        logger.info(summary)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def eval_step(
        self, state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        images, labels = batch
        logits = state.apply_fn(
            {"params": state.params},
            images,
            train=False,
        )
        loss = self.compute_loss(logits, labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return {"loss": loss, "accuracy": accuracy}

    def evaluate(self, state, dataloader, prefix):
        eval_metrics = []
        for batch in dataloader:
            metrics = self.eval_step(state, batch)
            eval_metrics.append(metrics)

        eval_metrics = common_utils.stack_forest(eval_metrics)
        return {
            f"{prefix}_{k}": v
            for k, v in jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics).items()
        }

    def validate(self, state, data_loader, epoch, prefix):
        val_metrics = self.evaluate(state, data_loader, prefix)
        epoch_prefix = "epoch" if "train" in prefix.lower() else "unlearn_epoch"
        val_metrics[epoch_prefix] = epoch
        wandb.log(val_metrics)
        logger.info(val_metrics)

    def save_checkpoint(self, state: train_state.TrainState, epoch: int):
        checkpoints.save_checkpoint(
            ckpt_dir=os.path.abspath(self.ckpt_path),
            target=state,
            step=epoch,
            keep=self.train_epochs + 1,
        )

    @staticmethod
    def load_checkpoint(
        ckpt_dir: str, state: train_state.TrainState, epoch: int
    ) -> train_state.TrainState:
        return checkpoints.restore_checkpoint(os.path.abspath(ckpt_dir), state, epoch)

    def fit(self) -> train_state.TrainState:
        if self.use_pretrained:
            state = self.load_checkpoint(
                self.use_pretrained, self.train_setup(), self.train_epochs
            )
            metrics = self.evaluate(state, self.val_dataloader, "Val (Train)")
            logger.info("Skipping training, load from checkpoint")
            logger.info(metrics)
            wandb.log(metrics)
            return state

        state = self.train_setup()
        start_time = time.time()
        self.save_checkpoint(state, 0)  # random model
        for epoch in range(1, self.train_epochs + 1):
            state = self.train_one_epoch(state, epoch)
            self.validate(state, self.val_dataloader, epoch, "Val (Train)")
            if (epoch + 1) % self.config["save_every"] == 0:
                self.save_checkpoint(state, epoch)
        total_time = time.time() - start_time
        logger.info(f"Training took {total_time} seconds")
        l2_norm_model = sum(
            jnp.sum(x**2) for x in jax.tree_util.tree_leaves(state.params) if x.ndim > 1
        )
        logger.info(f"L2 norm of model: {jnp.sqrt(l2_norm_model)}")
        return state

    def test(self, state):
        test_metrics = self.evaluate(state, self.test_dataloader, "Test")
        wandb.log(test_metrics)
        logger.info(test_metrics)

    @partial(jax.jit, static_argnums=(0,))
    def clip_params(self, params, max_norm):
        params_leaves = jax.tree_util.tree_leaves(params)
        l2_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in params_leaves))
        factor = jnp.minimum(1.0, max_norm / l2_norm)
        return jax.tree_util.tree_map(lambda x: x * factor, params)

    @partial(jax.jit, static_argnums=(0,))
    def clamp_params(self, params, max_norm):
        return jax.tree_util.tree_map(
            lambda x: clamp_matrix(x, -max_norm, max_norm), params
        )

    def add_noise_to_params(self, params, sigma, key) -> Tuple[jnp.ndarray, jax.Array]:
        # inspired from optax.dpsgd implementation
        params_flat, params_treedef = jax.tree_util.tree_flatten(params)

        new_key, *rngs = jax.random.split(key, len(params_flat) + 1)
        noised = [
            (p + sigma * jax.random.normal(r, p.shape, p.dtype))
            for p, r in zip(params_flat, rngs)
        ]
        return jax.tree_util.tree_unflatten(params_treedef, noised), new_key

    def unlearn_setup(self, state: train_state.TrainState) -> train_state.TrainState:
        self.set_lr_scheduler(phase="unlearning")
        self.set_optimizer(phase="unlearning")
        # print number of params in state
        # param_shapes = jax.tree_util.tree_map(lambda x: x.shape, state.params)
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        # for shape in jax.tree_util.tree_flatten(param_shapes)[0]:
        #     logger.info(f"{shape=}")
        logger.info(f"Number of parameters in model: {num_params}")
        if self.config["unlearning"]["algorithm"] == "dp-sgd":
            self.stop_unlearning = True
            return state

        if self.config["unlearning"]["algorithm"] == "retrain":
            self.stop_unlearning = True
            self.key, subkey = jax.random.split(self.key)
            variables = self.model.init(
                {"params": subkey},
                jnp.ones((1, *self.config["dataset"]["input_shape"])),
            )
            params = variables["params"]
            self.model_dimension = sum(x.size for x in tree_leaves(params))
            state = train_state.TrainState.create(
                apply_fn=self.model.apply, params=params, tx=self.optimizer
            )
            if self.start_from_all_zeros:
                state = state.replace(
                    params=jax.tree_util.tree_map(jnp.zeros_like, state.params)
                )
                logger.info(
                    f"Starting from all zeros norm = {optax.global_norm(state.params)}"
                )
                noisy_params, self.key = self.add_noise_to_params(
                    state.params, self.unlearn_init_sigma, self.key
                )
                clipped_params = self.clip_params(
                    noisy_params, self.unlearn_init_model_clip
                )
                state = state.replace(params=clipped_params)

            if self.clip_and_noise:
                clipped_params = self.clip_params(
                    state.params, self.unlearn_init_model_clip
                )
                final_init_params, self.key = self.add_noise_to_params(
                    clipped_params, self.unlearn_init_sigma, self.key
                )
                state = state.replace(params=final_init_params)
            return state

        if self.config["unlearning"]["init_model_clip_type"] == "clamp":
            clipped_params = self.clamp_params(
                state.params, self.unlearn_init_model_clip
            )
            self.unlearn_init_model_clip = jnp.sqrt(
                sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(clipped_params))
            )
            logger.info(
                f"Clamped, l2 norm of resulting clip {self.unlearn_init_model_clip}"
            )
        else:
            clipped_params = self.clip_params(
                state.params, self.unlearn_init_model_clip
            )

        if self.config["unlearning"]["algorithm"] == "dp-baseline":
            self.stop_unlearning = True
            self.unlearn_init_sigma = jnp.sqrt(
                2
                * np.log(1.25 / self.unlearn_delta)
                * jnp.power(2 * self.unlearn_init_model_clip, 2)
                / jnp.power(self.unlearn_epsilon, 2)
            )

        if self.unlearn_init_sigma:
            final_init_params, self.key = self.add_noise_to_params(
                clipped_params, self.unlearn_init_sigma, self.key
            )
            wandb.log({"init_sigma": self.unlearn_init_sigma})
        else:
            final_init_params = clipped_params

        if self.config["unlearning"]["algorithm"] == "contractive_coefficients":
            if self.unlearn_init_sigma:
                self.unlearn_current_delta = theta_epsilon(
                    epsilon=self.unlearn_epsilon,
                    r=2 * self.unlearn_init_model_clip / self.unlearn_init_sigma,
                )
            else:
                self.unlearn_current_delta = 1
            wandb.log({"delta": self.unlearn_current_delta, "state_step": 0})
        elif self.config["unlearning"]["algorithm"] == "iteration":
            self.pabi_steps = np.ceil(
                self.unlearn_init_model_clip
                / (self.unlearn_grad_clip * self.config["unlearning"]["max_lr"])
            )
            if self.unlearn_weight_decay:
                smaller_version = np.ceil(
                    np.log(
                        self.unlearn_weight_decay
                        * 2
                        * self.unlearn_init_model_clip
                        / self.unlearn_grad_clip
                    )
                    / (self.config["unlearning"]["max_lr"] * self.unlearn_weight_decay)
                )
                if smaller_version > 0 and smaller_version < self.pabi_steps:
                    wandb.log({"reg_saved_pabi_steps": True})
                    self.pabi_steps = smaller_version

            wandb.log({"pabi_steps": self.pabi_steps})

        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=final_init_params, tx=self.optimizer
        )

    def post_unlearn_setup(
        self, state: train_state.TrainState
    ) -> train_state.TrainState:
        self.set_lr_scheduler(phase="post_unlearning")
        self.set_optimizer(phase="post_unlearning")
        # reset opt cuz we dont want noisy momentum history
        self.logging_state_step_adjustment = state.step
        l2_norm_params = jnp.sqrt(
            sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(state.params))
        )
        logger.info(f"l2_norm after noise addition {l2_norm_params}")

        if self.post_unlearn_clip:
            logger.info("Clipping after unlearning")
            clipped_params = self.clip_params(state.params, self.post_unlearn_clip)
            l2_norm_clipped_params = jnp.sqrt(
                sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(clipped_params))
            )
            logger.info(f"l2 norm after clipping {l2_norm_clipped_params}")
            return train_state.TrainState.create(
                apply_fn=self.model.apply,
                params=clipped_params,
                tx=self.optimizer,
            )
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=state.params,
            tx=self.optimizer,
        )

    def unlearn_sigma_t(self, step):
        if self.config["unlearning"]["algorithm"] == "contractive_coefficients":
            return self.sigma
        if self.config["unlearning"]["algorithm"] != "iteration":
            raise ValueError("algorithm not supported")

        # calculating sigma for privacy amplification by iteration approach
        lambda_t = self.unlearn_weight_decay
        diam_init_model_clip = 2 * self.unlearn_init_model_clip
        T = self.pabi_steps
        if self.config["unlearning"]["init_model_clip_type"] == "clamp":
            diam_init_model_clip *= jnp.sqrt(self.model_dimension)

        if lambda_t:
            if self.config["unlearning"]["noise_schedule"] == "constant":
                if self.config["unlearning"]["lr_schedule"] == "constant":
                    eta_t = self.lr_schedule(step)
                else:
                    ValueError(
                        "Only constant learning rate is supported for constant noise addition"
                    )
                decay_factor = 1 - eta_t * lambda_t
                common_factor = jnp.power(decay_factor, T)
                grad_clip_term = (2 * self.unlearn_grad_clip / lambda_t) * (
                    1 - common_factor
                )
                var_t = jnp.power(
                    (diam_init_model_clip * common_factor + grad_clip_term), 2
                )
                numerator = eta_t * lambda_t * (2 - eta_t * lambda_t)
                denom = (
                    2 * self.unlearn_epsilon_renyi * (1 - jnp.power(common_factor, 2))
                )
                scale_factor = numerator / denom
                var_t *= scale_factor
                sigma_t = jnp.sqrt(var_t)
            elif self.config["unlearning"]["noise_schedule"] == "decreasing":
                if self.config["unlearning"]["lr_schedule"] == "constant":
                    eta_t = self.lr_schedule(step)
                else:
                    raise ValueError(
                        "Only constant learning rate is supported for constant noise addition"
                    )
                decay_factor = 1 - eta_t * lambda_t
                common_factor = jnp.power(decay_factor, T)
                grad_clip_term = (2 * self.unlearn_grad_clip / lambda_t) * (
                    1 - common_factor
                )
                var_t = jnp.power(
                    (diam_init_model_clip * common_factor + grad_clip_term), 2
                )
                scale = (
                    2
                    * self.unlearn_epsilon_renyi
                    * T
                    * jnp.power(decay_factor, 2 * (T - step - 1))
                )
                sigma_t = jnp.sqrt(var_t / scale)
        else:  
            if self.config["unlearning"]["lr_schedule"] == "constant":
                eta_t = self.lr_schedule(step)
            else:
                raise ValueError(
                    "Only constant learning rate is supported for constant noise addition"
                )
            var_t = jnp.power(
                diam_init_model_clip + 2 * self.unlearn_grad_clip * eta_t * T, 2
            )
            sigma_t = jnp.sqrt(var_t / (4 * self.unlearn_epsilon_renyi * T))
        return sigma_t

    @partial(jax.jit, static_argnums=(0,))
    def unlearn_step(
        self, state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[
        train_state.TrainState, Dict[str, jnp.ndarray], Dict[str, jnp.number], jax.Array
    ]:
        images, labels = batch

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                images,
                train=True,
            )
            loss = self.compute_loss(logits, labels)
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        eta_t = self.lr_schedule(state.step)
        sigma_t = self.unlearn_sigma_t(state.step)
        if self.config["unlearning"]["algorithm"] == "iteration":
            clipped_grads = self.clip_params(grads, self.unlearn_grad_clip)
            total_update = jax.tree_util.tree_map(
                lambda g, p: g + self.unlearn_weight_decay * p,
                clipped_grads,
                state.params,
            )
            new_state = state.apply_gradients(
                grads=total_update,
            )
            params_with_noise, new_key = self.add_noise_to_params(
                new_state.params, sigma_t, self.key
            )
            new_state = new_state.replace(params=params_with_noise)
        elif self.config["unlearning"]["algorithm"] == "contractive_coefficients":
            total_update = grads
            if self.unlearn_weight_decay:
                total_update = jax.tree_util.tree_map(
                    lambda g, p: g + self.unlearn_weight_decay * p,
                    total_update,
                    state.params,
                )

            if self.unlearn_grad_clip:
                total_update = self.clip_params(total_update, self.unlearn_grad_clip)
            # jax.debug.print("515: state norm {}", optax.global_norm(state.params))
            new_state = state.apply_gradients(
                grads=total_update,
            )
            # jax.debug.print("518: state norm {}", optax.global_norm(new_state.params))
            if self.noise_addition_before_projection:
                params_with_noise, new_key = self.add_noise_to_params(
                    new_state.params, sigma_t, self.key
                )
                new_state = new_state.replace(params=params_with_noise)
            clipped_model = self.clip_params(new_state.params, self.model_clip)
            if self.noise_addition_after_projection:
                key = new_key if self.noise_addition_before_projection else self.key
                params_with_noise, new_key = self.add_noise_to_params(
                    clipped_model, sigma_t, key
                )
                new_state = new_state.replace(params=params_with_noise)
            else:
                new_state = new_state.replace(params=clipped_model)

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return new_state, metrics, sigma_t, eta_t, new_key

    def dp_accountant(self, step, sigma_t, eta_t) -> Optional[bool]:
        if self.config["unlearning"]["algorithm"] == "contractive_coefficients":
            if self.stop_unlearning:
                gamma = 1
            elif self.unlearn_grad_clip and self.noise_addition_before_projection:
                epsilon_tilde = np.log(
                    1 + (np.exp(self.unlearn_epsilon) - 1) / self.unlearn_current_delta
                )
                gamma = theta_epsilon(
                    epsilon_tilde,
                    2 * (self.model_clip + eta_t * self.unlearn_grad_clip) / sigma_t,
                )
            elif (not self.unlearn_grad_clip) and self.noise_addition_after_projection:
                epsilon_tilde = np.log(
                    1 + (np.exp(self.unlearn_epsilon) - 1) / self.unlearn_current_delta
                )
                gamma = theta_epsilon(epsilon_tilde, 2 * self.model_clip / sigma_t)
            else:
                raise ValueError("Dont have theorem for this combination")
            self.unlearn_current_delta *= gamma
            wandb.log(
                {
                    "gamma": gamma,
                    "state_step": self.logging_state_step_adjustment + step,
                    "delta": self.unlearn_current_delta,
                    "epsilon": self.unlearn_epsilon,
                }
            )
            logger.debug(
                {
                    "gamma": gamma,
                    "state_step": self.logging_state_step_adjustment + step,
                    "delta": self.unlearn_current_delta,
                    "epsilon": self.unlearn_epsilon,
                }
            )
            if self.unlearn_current_delta <= self.unlearn_delta:
                if not self.stop_unlearning:
                    self.stop_unlearning = True
                    wandb.log({"stop_unlearn": step})
                    logger.info(f"Stopped unlearning, reached delta, {step=}")
                    return True

        elif self.config["unlearning"]["algorithm"] == "iteration":
            if self.stop_unlearning:
                return None
            elif step == self.pabi_steps:
                eps_renyi = self.unlearn_epsilon_renyi
                alpha = np.linspace(
                    1.001, 1000, 10000
                )
                eps_array = (
                    eps_renyi
                    - (np.log(self.unlearn_delta) + np.log(alpha)) / (alpha - 1)
                    + np.log((alpha - 1) / alpha)
                )
                best_alpha = alpha[np.argmin(eps_array)]
                eps = np.min(eps_array)
                wandb.log(
                    {"stop_unlearn": step, "epsilon": eps, "delta": self.unlearn_delta}
                )
                logger.info(f"Stopped unlearning, Best alpha: {best_alpha}")
                logger.info(f"(eps, delta): ({eps}, {self.unlearn_delta})")
                self.stop_unlearning = True
                return True

            elif not self.stop_unlearning and self.pabi_calc_intermediate_eps_renyi:
                diam_kappa = 2 * self.unlearn_init_model_clip
                rho_t = 1 - eta_t * self.unlearn_weight_decay
                s_t = 2 * eta_t * self.unlearn_grad_clip
                if len(self.pabi_intermediate_rho_t) > 1:
                    """
                    numerator = 0.5*{\left[ \left(\prod_{t=0}^{T-1} \rho_t \right) \diam(\cK) + \sum_{t=0}^{T-1} \left(\prod_{k=1}^{T-1-t} \rho_k \right) s_t \right]^2}
                    denominator = {\sum_{t=0}^{T-1} \left(\prod_{k=1}^{T-1-t} \rho_k^2 \right) \sigma_t^2}
                    """

                    rho_t_prod = jnp.cumprod(self.pabi_intermediate_rho_t)
                    rho_t_sq_prod = jnp.cumprod(self.pabi_intermediate_rho_t**2)

                    sum_rho_ts_s_t = jnp.sum(
                        self.pabi_intermediate_s_t
                        * jnp.flip(rho_t_prod / rho_t_prod[0])
                    )
                    numerator = (
                        0.5 * (rho_t_prod[-1] * diam_kappa + sum_rho_ts_s_t) ** 2
                    )
                    denominator = jnp.sum(
                        self.pabi_intermediate_sigma_t**2
                        * jnp.flip(rho_t_sq_prod / rho_t_sq_prod[0])
                    )

                    epsilon_t = numerator / denominator
                    self.pabi_intermediate_eps_renyi.append(epsilon_t)
                    wandb.log(
                        {
                            "epsilon_renyi": epsilon_t,
                            "state_step": self.logging_state_step_adjustment + step,
                        }
                    )
                self.pabi_intermediate_rho_t = jnp.append(
                    self.pabi_intermediate_rho_t, rho_t
                )
                self.pabi_intermediate_s_t = jnp.append(self.pabi_intermediate_s_t, s_t)
                self.pabi_intermediate_sigma_t = jnp.append(
                    self.pabi_intermediate_sigma_t, sigma_t
                )
        return None

    def unlearn_one_epoch(
        self, state: train_state.TrainState, epoch
    ) -> train_state.TrainState:
        unlearn_metrics = []

        for i, batch in enumerate(self.retain_dataloader):
            # jax.debug.pri)nt("643: state norm {}", optax.global_norm(state.params))
            if self.stop_unlearning:
                state, metrics = self.post_unlearn_step(state, batch)
                sigma_t = 0
                eta_t = self.lr_schedule(state.step)
            else:
                state, metrics, sigma_t, eta_t, self.key = self.unlearn_step(
                    state, batch
                )
            # jax.debug.print("645: state norm {}", optax.global_norm(state.params))

            unlearn_stop = self.dp_accountant(state.step, sigma_t, eta_t)

            wandb.log(
                {
                    **metrics,
                    "sigma_t": sigma_t,
                    "eta_t": eta_t,
                    "state_step": self.logging_state_step_adjustment + state.step,
                }
            )
            logger.debug(
                {
                    **metrics,
                    "sigma_t": sigma_t,
                    "eta_t": eta_t,
                    "state_step": self.logging_state_step_adjustment + state.step,
                }
            )

            if unlearn_stop:
                logger.info("Validating after all noise addition")
                self.validate(
                    state, self.val_dataloader, epoch, "Val (Post Noise Addition)"
                )
                self.validate(
                    state, self.forget_dataloader, epoch, "Forget (Post Noise Addition)"
                )
                self.validate(
                    state, self.retain_dataloader, epoch, "Retain (Post Noise Addition)"
                )
                state = self.post_unlearn_setup(state=state)
            if self.fine_grained_validation:
                if not self.stop_unlearning:
                    self.validate(
                        state,
                        self.val_dataloader,
                        epoch - 1 + i / self.unlearn_steps_per_epoch,
                        "Val (Unlearn)",
                    )
                    self.validate(
                        state,
                        self.retain_dataloader,
                        epoch - 1 + i / self.unlearn_steps_per_epoch,
                        "Retain (Unlearn)",
                    )
                elif i == self.unlearn_steps_per_epoch // 2:
                    self.validate(
                        state, self.val_dataloader, epoch - 0.5, "Val (Unlearn)"
                    )
                    self.validate(
                        state, self.retain_dataloader, epoch - 0.5, "Retain (Unlearn)"
                    )
            unlearn_metrics.append(metrics)

        unlearn_metrics = common_utils.stack_forest(unlearn_metrics)

        summary = {
            f"retain_{k}": v
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), unlearn_metrics
            ).items()
        }
        summary["unlearn_epoch"] = epoch

        wandb.log(summary)
        logger.info(summary)
        return state

    def unlearn(self, state: train_state.TrainState):
        start_time = time.time()
        l2_norm_params = jnp.sqrt(
            sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(state.params))
        )
        l_inf_norm_params = max(
            jnp.max(jnp.abs(x)) for x in jax.tree_util.tree_leaves(state.params)
        )
        logger.info(f"after pruning {l2_norm_params=} {l_inf_norm_params=}")
        state = self.unlearn_setup(state)
        post_setup_metrics = self.evaluate(
            state, self.val_dataloader, "Val (Post Unlearning Setup)"
        )
        l2_norm_params = jnp.sqrt(
            sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(state.params))
        )
        l_inf_norm_params = max(
            jnp.max(jnp.abs(x)) for x in jax.tree_util.tree_leaves(state.params)
        )
        logger.info(f"after unlearn setup {l2_norm_params=} {l_inf_norm_params=}")
        wandb.log(post_setup_metrics)
        logger.info(post_setup_metrics)
        if self.stop_unlearning:
            wandb.log(
                {
                    "epsilon": self.unlearn_epsilon,
                    "delta": self.unlearn_delta,
                    "state_step": 0,
                }
            )
            logger.info("Validating after all noise addition")
            self.validate(state, self.val_dataloader, 0, "Val (Post Noise Addition)")
            self.validate(
                state, self.forget_dataloader, 0, "Forget (Post Noise Addition)"
            )
            self.validate(
                state, self.retain_dataloader, 0, "Retain (Post Noise Addition)"
            )
            state = self.post_unlearn_setup(state=state)

        for epoch in range(1, self.unlearn_epochs + 1):
            state = self.unlearn_one_epoch(state, epoch)
            self.validate(state, self.val_dataloader, epoch, "Val (Unlearn)")
            self.validate(state, self.forget_dataloader, epoch, "Forget (Unlearn)")
        total_time = time.time() - start_time
        logger.info(f"Unlearning took {total_time} seconds")
        if (
            self.config["unlearning"]["algorithm"] == "iteration"
            and self.pabi_calc_intermediate_eps_renyi
        ):
            import matplotlib.pyplot as plt

            plt.plot(self.pabi_intermediate_eps_renyi)
            plt.xlabel("Step")
            plt.ylabel("Epsilon Renyi")
            plt.axhline(
                y=self.unlearn_epsilon_renyi,
                color="r",
                linestyle="--",
                label=f"Target Epsilon Renyi {self.unlearn_epsilon_renyi}",
            )
            plt.axhline(
                y=self.pabi_intermediate_eps_renyi[-1],
                color="g",
                linestyle="--",
                label=f"Final Epsilon Renyi {self.pabi_intermediate_eps_renyi[-1]}",
            )
            plt.legend()
            plt.savefig(f"{self.run_dir}/eps_renyi_vs_step.png")

        return state

    @partial(jax.jit, static_argnums=(0,))
    def post_unlearn_step(
        self, state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
        images, labels = batch

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                images,
                train=True,
            )
            loss = self.compute_loss(logits, labels)
            weight_penalty_params = jax.tree_util.tree_leaves(params)
            weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params)
            weight_penalty = self.post_unlearn_weight_decay * 0.5 * weight_l2
            total_loss = loss + weight_penalty
            return total_loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        new_state = state.apply_gradients(
            grads=grads,
        )
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return new_state, metrics
