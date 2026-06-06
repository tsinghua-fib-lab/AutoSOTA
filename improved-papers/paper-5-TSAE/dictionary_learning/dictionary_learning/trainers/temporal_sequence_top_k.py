import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from collections import namedtuple
from typing import Optional
from math import isclose

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


def apply_temperature(probabilities: list[float], temperature: float) -> list[float]:
    """
    Apply temperature scaling to a list of probabilities using PyTorch.

    Args:
        probabilities (list[float]): Initial probability distribution
        temperature (float): Temperature parameter (> 0)

    Returns:
        list[float]: Scaled and normalized probabilities
    """
    probs_tensor = t.tensor(probabilities, dtype=t.float32)
    logits = t.log(probs_tensor)
    scaled_logits = logits / temperature
    scaled_probs = t.nn.functional.softmax(scaled_logits, dim=0)

    return scaled_probs.tolist()


class TemporalMatryoshkaBatchTopKSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, group_sizes: list[int], temporal: bool):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert sum(group_sizes) == dict_size, "group sizes must sum to dict_size"
        assert all(s > 0 for s in group_sizes), "all group sizes must be positive"

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.active_groups = len(group_sizes)
        group_indices = [0] + list(t.cumsum(t.tensor(group_sizes), dim=0))
        self.group_indices = group_indices

        self.register_buffer("group_sizes", t.tensor(group_sizes))

        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(t.zeros(dict_size))
        self.W_dec = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(dict_size, activation_dim)))
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.W_dec.data = set_decoder_norm_to_unit_norm(
            self.W_dec.data.T, activation_dim, dict_size
        ).T
        self.W_enc.data = self.W_dec.data.clone().T
        self.temporal = temporal

    def encode(self, x: t.Tensor, return_active: bool = False, use_threshold: bool = True):

        post_relu_feat_acts_BF = nn.functional.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
        else:
            # Flatten and perform batch top-k
            flattened_acts = post_relu_feat_acts_BF.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)
            encoded_acts_BF = (
                t.zeros_like(post_relu_feat_acts_BF.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts_BF.shape)
            )

        max_act_index = self.group_indices[self.active_groups]
        encoded_acts_BF[:, max_act_index:] = 0

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return x @ self.W_dec + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @t.no_grad()
    def scale_biases(self, scale: float):
        self.b_enc.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(cls, path, k=None, temporal=False, device=None, **kwargs) -> "TemporalMatryoshkaBatchTopKSAE":
        state_dict = t.load(path)
        activation_dim, dict_size = state_dict["W_enc"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        group_sizes = state_dict["group_sizes"].tolist()

        autoencoder = cls(activation_dim, dict_size, k=k, group_sizes=group_sizes, temporal=temporal)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class TemporalMatryoshkaBatchTopKTrainer(SAETrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        temporal: bool,
        contrastive: bool,
        layer: int,
        lm_name: str,
        group_fractions: list[float],
        group_weights: Optional[list[float]] = None,
        dict_class: type = TemporalMatryoshkaBatchTopKSAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        temp_alpha: float = 1 / 10,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "TemporalMatryoshkaBatchTopKSAE",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.temporal = temporal
        self.contrastive = contrastive
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        assert isclose(sum(group_fractions), 1.0), "group_fractions must sum to 1.0"
        # Calculate all groups except the last one
        group_sizes = [int(f * dict_size) for f in group_fractions[:-1]]
        # Put remainder in the last group
        group_sizes.append(dict_size - sum(group_sizes))

        if group_weights is None:
            group_weights = [(1.0 / len(group_sizes))] * len(group_sizes)

        assert len(group_sizes) == len(group_weights), (
            "group_sizes and group_weights must have the same length"
        )

        self.group_fractions = group_fractions
        self.group_sizes = group_sizes
        self.group_weights = group_weights

        self.ae = dict_class(activation_dim, dict_size, k, group_sizes, temporal)

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper

        self.temp_alpha = temp_alpha

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # We don't want to apply the bias
            x_reconstruct_aux = auxk_acts_BF @ self.ae.W_dec
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f: t.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=t.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x, step=None, logging=False):
        if self.temporal:
            f, active_indices_F, post_relu_acts_BF = self.ae.encode(
                x[:, 0], return_active=True, use_threshold=False
            )
            f_temp, active_indices_F_temp, post_relu_acts_BF_temp = self.ae.encode(
                x[:, 1], return_active=True, use_threshold=False
            )
            x_reconstruct = t.zeros_like(x[:, 0]) + self.ae.b_dec
        else:
            print("ERROR: must be temporal")

        if step > self.threshold_start_step:
            self.update_threshold(f)

        total_l2_loss = 0.0
        l2_losses = t.tensor([]).to(self.device)

        W_dec_chunks = t.split(self.ae.W_dec, self.ae.group_sizes.tolist(), dim=0)
        f_chunks = t.split(f, self.ae.group_sizes.tolist(), dim=1)

        ###OTHER VERSION
        start = 0
        if self.temporal:
            f_temp_chunks = t.split(f_temp, self.ae.group_sizes.tolist(), dim=1)

            x_temp_similarity = F.cosine_similarity(x[:, 0], x[:, 1], dim=-1)
            
            W_dec_slice = W_dec_chunks[0]
            acts_slice = f_chunks[0]
            x_reconstruct = x_reconstruct + acts_slice @ W_dec_slice

            l2_loss = ((x[:, 0] - x_reconstruct).pow(2).sum(dim=-1) * self.group_weights[0]).mean() 
            
            total_l2_loss += l2_loss
            l2_losses = t.cat([l2_losses, l2_loss.unsqueeze(0)])

            temp_loss = (t.abs(f_chunks[0] - f_temp_chunks[0]).sum(dim=-1) * x_temp_similarity * self.group_weights[0]).mean()

            ## Contrastive loss
            if self.contrastive:
                # x_temp_similarity = x_temp_similarity.unsqueeze(0)

                logits_per_curr_t =  (f_chunks[0] @ f_temp_chunks[0].T)#*(x_temp_similarity.T@t.ones_like(x_temp_similarity)).T
                logits_per_cumsum_t = logits_per_curr_t.T#*(x_temp_similarity.T@t.ones_like(x_temp_similarity))

                labels = t.arange(logits_per_curr_t.shape[0], device=self.device, dtype=t.long)            

                temp_loss = (
                    F.cross_entropy(logits_per_curr_t, labels) +
                    F.cross_entropy(logits_per_cumsum_t, labels)
                ) / 2

            start = 1
        else:
            temp_loss = t.tensor([0.0]).to(self.device)
        ###END

        for i in range(start, self.ae.active_groups):
            W_dec_slice = W_dec_chunks[i]
            acts_slice = f_chunks[i]
            x_reconstruct = x_reconstruct + acts_slice @ W_dec_slice

            if self.temporal:
                l2_loss = (x[:, 0] - x_reconstruct).pow(2).sum(dim=-1).mean() * self.group_weights[i]
            else:
                l2_loss = (x - x_reconstruct).pow(2).sum(dim=-1).mean() * self.group_weights[i]
            total_l2_loss += l2_loss
            l2_losses = t.cat([l2_losses, l2_loss.unsqueeze(0)])

        if self.temporal:
            x = x[:, 0]

        min_l2_loss = l2_losses.min().item()
        max_l2_loss = l2_losses.max().item()
        mean_l2_loss = l2_losses.mean()

        self.effective_l0 = self.k

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        auxk_loss = self.get_auxiliary_loss((x - x_reconstruct).detach(), post_relu_acts_BF)
        loss = mean_l2_loss + self.auxk_alpha * auxk_loss + self.temp_alpha * temp_loss


        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_reconstruct,
                f,
                {
                    "l2_loss": mean_l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "temp_loss": temp_loss.item(),
                    "loss": loss.item(),
                    "min_l2_loss": min_l2_loss,
                    "max_l2_loss": max_l2_loss,
                },
            )

    def update(self, step, x):
        if step == 0:
            if self.temporal:
                median = self.geometric_median(x[:, 0])
            else:
                median = self.geometric_median(x)
            self.ae.b_dec.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, self.ae.W_dec.grad.T, self.ae.activation_dim, self.ae.dict_size
        ).T
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "TemporalMatryoshkaBatchTopKTrainer",
            "dict_class": "TemporalMatryoshkaBatchTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "group_fractions": self.group_fractions,
            "group_weights": self.group_weights,
            "group_sizes": self.group_sizes,
            "k": self.ae.k.item(),
            "temporal": self.temporal,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break

        return guess
    
class TemporalBatchTopKSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, temporal: bool):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(t.zeros(activation_dim))
        self.temporal = temporal

    def encode(self, x: t.Tensor, return_active: bool = False, use_threshold: bool = True):
        if self.temporal:
            x = x[:, 0] #FIXME: ignoring temporal part

        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
        else:
            # Flatten and perform batch top-k
            flattened_acts = post_relu_feat_acts_BF.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)

            encoded_acts_BF = (
                t.zeros_like(post_relu_feat_acts_BF.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts_BF.shape)
            )

        ### CHANGED LINES
        # # Sum instead of flatten 
        # sequence_sum_acts = post_relu_feat_acts_BF.sum(1)
        # sequence_topk = sequence_sum_acts.topk(self.k * 128, sorted=False, dim=-1)
        # mask = (
        #     t.zeros_like(post_relu_feat_acts_BF.sum(1))
        #     .scatter_(-1, sequence_topk.indices, sequence_topk.values)
        # )
        # mask = (mask > 0.0).float()
        # print(mask.shape, post_relu_feat_acts_BF.shape)
        # encoded_acts_BF = mask * post_relu_feat_acts_BF
        # encoded_acts_BF = (encoded_acts_BF > self.threshold)
        ### END CHANGED LINES

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(cls, path, k=None, temporal=False, device=None, **kwargs) -> "BatchTopKSAE":
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = cls(activation_dim, dict_size, k, temporal)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class TemporalBatchTopKTrainer(SAETrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        temporal: bool,
        layer: int,
        lm_name: str,
        dict_class: type = TemporalBatchTopKSAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "TemporalBatchTopKSAE",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.temporal = temporal
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.ae = dict_class(activation_dim, dict_size, k, temporal)

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f: t.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=t.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def temporal_loss(self, f, f_temp, step=None, logging=False):

        pass

    def loss(self, x, step=None, logging=False):
        if self.temporal:
            f, active_indices_F, post_relu_acts_BF = self.ae.encode(
                x[:, 0], return_active=True, use_threshold=False
            )
            f_temp, active_indices_F_temp, post_relu_acts_BF_temp = self.ae.encode(
                x[:, 1], return_active=True, use_threshold=False
            )
        else:
            f, active_indices_F, post_relu_acts_BF = self.ae.encode(
                x, return_active=True, use_threshold=False
            )
        # l0 = (f != 0).float().sum(dim=-1).mean().item()

        if step > self.threshold_start_step:
            self.update_threshold(f)

        x_hat = self.ae.decode(f)

        if self.temporal:
            x = x[:, 0]
        e = x - x_hat

        self.effective_l0 = self.k

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()},
            )

    def update(self, step, x):
        if step == 0:
            if self.temporal:
                median = self.geometric_median(x[:, 0])
            else:
                median = self.geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "BatchTopKTrainer",
            "dict_class": "BatchTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "temporal": self.temporal,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break

        return guess
