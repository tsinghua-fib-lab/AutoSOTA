import numpy as np
import torch
import torch.nn as nn
from diff_models_table import diff_CSDI
import yaml
from torch import Tensor
import typing as ty
import torch.nn.init as nn_init
import pickle
import math

# partially stole from https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py
class Tokenizer(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()

        d_bias = d_numerical + len(categories)
        category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        self.d_token = d_token
        self.register_buffer("category_offsets", category_offsets)
        self.category_embeddings = nn.Embedding(sum(categories) + 1, self.d_token)
        self.category_embeddings.weight.requires_grad = False
        nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(Tensor(d_numerical, self.d_token))
        self.weight.requires_grad = False

        self.bias = nn.Parameter(Tensor(d_bias, self.d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            self.bias.requires_grad = False

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        x_cat = x_cat.type(torch.int32)

        assert x_some is not None
        x = self.weight.T * x_num

        if x_cat is not None:
            x = x[:, np.newaxis, :, :]
            x = x.permute(0, 1, 3, 2)
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=2,
            )
        if self.bias is not None:
            x = x + self.bias[None]

        return x

    def recover(self, Batch, d_numerical):
        B, L, K = Batch.shape
        L_new = int(L / self.d_token)
        Batch = Batch.reshape(B, L_new, self.d_token)
        Batch = Batch - self.bias

        Batch_numerical = Batch[:, :d_numerical, :]
        Batch_numerical = Batch_numerical / self.weight
        Batch_numerical = torch.mean(Batch_numerical, 2, keepdim=False)

        Batch_cat = Batch[:, d_numerical:, :]
        new_Batch_cat = torch.zeros([Batch_cat.shape[0], Batch_cat.shape[1]])
        for i in range(Batch_cat.shape[1]):
            token_start = self.category_offsets[i] + 1
            if i == Batch_cat.shape[1] - 1:
                token_end = self.category_embeddings.weight.shape[0] - 1
            else:
                token_end = self.category_offsets[i + 1]
            emb_vec = self.category_embeddings.weight[token_start : token_end + 1, :]
            for j in range(Batch_cat.shape[0]):
                distance = torch.norm(emb_vec - Batch_cat[j, i, :], dim=1)
                nearest = torch.argmin(distance)
                new_Batch_cat[j, i] = nearest + 1
            new_Batch_cat = new_Batch_cat.to(Batch_numerical.device)
        return torch.cat([Batch_numerical, new_Batch_cat], dim=1)


class CSDI_base(nn.Module):
    def __init__(self, exe_name, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        # load embedding vector dimension.
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        # For categorical variables
        self.mixed = config["model"]["mixed"]

        if exe_name == "census":
            with open("./data_census_ft/transformed_columns.pk", "rb") as f:
                cont_list, num_cate_list = pickle.load(f)

        self.cont_list = cont_list

        if self.mixed:
            self.token_dim = config["model"]["token_emb_dim"]

            # set tokenizer
            d_numerical = len(cont_list)
            categories = num_cate_list
            d_token = self.token_dim
            token_bias = True

            self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)

        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        tot_feature_num = len(cont_list) + len(num_cate_list)
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        # Perform forward step. Adding noise to all data.
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L*token_dim)

        target_mask = observed_mask - cond_mask
        target_mask = torch.repeat_interleave(target_mask, self.token_dim, dim=2)
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):

        cond_mask = torch.repeat_interleave(cond_mask, self.token_dim, dim=2)

        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        B, old_input_dim, K, L = total_input.shape
        total_input = total_input.reshape(
            B, old_input_dim, K, int(L / self.token_dim), self.token_dim
        )
        total_input = total_input.permute(0, 1, 4, 2, 3)
        total_input = total_input.reshape(
            B, old_input_dim * self.token_dim, K, int(L / self.token_dim)
        )

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):

        B, K, L = observed_data.shape
        cond_mask = torch.repeat_interleave(cond_mask, self.token_dim, dim=2)

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        # Perform n_samples times of forward and backward pass for same input data.
        for i in range(n_samples):
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                # perform T steps forward
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                        t
                    ] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)
            # perform T steps backward
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    # fix original x^{co} as condition
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    B, old_input_dim, K, L = diff_input.shape
                    diff_input = diff_input.reshape(
                        B, old_input_dim, K, int(L / self.token_dim), self.token_dim
                    )
                    diff_input = diff_input.permute(0, 1, 4, 2, 3)
                    diff_input = diff_input.reshape(
                        B, old_input_dim * self.token_dim, K, int(L / self.token_dim)
                    )

                predicted = self.diffmodel(
                    diff_input, side_info, torch.tensor([t]).to(self.device)
                )  # (B,K,L)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)

        # In testing, using `gt_mask` (generated with fixed missing rate) as cond_mask.
        # In training, generate random mask as cond_mask
        if is_train == 0:
            cond_mask = gt_mask
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)

        # The main calculation procedures are in `self.calc_loss()`
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            # gt_mask: 0 for missing elements and manully maksed elements
            cond_mask = gt_mask
            # target_mask: 1 for manually masked elements
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class TabCSDI(CSDI_base):
    def __init__(self, exe_name, config, device, target_dim=1):
        super().__init__(exe_name, target_dim, config, device)

    def process_data(self, batch):
        # Insert K=1 axis. All mask now with shape (B, 1, L). L=# of attributes.
        observed_data = batch["observed_data"][:, np.newaxis, :]
        observed_data = observed_data.to(self.device).float()
        observed_data = self.tokenizer(
            observed_data[:, :, self.cont_list],
            observed_data[:, :, len(self.cont_list) :],
        )
        B, K, L, C = observed_data.shape
        observed_data = observed_data.reshape(B, K, L * C)
        observed_mask = batch["observed_mask"][:, np.newaxis, :]
        observed_mask = observed_mask.to(self.device).float()

        observed_tp = batch["timepoints"].to(self.device).float()

        gt_mask = batch["gt_mask"][:, np.newaxis, :]
        gt_mask = gt_mask.to(self.device).float()

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


