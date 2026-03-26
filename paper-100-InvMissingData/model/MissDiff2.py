import gc
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from typing import Any, List
from torch.utils.data import Dataset, DataLoader
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item, ...]
    def __len__(self):
        return self.data.shape[0]

def time_embedding(pos, d_model=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device)
    position = pos.unsqueeze(2)
    div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(device) / d_model)
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe

def get_randmask(observed_mask):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    for i in range(len(observed_mask)):
        sample_ratio = np.random.rand()
        num_observed = observed_mask[i].sum().item()
        num_masked = round(num_observed * sample_ratio)
        rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask

############################
#  Fundamental Block for CSDI
############################
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=16, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    # t_embedding(t). The embedding dimension is 128 in total for every time step t.
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # Weight initialization
    nn.init.kaiming_normal_(layer.weight)
    return layer



def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # Temporal Transformer layer
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        # Feature Transformer layer
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        # (B*K, C, L) -> (L, B*K, C) -> (B*K, C, L)
        # input shape for transformerencoder: [seq, batch, emb]
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        # (B*L, C, K) -> (K, B*L, C) -> (B*L, C, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        # diffusion_emb is
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        # logger.info(f"the x shape: {x.shape}, diff embed: {diffusion_emb.shape}")
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)


        y = y

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip




class MissdiffMyCSDIT(nn.Module):
    def __init__(self, layer_number, n_channels, side_dim, diff_embedding, heads_num, diff_steps, device):
        super(MissdiffMyCSDIT, self).__init__()

        self.channels = n_channels
        self.input_projection = Conv1d_with_init(2, n_channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=diff_embedding,)
        self.device = device

        self.residual_layers = nn.ModuleList([ResidualBlock(side_dim=side_dim,
                                                            channels=self.channels,
                                                            diffusion_embedding_dim=diff_embedding,
                                                            nheads=heads_num,)
                                              for _ in range(layer_number)
                                              ])

    def forward(self, x, diffusion_step):


        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, int(K * L))
        # logger.info(f"the x dtype: {x.dtype}")
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_embed = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_embed)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, -1)

        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        return x.reshape(B, K, -1)



class MyCSDIT(nn.Module):
    def __init__(self, layer_number, n_channels, side_dim, diff_embedding, heads_num,
                 diff_steps,
                 schedule,
                 device):
        super(MyCSDIT, self).__init__()
        self.layer_number = layer_number
        self.n_channels = n_channels
        self.side_dim = side_dim
        self.diff_embedding = diff_embedding
        self.heads_num = heads_num
        self.diff_steps = diff_steps
        self.device = device


        assert schedule in ["quad", "linear"], f"the scheduel must in 'linear' or 'quad', but {schedule}"
        self.schedule = schedule
        if schedule == "quad":
            self.beta = np.linspace(0.0001 ** 0.5, 0.5 ** 0.5, self.diff_steps,) ** 2
        elif schedule == "linear":
            self.beta = np.linspace(0.0001, 0.5, self.diff_steps)


        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (torch.tensor(self.alpha).float().to(self.device).unsqueeze(1))
        self.diffusion_model = MissdiffMyCSDIT(layer_number=layer_number, n_channels=n_channels,
                        side_dim=side_dim, diff_embedding=diff_embedding, heads_num=heads_num,
                        diff_steps=diff_steps, device=device)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def set_input_to_diff(self, noisy_data, observed_data, cond_mask):
        # cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        # print(f"the cond obs: {cond_obs.shape}, noisy shape: {noisy_target.shape}, {noisy_data.shape}")
        total_input = torch.cat([cond_obs, noisy_target], dim=1).unsqueeze(-1)
        # total_input = torch.cat([noisy_target, noisy_target], dim=1).unsqueeze(-1)
        return total_input

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

    def impute(self, observed_data, cond_mask, n_samples):
        observed_data = observed_data.unsqueeze(-1)
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.diff_steps - 1, -1, -1):
                cond_observation = (cond_mask.unsqueeze(-1) * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask.unsqueeze(-1)) * current_sample).unsqueeze(1)
                diffusion_input = torch.cat([cond_observation, noisy_target], dim=1)
                predicted = self.diffusion_model(diffusion_input, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample = current_sample + sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        return torch.median(imputed_samples.squeeze(-1), dim=1).values#.median(dim=1)

    def cal_loss(self, observed_data, cond_mask, observed_mask, is_train, set_t=-1):
        # observed_data = observed_data.unsqueeze(-1)
        # cond_mask = cond_mask.unsqueeze(-1)
        B = observed_data.shape[0]

        if is_train != 1:
            # if we are not training
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            # if we are training
            t = torch.randint(0, self.diff_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        # logger.info(f"the current alpha shape: {current_alpha.shape}, observed data shape: {observed_data.shape}, noise: {noise.shape}")
        noisy_data = (current_alpha**0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        # print(f"obser shape: {observed_data.shape}, noise: {noise.shape}, current alpha: {current_alpha.shape}")
        total_input = self.set_input_to_diff(noisy_data, observed_data, cond_mask)
        # logger.info(f"the total input: {total_input.dtype}")
        # logger.info(f"尝试把东西喂进去!, total input: {total_input.shape}, t: {t.shape}")
        # sys.exit(0)
        predicted = self.diffusion_model(total_input, t).squeeze(-1)
        target_mask = observed_mask - cond_mask
        # print(f"the predicted shape: {predicted.shape}")
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def forward(self, batch, is_train=1, ground_truth_mask=None):
        if ground_truth_mask == None:
            ground_truth_mask = torch.ones_like(batch)
        cond_mask = ground_truth_mask
        return self.cal_loss(batch, cond_mask, torch.ones_like(batch), is_train)


class MissDiffImputation(object):
    def __init__(self, layer_number=4, n_channels=16, side_dim=32,
                 particle_number=50,
                 diff_embedding=62, heads_num=2,
                 batch_size=128,
                 epochs=500,
                 noise=1e-4,
                 lr=1.0e-3,
                 diff_steps=100, schedule="quad",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(MissDiffImputation, self).__init__()
        self.layer_number = layer_number
        self.n_channels = n_channels
        self.side_dim = side_dim
        self.diff_embedding = diff_embedding
        self.particle_number = particle_number
        self.heads_num = heads_num
        self.diff_steps = diff_steps
        self.noise = noise
        self.device = device
        self.csdi_t_model = MyCSDIT(layer_number=2, n_channels=16, side_dim=16, diff_embedding=16,
                     heads_num=2, diff_steps=20, schedule=schedule, device=device).to(device)
        self.optimizer = torch.optim.Adam(self.csdi_t_model.parameters(), lr=lr, weight_decay=1e-6)
        self.num_epoch = epochs
        self.batch_size = batch_size
        p0 = int(0.25 * epochs)
        p1 = int(0.5 * epochs)
        p2 = int(0.75 * epochs)
        p3 = int(0.9 * epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[p0, p1, p2, p3], gamma=0.1)

    def fit_transform(self, X:pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:

        X = torch.tensor(np.array(X), dtype=torch.float64).to(self.device)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape, device=self.device).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(self.device)

        # obtain the filled values
        X_filled = X.clone()
        X_filled[mask.bool()] = imps

        mask = mask.to(self.device)

        # the data loader for imputation dataset
        dataloader = DataLoader(MyDataset(data=torch.cat([X_filled.unsqueeze(-1), (1.0 - mask.unsqueeze(-1))], dim=-1)),
                                batch_size=self.batch_size,
                                shuffle=True)

        self.csdi_t_model.train()
        for epoch in range(self.num_epoch):
            self.csdi_t_model.train()

            for _, train_batch in enumerate(dataloader):
                # logger.info("进来做train了!")
                # logger.info(f"the train batch: {train_batch.shape}, dtype: {train_batch.dtype}")
                loss = self.csdi_t_model(train_batch[..., 0], 1, train_batch[..., 1],)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

        # empty the memory
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        imputed_list = []
        self.csdi_t_model.eval()
        with torch.no_grad():
            concat_data = torch.cat([X_filled.unsqueeze(-1), (1.0 - mask.unsqueeze(-1))], dim=-1)
            test_dataloader = DataLoader(MyDataset(data=concat_data), batch_size=self.batch_size, shuffle=False)
            for _, test_batch in enumerate(test_dataloader):
                imputed_value = self.csdi_t_model.impute(observed_data=test_batch[..., 0],
                                              cond_mask=test_batch[..., 1],
                                              n_samples=100)
                imputed_list.append(imputed_value)
            final_imputed_value = torch.cat(imputed_list, dim=0)
        final_imputed_value = final_imputed_value.detach().cpu().numpy()
        return final_imputed_value





if __name__ == "__main__":
    torch.set_default_tensor_type('torch.DoubleTensor')
    import sys
    from loguru import logger
    torch.manual_seed(seed=1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv_net = Conv1d_with_init(2, 3, 1).to(device)
    test_tensor = torch.randn([64, 2], device=device).unsqueeze(-1)

    output_tensor = conv_net(test_tensor)
    logger.info(f"the output tensor shape: {output_tensor.shape}")

    # specify the samples and tensors

    test_sample = torch.rand([64, 13], device=device)
    noisy_sample = torch.randn([64, 13], device=device)
    mask = torch.bernoulli(torch.empty(64, 13).uniform_(0, 1).to(device))
    # print(mask[0, :20])
    cond_obs = (mask * test_sample).unsqueeze(1)
    noisy_target = ((1 - mask) * noisy_sample).unsqueeze(1)
    total_input = torch.cat([cond_obs, noisy_target], dim=1).unsqueeze(-1)# .permute((0, 2, 1, 3))
    print(f"the total input shape: {total_input.shape}")
    # sys.exit(0)
    B = test_sample.shape[0]
    num_steps = 20
    t = torch.randint(0, num_steps, [B]).to(device)

    model = MissdiffMyCSDIT(layer_number=2, n_channels=16,
                        side_dim=16, diff_embedding=16, heads_num=2,
                        diff_steps=num_steps, device=device).to(device)

    output = model(total_input, t)
    logger.info(f"the output shape: {output.shape}")
    random_mask = get_randmask(observed_mask=torch.ones_like(test_sample))
    print(random_mask[0, :])


    csdi_t = MyCSDIT(layer_number=2, n_channels=16, side_dim=16, diff_embedding=16,
                     heads_num=2, diff_steps=20, schedule="quad", device=device).to(device)

    loss = csdi_t.forward(batch=test_sample)
    logger.info(f"the test sample shape: {test_sample.shape}")
    logger.info(f"the loss shape: {loss.shape}")

    with torch.no_grad():
        ground_truth_mask = torch.bernoulli(torch.empty(64, 13).uniform_(0, 1).to(device))
        given_data = torch.rand([64, 13], device=device) * ground_truth_mask + (1.0 - ground_truth_mask)
        given_data = given_data # .unsqueeze(-1)
        logger.info(f"the given data shape: {given_data.shape}")
        imputed_value = csdi_t.impute(observed_data=given_data, cond_mask=ground_truth_mask, n_samples=100)
        logger.info(f"the imputed shape: {imputed_value.shape}")


    my_model = MissDiffImputation(layer_number=4, n_channels=16, side_dim=32,
                 particle_number=50,
                 diff_embedding=32, heads_num=2,
                 batch_size=128,
                 epochs=10,
                 lr=1.0e-3,
                 diff_steps=100, schedule="quad",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    nan_ratio = 0.1  # 比如10%

    # 生成一个随机mask，标记哪些位置将被设置为NaN
    mask = torch.rand(64, 13) < nan_ratio

    # 将选中的位置设置为NaN
    test_tensor = torch.randn([64, 13], device=device, dtype=torch.float32)
    test_tensor[mask] = torch.nan

    imputed_value = my_model.fit_transform(X=test_tensor.cpu().numpy())
    logger.info(f"the imputed value shape: {imputed_value.shape}")














