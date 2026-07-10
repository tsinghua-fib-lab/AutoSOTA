# -*- coding: utf-8 -*-
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
warnings.filterwarnings('ignore')

try:
    from thop import profile

    HAS_THOP = True
except ImportError:
    HAS_THOP = False



def visual_alignment_dsd(G_s, G_t, folder_path, batch_idx=0, epoch_label='Test'):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    g_s_np = G_s.detach().cpu().numpy()
    g_t_np = G_t.detach().cpu().numpy()

    if batch_idx == 0:
        np.save(os.path.join(folder_path, f'gram_teacher.npy'), g_t_np)
        np.save(os.path.join(folder_path, f'gram_student_{epoch_label}.npy'), g_s_np)

    diff = np.abs(g_t_np - g_s_np)

    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 3)

    # 1. Teacher Geometry
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(g_t_np, ax=ax1, cmap='viridis', vmin=-1, vmax=1, cbar=True)
    ax1.set_title("Teacher Geometry (Macro Structure)", fontsize=12)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Sample Index")

    # 2. Student Geometry
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(g_s_np, ax=ax2, cmap='viridis', vmin=-1, vmax=1, cbar=True)
    ax2.set_title(f"Student Geometry ({epoch_label})", fontsize=12)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Sample Index")

    # 3. Difference
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(diff, ax=ax3, cmap='Reds', vmin=0, vmax=0.5, cbar=True)
    ax3.set_title("Alignment Gap (Diff)", fontsize=12)

    plt.suptitle(f"DSD Alignment Visualization - {epoch_label} - Batch {batch_idx}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    save_name = f'dsd_align_{epoch_label}_batch{batch_idx}.png'
    plt.savefig(os.path.join(folder_path, save_name), dpi=150)
    plt.close()



class SinkhornDistance(nn.Module):

    def __init__(self, eps=0.1, max_iter=20):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, x, y):
        # x: Student [B, K, D]
        # y: Teacher [B, P, D]

        B, K, D = x.shape
        _, P, _ = y.shape

        x_norm = F.normalize(x, dim=2)
        y_norm = F.normalize(y, dim=2)

        # C[b, k, p] = 1 - dot(x[b,k], y[b,p])
        C = 1.0 - torch.bmm(x_norm, y_norm.transpose(1, 2))


        mu = torch.ones(B, K, device=x.device) / K
        nu = torch.ones(B, P, device=x.device) / P

        # u = torch.zeros_like(mu)
        # v = torch.zeros_like(nu)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        K_mat = torch.exp(-C / self.eps)  # [B, K, P]

        for _ in range(self.max_iter):
            # Update u
            Kv = torch.bmm(K_mat, v.unsqueeze(-1)).squeeze(-1)
            u = mu / (Kv + 1e-8)

            # Update v
            KTu = torch.bmm(K_mat.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            v = nu / (KTu + 1e-8)

        T = u.unsqueeze(2) * K_mat * v.unsqueeze(1)
        dist = torch.sum(T * C, dim=(1, 2))  # [B]
        return dist.mean()



class DualStructuralAligner(nn.Module):
    def __init__(self, s_dim, t_dim, num_latent_tokens=4):
        super(DualStructuralAligner, self).__init__()

        # --- Macro Projector (SPKD) ---
        self.projector_global = nn.Linear(s_dim, t_dim)

        # --- Micro Projector (OT) ---
        self.num_latent_tokens = num_latent_tokens
        self.projector_local = nn.Sequential(
            nn.Linear(s_dim, t_dim * num_latent_tokens),
            nn.LayerNorm(t_dim * num_latent_tokens),
            nn.GELU(),
            nn.Linear(t_dim * num_latent_tokens, t_dim * num_latent_tokens)
        )

        self.mse_loss = nn.MSELoss()
        self.ot_solver = SinkhornDistance(eps=0.1, max_iter=20)

    def forward(self, s_feat, t_feat_patch, t_feat_mean):

        s_flat = s_feat.reshape(-1, s_feat.shape[-1])
        t_flat = t_feat_mean.reshape(-1, t_feat_mean.shape[-1])

        s_proj_global = self.projector_global(s_flat)

        # L2 Normalize
        s_norm = F.normalize(s_proj_global, p=2, dim=1)
        t_norm = F.normalize(t_flat, p=2, dim=1)

        # Batch Similarity Matrix (Batch-wise Relation)
        s_batch_view = s_norm.reshape(s_feat.shape[0], -1)
        t_batch_view = t_norm.reshape(t_feat_mean.shape[0], -1)

        s_batch_view = F.normalize(s_batch_view, p=2, dim=1)
        t_batch_view = F.normalize(t_batch_view, p=2, dim=1)

        G_s = torch.mm(s_batch_view, s_batch_view.t())  # [B, B]
        G_t = torch.mm(t_batch_view, t_batch_view.t())  # [B, B]

        loss_macro = self.mse_loss(G_s, G_t)


        loss_micro = torch.tensor(0.0, device=s_feat.device)


        if len(t_feat_patch.shape) == 4 and t_feat_patch.shape[-1] > 1:
            P = t_feat_patch.shape[-1]
            t_local = t_feat_patch.permute(0, 1, 3, 2).reshape(-1, P, t_feat_patch.shape[2])

            s_local_proj = self.projector_local(s_feat)  # [B, N, K*D]
            s_local = s_local_proj.reshape(-1, self.num_latent_tokens, t_feat_patch.shape[2])

            loss_micro = self.ot_solver(s_local, t_local)


        return loss_macro, loss_micro, G_s, G_t



class FrequencyAlignmentLoss(nn.Module):
    """ALGO-04: Frequency-domain alignment loss via FFT magnitude spectra.

    Computes MSE between |FFT(s_out)| and |FFT(t_out)| over the time dimension.
    Captures periodic structure (daily/weekly cycles) that time-domain MSE misses.
    """
    def __init__(self):
        super(FrequencyAlignmentLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, s_out, t_out):
        # s_out, t_out: [B, P, C] where P is prediction horizon
        # Compute FFT along the time dimension (dim=1)
        s_fft = torch.fft.rfft(s_out.float(), dim=1)
        t_fft = torch.fft.rfft(t_out.float(), dim=1)
        # Compare magnitude spectra (real-valued)
        s_mag = torch.abs(s_fft)
        t_mag = torch.abs(t_fft)
        return self.mse(s_mag, t_mag)

class Exp_Hetero_DSD(Exp_Basic):
    def __init__(self, args):
        super(Exp_Hetero_DSD, self).__init__(args)


        self.result_file = getattr(self.args, 'result_file', 'result_long_term_forecast.csv')


        self.lambda_kd = getattr(self.args, 'lambda_kd', 1.0)

        self.lambda_align = getattr(self.args, 'lambda_align', 1.0)
        self.alpha_ot = getattr(self.args, 'alpha_ot', 0.5)

        self.kd_gamma = float(getattr(self.args, 'kd_gamma', 0.5))
        self.kd_time_weight_type = getattr(self.args, 'kd_time_weight_type', 'linear')
        self.kd_w_max = float(getattr(self.args, 'kd_w_max', 2.0))

        self.teacher_ckpt = getattr(self.args, 'teacher_ckpt', None)

        self.lambda_freq = float(getattr(self.args, 'lambda_freq', 0.05))
        self.freq_loss_fn = FrequencyAlignmentLoss().to(self.device) if self.lambda_freq > 0 else None

        self._build_teacher()


        if hasattr(self, 'teacher_args'):
            s_dim = self.args.d_model
            t_dim = self.teacher_args.d_model
            self.aligner = DualStructuralAligner(s_dim, t_dim, num_latent_tokens=4).to(self.device)
            print(f">>> [Aligner] Initialized: Dual-Structural (DSD)")
        else:
            print(">>> [Error] Teacher args not found, Aligner skipped!")

    def _build_teacher(self):
        import copy
        from collections import OrderedDict

        print(f">>> [Teacher] Loading from {self.teacher_ckpt} ...")
        ckpt = torch.load(self.teacher_ckpt, map_location='cpu')

        if isinstance(ckpt, dict) and 'args' in ckpt:
            t_args = ckpt['args']
        else:
            t_args = copy.deepcopy(self.args)
            t_args.model = getattr(self.args, 'teacher_model', 'PatchTST')

        t_args.task_name = 'long_term_forecast'
        TeacherModel = self.model_dict[t_args.model].Model
        teacher = TeacherModel(t_args).float()

        state_dict = ckpt['model'] if (isinstance(ckpt, dict) and 'model' in ckpt) else ckpt
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model_state = teacher.state_dict()
        filtered_state = {k: v for k, v in new_state_dict.items() if
                          k in model_state and v.shape == model_state[k].shape}
        teacher.load_state_dict(filtered_state, strict=False)

        teacher.eval()
        for p in teacher.parameters(): p.requires_grad = False
        if self.args.use_gpu: teacher.to(self.device)
        self.teacher = teacher
        self.teacher_args = t_args

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)


        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.total_params = total_params
        self.trainable_params = trainable_params

        print(">>> [Student] Model: {}".format(self.args.model))
        print(">>> [Student] Total params: {:.2f} M".format(total_params / 1e6))
        print(">>> [Student] Trainable params: {:.2f} M".format(trainable_params / 1e6))
        print(">>> Seq_len = {}, Pred_len = {}, d_model = {}, e_layers = {}, d_ff = {}".format(
            self.args.seq_len,
            self.args.pred_len,
            getattr(self.args, 'd_model', None),
            getattr(self.args, 'e_layers', None),
            getattr(self.args, 'd_ff', None),
        ))
        return model

    def _select_optimizer(self):
        model_params = list(self.model.parameters())
        if hasattr(self, 'aligner'):
            model_params += list(self.aligner.parameters())
        return optim.Adam(model_params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_kd_time_weights(self, length, device):
        if self.kd_time_weight_type == 'none' or self.kd_w_max <= 1.0:
            return torch.ones(1, length, 1, device=device)
        L = length
        s = torch.linspace(0.0, 1.0, steps=L, device=device)
        t_type = self.kd_time_weight_type
        if t_type == 'linear':
            w = 1.0 + (self.kd_w_max - 1.0) * s
        elif t_type == 'exp':
            num = torch.exp(self.kd_beta * s) - 1.0
            den = torch.exp(torch.tensor(self.kd_beta, device=device)) - 1.0 + 1e-8
            w = 1.0 + (self.kd_w_max - 1.0) * (num / den)
        elif t_type == 'poly':
            w = 1.0 + (self.kd_w_max - 1.0) * (s ** self.kd_poly_p)
        elif t_type == 'tail':
            w = torch.ones(L, device=device)
            if L > 0 and self.kd_tail_ratio > 0.0:
                idx = max(0, min(int((1.0 - self.kd_tail_ratio) * L), L - 1))
                w[idx:] = self.kd_w_max
        elif t_type == 'cosine':
            w = 1.0 + (self.kd_w_max - 1.0) * (1.0 - torch.cos(torch.pi * s)) / 2.0
        else:
            w = torch.ones(L, device=device)
        return w.view(1, L, 1)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp: scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            if hasattr(self, 'aligner'): self.aligner.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Teacher Forward
                with torch.no_grad():
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            t_out, t_feat = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                         return_feature=True)
                    else:
                        t_out, t_feat = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_feature=True)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    t_out = t_out[:, -self.args.pred_len:, f_dim:]

                # Student Forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        s_out, s_feat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_feature=True)
                        s_out = s_out[:, -self.args.pred_len:, f_dim:]
                        y_true = batch_y[:, -self.args.pred_len:, f_dim:]

                        # Loss A
                        loss_gt = criterion(s_out, y_true)

                        # Loss B
                        loss_kd_out = torch.tensor(0.0, device=self.device)
                        if self.lambda_kd > 0:
                            w_static = self._get_kd_time_weights(self.args.pred_len, s_out.device)
                            if self.kd_gamma > 0:
                                with torch.no_grad():
                                    t_err = torch.mean(torch.abs(t_out - y_true), dim=-1, keepdim=True)
                                    conf_gate = torch.exp(-self.kd_gamma * t_err)
                            else:
                                conf_gate = 1.0
                            w_final = w_static * conf_gate
                            loss_kd_out = torch.mean(((s_out - t_out) ** 2) * w_final)

                        # Loss C
                        l_sp = torch.tensor(0.0, device=self.device)
                        l_ot = torch.tensor(0.0, device=self.device)
                        G_s, G_t = None, None

                        loss_align = torch.tensor(0.0, device=self.device)

                        if hasattr(self, 'aligner'):

                            if isinstance(t_feat,list):
                                t_feat=t_feat[-1]


                            if len(t_feat.shape) == 4:

                                if t_feat.shape[-1] == self.teacher_args.d_model:
                                    t_feat = t_feat.permute(0, 1, 3, 2)
                                t_feat_patch = t_feat
                                t_feat_mean = t_feat.mean(dim=-1)
                            else:

                                t_feat_mean = t_feat[:, :self.args.enc_in, :]
                                t_feat_patch = t_feat_mean.unsqueeze(-1)


                            l_sp, l_ot, G_s, G_t = self.aligner(s_feat, t_feat_patch, t_feat_mean)

                            weighted_ot = torch.tensor(0.0, device=self.device)
                            if len(t_feat.shape) == 4:  # I_{P>1}
                                weighted_ot = self.alpha_ot * l_ot

                            # Alignment Loss
                            loss_align = self.lambda_align * (l_sp + weighted_ot)

                        # Total Loss
                        loss = (1 - self.lambda_kd) * loss_gt + self.lambda_kd * loss_kd_out + loss_align
                    if self.lambda_freq > 0:
                        loss_freq = self.freq_loss_fn(s_out, t_out)
                        loss = loss + self.lambda_freq * loss_freq
                        if self.lambda_freq > 0:
                            loss_freq = self.freq_loss_fn(s_out, t_out)
                            loss = loss + self.lambda_freq * loss_freq
                else:

                    s_out, s_feat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_feature=True)
                    s_out = s_out[:, -self.args.pred_len:, f_dim:]
                    y_true = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss_gt = criterion(s_out, y_true)

                    loss_kd_out = torch.tensor(0.0, device=self.device)
                    if self.lambda_kd > 0:
                        w_static = self._get_kd_time_weights(self.args.pred_len, s_out.device)
                        if self.kd_gamma > 0:
                            with torch.no_grad():
                                t_err = torch.mean(torch.abs(t_out - y_true), dim=-1, keepdim=True)
                                conf_gate = torch.exp(-self.kd_gamma * t_err)
                        else:
                            conf_gate = 1.0
                        w_final = w_static * conf_gate
                        loss_kd_out = torch.mean(((s_out - t_out) ** 2) * w_final)

                    l_sp = torch.tensor(0.0, device=self.device)
                    l_ot = torch.tensor(0.0, device=self.device)
                    G_s, G_t = None, None

                    loss_align = torch.tensor(0.0, device=self.device)

                    if hasattr(self, 'aligner'):

                        if isinstance(t_feat, list):
                            t_feat = t_feat[-1]


                        if len(t_feat.shape) == 4:

                            if t_feat.shape[-1] == self.teacher_args.d_model:
                                t_feat = t_feat.permute(0, 1, 3, 2)
                            t_feat_patch = t_feat
                            t_feat_mean = t_feat.mean(dim=-1)
                        else:

                            t_feat_mean = t_feat[:, :self.args.enc_in, :]
                            t_feat_patch = t_feat_mean.unsqueeze(-1)

                        l_sp, l_ot, G_s, G_t = self.aligner(s_feat, t_feat_patch, t_feat_mean)

                        weighted_ot = torch.tensor(0.0, device=self.device)
                        if len(t_feat.shape) == 4:  # I_{P>1}
                            weighted_ot = self.alpha_ot * l_ot

                        # Alignment Loss
                        loss_align = self.lambda_align * (l_sp + weighted_ot)

                    # Total Loss
                    loss = (1 - self.lambda_kd) * loss_gt + self.lambda_kd * loss_kd_out + loss_align

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} | "
                          f"GT: {loss_gt.item():.4f}, SPKD: {l_sp.detach().item():.4f}, OT: {l_ot.detach().item():.4f}")
                    iter_count = 0
                    time_now = time.time()


                if i == 0 and epoch == 0 and G_s is not None:
                    visual_path = os.path.join(self.args.checkpoints, setting, 'vis_dsd')
                    print(f">>> [Vis] Saving DSD Geometry Heatmap to {visual_path}")
                    visual_alignment_dsd(G_s, G_t, visual_path, batch_idx=i, epoch_label='Epoch_0')

                # Optimization
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop: break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, tuple): outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, tuple): outputs = outputs[0]
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []

        # Statistics
        total_time = 0.0
        total_batches = 0
        warmup = 3
        if self.args.use_gpu: torch.cuda.reset_peak_memory_stats()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if hasattr(self, 'teacher'): self.teacher.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if i >= warmup:
                    if self.args.use_gpu: torch.cuda.synchronize()
                    start = time.perf_counter()

                # Inference
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        s_out, s_feat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_feature=True)

                        if hasattr(self, 'teacher') and i < 3:
                            _, t_feat = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_feature=True)
                else:
                    s_out, s_feat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_feature=True)
                    if hasattr(self, 'teacher') and i < 3:
                        _, t_feat = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_feature=True)

                if i >= warmup:
                    if self.args.use_gpu: torch.cuda.synchronize()
                    end = time.perf_counter()
                    total_time += (end - start)
                    total_batches += 1


                if i < 3 and hasattr(self, 'aligner') and hasattr(self, 'teacher'):
                    visual_path = './visual_results/' + setting + '/'

                    if isinstance(t_feat, list):
                        t_feat = t_feat[-1]

                    if len(t_feat.shape) == 4:
                        if t_feat.shape[-1] == self.teacher_args.d_model:
                            t_feat = t_feat.permute(0, 1, 3, 2)
                        t_feat_patch = t_feat
                        t_feat_mean = t_feat.mean(dim=-1)
                    else:
                        t_feat_mean = t_feat[:, :self.args.enc_in, :]
                        t_feat_patch = t_feat_mean.unsqueeze(-1)


                    _, _, G_s, G_t = self.aligner(s_feat, t_feat_patch, t_feat_mean)

                    print(f">>> [Vis] Drawing DSD Heatmap for batch {i}")
                    visual_alignment_dsd(G_s, G_t, visual_path, batch_idx=i, epoch_label='Final_Test')


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = s_out[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))



                if i == 0 and HAS_THOP:
                    dummy_x = batch_x
                    dummy_x_mark = batch_x_mark
                    dummy_dec = dec_inp
                    dummy_y_mark = batch_y_mark

                    macs, params = profile(
                        self.model,
                        inputs=(dummy_x, dummy_x_mark, dummy_dec, dummy_y_mark),
                        verbose=False
                    )
                    print(">>> [thop] MACs: {:.2f} G, Params: {:.2f} M".format(
                        macs / 1e9, params / 1e6
                    ))
                    self.macs = macs
                    self.params_profile = params

                elif i == 0 and not HAS_THOP:
                    print(">>> thop not installed, skip MACs/Params profiling. `pip install thop` if needed.")

        if total_batches > 0:
            avg_time = total_time / total_batches
            self.throughput = self.args.batch_size / avg_time
            self.avg_time_per_sample = avg_time / self.args.batch_size

        else:
            self.throughput = 0
            self.avg_time_per_sample = 0

        if self.args.use_gpu:
            self.max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            self.max_mem = 0

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))


        print(f">>> Writing results to :{self.result_file} ...")
        result_dir = os.path.dirname(self.result_file)
        if result_dir and not os.path.exists(result_dir):
            os.makedirs(result_dir)


        # Result Write
        with open(self.result_file, "a") as f:
            f.write(setting + "  \n")

            # Student Config
            f.write(
                "[Student Config] Seq_len = {}, Label_len = {}, Pred_len = {}, "
                "d_model = {}, e_layers = {}, d_ff = {}, "
                "enc_in = {}, c_out = {}\n".format(
                    self.args.seq_len, self.args.label_len, self.args.pred_len,
                    getattr(self.args, "d_model", "N/A"), getattr(self.args, "e_layers", "N/A"),
                    getattr(self.args, "d_ff", "N/A"), getattr(self.args, "enc_in", "N/A"),
                    getattr(self.args, "c_out", "N/A"),
                )
            )

            # DSD Config
            teacher_model = getattr(self.args, 'teacher_model', 'None')
            lambda_kd = getattr(self.args, 'lambda_kd', 0.0)
            kd_gamma = getattr(self.args, 'kd_gamma', 0.0)

            lambda_align = getattr(self.args, 'lambda_align', 0.0)
            alpha_ot = getattr(self.args, 'alpha_ot', 0.0)

            f.write(
                "[Distillation Config] Teacher: {}, Lambda_KD: {}, "
                "Adapt_Gamma: {}, lambda_align: {}, Alpha_OT: {}\n".format(
                    teacher_model, lambda_kd, kd_gamma, lambda_align, alpha_ot
                )
            )

            # Metrics
            f.write(
                "[Metrics_Detail] mse:{}, mae:{}, dtw:{}, "
                "params:{:.2f}M, trainable_params:{:.2f}M, "
                "latency:{:.3f}ms/sample, throughput:{:.1f} samples/s, "
                "max_mem:{:.1f}MB, "
                "macs:{:.4f}G\n".format(
                    mse, mae, 'Not calculated',
                    getattr(self, "total_params", 0) / 1e6,
                    getattr(self, "trainable_params", 0) / 1e6,
                    getattr(self, "avg_time_per_sample", 0.0) * 1000,
                    getattr(self, "throughput", 0.0),
                    getattr(self, "max_mem", 0.0),
                    getattr(self, "macs", 0.0) / 1e9,
                )
            )
            f.write("[Result_Brief]   mse:{:.3f}, mae:{:.3f}\n".format(mse, mae))
            f.write("\n")

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return