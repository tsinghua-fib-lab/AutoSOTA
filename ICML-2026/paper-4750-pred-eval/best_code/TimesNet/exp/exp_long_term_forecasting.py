from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== Module-level coherence helpers ====================

def coherence_nmse_lb_gpu_batched(
    x_tail_bcT: torch.Tensor,   # [B, C, Np]
    y_head_bcT: torch.Tensor,   # [B, C, Np]
    win_len: int,
    overlap: float = 0.5,
    eps: float = 1e-8,
    chunk_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert x_tail_bcT.shape == y_head_bcT.shape, "x_tail and y_head must have same [B,C,Np]"
    device = x_tail_bcT.device
    B, C, Np = x_tail_bcT.shape

    M = max(32, int(win_len))
    M = min(M, Np)
    if M < 8:
        nmse_lb = torch.ones((B, C), device=device, dtype=torch.float32)
        P_lin   = torch.zeros((B, C), device=device, dtype=torch.float32)
        return nmse_lb, P_lin

    hop = max(1, int(M * (1.0 - overlap)))
    w = torch.hann_window(M, periodic=True, device=device, dtype=torch.float32)
    U = (w * w).sum() + eps

    if (chunk_channels is None) or (chunk_channels <= 0):
        chunk_channels = C

    nmse_chunks, plin_chunks = [], []

    for c0 in range(0, C, chunk_channels):
        c1 = min(C, c0 + chunk_channels)
        x = x_tail_bcT[:, c0:c1, :].to(torch.float32)
        y = y_head_bcT[:, c0:c1, :].to(torch.float32)

        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)

        xw = x.unfold(dimension=-1, size=M, step=hop)
        yw = y.unfold(dimension=-1, size=M, step=hop)
        if xw.size(-2) == 0:
            nmse_chunks.append(torch.ones((B, x.size(1)), device=device, dtype=torch.float32))
            plin_chunks.append(torch.zeros((B, x.size(1)), device=device, dtype=torch.float32))
            continue

        xw = xw * w
        yw = yw * w

        # rFFT -> [B, Cc, S, Kf]
        X = torch.fft.rfft(xw, dim=-1)
        Y = torch.fft.rfft(yw, dim=-1)

        Px  = (X.abs()**2) / U
        Py  = (Y.abs()**2) / U
        Pxy = (X * torch.conj(Y)) / U

        Sxx = Px.mean(dim=-2)  # [B, Cc, Kf]
        Syy = Py.mean(dim=-2)
        Sxy = Pxy.mean(dim=-2)

        gamma2 = (Sxy.abs()**2) / (Sxx.clamp_min(eps) * Syy.clamp_min(eps))
        gamma2 = gamma2.clamp(0.0, 1.0)

        num = (Syy * gamma2).sum(dim=-1)     # [B, Cc]
        den = Syy.sum(dim=-1).clamp_min(eps) # [B, Cc]
        P_lin = (num / den).clamp(0.0, 1.0)  # [B, Cc]
        nmse  = 1.0 - P_lin

        nmse_chunks.append(nmse)
        plin_chunks.append(P_lin)

    nmse_lb = torch.cat(nmse_chunks, dim=1)  # [B, C]
    P_lin   = torch.cat(plin_chunks, dim=1)  # [B, C]
    return nmse_lb, P_lin


def coherence_nmse_lb_cpu_numpy(
    x_tail: np.ndarray,
    y_head: np.ndarray,
    win_len: int,
    overlap: float = 0.5,
    eps: float = 1e-8,
) -> tuple[float, float]:

    x = np.asarray(x_tail, dtype=np.float64)
    y = np.asarray(y_head, dtype=np.float64)
    N = min(len(x), len(y))
    if N < 8:
        return 1.0, 0.0

    x = x[-N:] - np.mean(x[-N:])
    y = y[:N]  - np.mean(y[:N])

    M = max(32, int(win_len))
    M = min(M, N)
    if M < 8:
        return 1.0, 0.0

    hop = max(1, int(M * (1.0 - overlap)))
    w = np.hanning(M)
    U = (w**2).sum() + eps

    Sxx = Syy = Sxy = None
    segs = 0
    for st in range(0, N - M + 1, hop):
        xs = x[st:st+M] * w
        ys = y[st:st+M] * w
        X = np.fft.rfft(xs)
        Y = np.fft.rfft(ys)
        Px  = (np.abs(X)**2) / U
        Py  = (np.abs(Y)**2) / U
        Pxy = (X * np.conj(Y)) / U
        if Sxx is None:
            Sxx, Syy, Sxy = Px, Py, Pxy
        else:
            Sxx += Px; Syy += Py; Sxy += Pxy
        segs += 1

    if segs == 0:
        return 1.0, 0.0

    Sxx /= segs; Syy /= segs; Sxy /= segs
    gamma2 = (np.abs(Sxy)**2) / ((Sxx + eps) * (Syy + eps))
    gamma2 = np.clip(gamma2, 0.0, 1.0)

    num = (Syy * gamma2).sum()
    den = (Syy + eps).sum()
    P_lin = float(num / (den + eps))
    NMSE_lb = 1.0 - P_lin
    return float(NMSE_lb), float(P_lin)



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # For PS loss
        self.ps_lambda = args.ps_lambda
        self.use_ps_loss = args.use_ps_loss
        self.patch_len_threshold = args.patch_len_threshold
        self.kl_loss = nn.KLDivLoss(reduction='none')


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def create_patches(self, x, patch_len, stride):
        
        x = x.permute(0, 2, 1) # [B, C, L] -> [B, L, C]
        B, C, L = x.shape
        
        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)
        
        return patches

    def fouriour_based_adaptive_patching(self, true, pred):

        # Get patch length an stride
        true_fft = torch.fft.rfft(true, dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)
        period = (true.shape[1] // top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        stride = patch_len // 2
        
        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        return true_patch, pred_patch
    
    def patch_wise_structural_loss(self, true_patch, pred_patch):
        
        # Calculate mean
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        
        # Calculate variance and standard deviation
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var)
        pred_patch_std = torch.sqrt(pred_patch_var)
        
        # Calculate Covariance
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)
        
        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
        
        # 3. Mean loss
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()
        
        return linear_corr_loss, var_loss, mean_loss

    def ps_loss(self, true, pred):

        # Fourior based adaptive patching
        true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred)
        
        # Pacth-wise structural loss
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)
        
        # Gradient based dynamic weighting
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)

        # Final PS loss
        ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss
        
        return ps_loss
    
    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):
        
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var)
        pred_std = torch.sqrt(pred_var)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)
   
        # Gradiant based dynamic weighting
        corr_gradient = torch.autograd.grad(corr_loss, self.model.projection.parameters(), create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, self.model.projection.parameters(), create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, self.model.projection.parameters(), create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta =  gradiant_avg.norm().detach() /  var_gradient.norm().detach()
        gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()
        
        return aplha, beta, gamma
    

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        
        self.model.train()
        
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            mse_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = criterion(outputs, batch_y)
                    
                    # Add PS Loss
                    mse_losses.append(loss.item())
                    if self.use_ps_loss:
                        ps_loss = self.ps_loss(batch_y, outputs)
                        loss += ps_loss * self.ps_lambda
                    
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

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
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cuda:0')))

        return self.model

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

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
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def evaluate_predictability(
        self,
        setting: str,
        load: bool = True,
        alpha_boundary: float = 1,
        welch_win_frac: float = 0.25,
        welch_overlap: float = 0.5,
        workers: int = 8,
        outdir: str = "./predictability_results",
        limit_batches: int | None = None,
        use_cuda_coherence: bool = True,     
        coh_chunk_channels: int | None = 32, 
    ):

        test_data, test_loader = self._get_data(flag='test')
        if load:
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            else:
                print(f"[WARN] checkpoint not found: {ckpt_path} (use current weights)")
        self.model.eval()

        dataname = str(self.args.data_path)[:-4]
        subdir = (
            f"{dataname}/"
            f"sl{self.args.seq_len}_ll{self.args.label_len}_pl{self.args.pred_len}/"
            f"alpha{alpha_boundary}_w{welch_win_frac}_ov{welch_overlap}/"
            f"{self.args.model}"
        )
        root_out = Path(outdir) / subdir / setting
        root_out.mkdir(parents=True, exist_ok=True)

        rows = []
        sample_id_global = 0
        f_dim = -1 if self.args.features == 'MS' else 0
        eps_var = 1e-12

        total_batches = len(test_loader)
        if limit_batches is not None:
            total_batches = min(total_batches, limit_batches)

        with torch.no_grad():
            for bi, batch in enumerate(islice(iter(test_loader), total_batches)):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs      = outputs[:, -self.args.pred_len:, f_dim:]  # [B, T, C]
                batch_y_eval = batch_y[:, -self.args.pred_len:, f_dim:]  # [B, T, C]
                batch_x_eval = batch_x[:, :, f_dim:]                     # [B, Tx, C]

                B, T, C = outputs.shape
                Np = max(8, int(alpha_boundary * min(batch_x_eval.shape[1], T)))
                win_len = max(32, int(welch_win_frac * Np))

                y_true_front = batch_y_eval[:, :Np, :]        # [B, Np, C]
                y_pred_front = outputs[:, :Np, :]             # [B, Np, C]
                x_tail       = batch_x_eval[:, -Np:, :]       # [B, Np, C]

                if use_cuda_coherence and self.args.use_gpu and torch.cuda.is_available():

                    x_tail_bcT = x_tail.permute(0, 2, 1).contiguous()        # [B, C, Np]
                    y_head_bcT = y_true_front.permute(0, 2, 1).contiguous()  # [B, C, Np]

                    nmse_lb_bc, P_lin_bc = coherence_nmse_lb_gpu_batched(
                        x_tail_bcT, y_head_bcT,
                        win_len=win_len,
                        overlap=welch_overlap,
                        eps=1e-8,
                        chunk_channels=coh_chunk_channels,
                    )  # [B, C], [B, C]

                    var_y_bc = y_true_front.var(dim=1, unbiased=False)            # [B, C]
                    mu_y_bc  = y_true_front.mean(dim=1)                           # [B, C]
                    mu_x_bc  = x_tail.mean(dim=1)                                 # [B, C]

                    mse_model_bc = ((y_true_front - y_pred_front)**2).mean(dim=1) # [B, C]
                    mae_model_bc = (y_true_front - y_pred_front).abs().mean(dim=1)# [B, C]

                    mse_lb_bc    = nmse_lb_bc * var_y_bc + (mu_y_bc - mu_x_bc)**2 # [B, C]

                    # nmse_model_bc = mse_model_bc / (var_y_bc + eps_var)     # [B, C]
                    tau = 1e-2
                    nmse_model_bc = torch.where(var_y_bc < tau, torch.full_like(mse_model_bc, float('nan')),
                                                mse_model_bc / (var_y_bc + 1e-6))


                    nmse_np   = nmse_lb_bc.detach().cpu().numpy()
                    plin_np   = P_lin_bc.detach().cpu().numpy()
                    mselb_np  = mse_lb_bc.detach().cpu().numpy()
                    msem_np   = mse_model_bc.detach().cpu().numpy()
                    mae_np    = mae_model_bc.detach().cpu().numpy()
                    vary_np   = var_y_bc.detach().cpu().numpy()

                    nmse_model_np = nmse_model_bc.detach().cpu().numpy()

                    for b in range(B):
                        for c in range(C):
                            rows.append({
                                "sample_id": sample_id_global + b,
                                "channel": int(c),
                                "MSE_model": float(msem_np[b, c]),
                                "MAE_model": float(mae_np[b, c]),
                                "NMSE_model": float(nmse_model_np[b, c]),  
                                "MSE_lb": float(mselb_np[b, c]),
                                "NMSE_lb": float(nmse_np[b, c]),
                                "P_lin": float(plin_np[b, c]),
                                "Var_y": float(vary_np[b, c]),
                                "Np": int(Np),
                            })


                else:

                    y_true_np = y_true_front.detach().cpu().numpy()
                    y_pred_np = y_pred_front.detach().cpu().numpy()
                    x_hist_np = x_tail.detach().cpu().numpy()

                    def _one_channel_metrics(bi_, ci_):
                        y_t = y_true_np[bi_, :, ci_]     # [Np]
                        y_p = y_pred_np[bi_, :, ci_]
                        x_c = x_hist_np[bi_, :, ci_]

                        nmse_lb, P_lin = coherence_nmse_lb_cpu_numpy(
                            x_tail=x_c, y_head=y_t, win_len=win_len, overlap=welch_overlap, eps=1e-8
                        )
                        var_y     = float(np.var(y_t))
                        mse_lb    = float(nmse_lb) * var_y + float((np.mean(y_t) - np.mean(x_c))**2)
                        mse_model = float(np.mean((y_t - y_p)**2))
                        mae_model = float(np.mean(np.abs(y_t - y_p)))
                        nmse_model = float(mse_model / max(var_y, eps_var))


                        return {
                            "sample_id": sample_id_global + bi_,
                            "channel": int(ci_),
                            "MSE_model": mse_model,
                            "MAE_model": mae_model,
                            "NMSE_model": nmse_model, 
                            "MSE_lb": mse_lb,
                            "NMSE_lb": float(nmse_lb),
                            "P_lin": float(P_lin),
                            "Var_y": var_y,
                            "Np": int(Np),
                        }

                    jobs = []
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        for b in range(B):
                            for c in range(C):
                                jobs.append(ex.submit(_one_channel_metrics, b, c))
                        for fut in as_completed(jobs):
                            rows.append(fut.result())

                sample_id_global += B
                if (bi + 1) % 10 == 0:
                    print(f"[eval] batch {bi+1}/{len(test_loader)} | rows={len(rows)}")

        if not rows:
            print("[WARN] No rows collected.")
            return

        import pandas as pd
        df = pd.DataFrame(rows)
        df["data"] = self.args.data
        df["model"] = self.args.model
        df["setting"] = setting
        df["seq_len"] = self.args.seq_len
        df["label_len"] = self.args.label_len
        df["pred_len"] = self.args.pred_len
        df["alpha"] = alpha_boundary
        df["welch_win_frac"] = welch_win_frac
        df["welch_overlap"] = welch_overlap

        detail_path = root_out / (
            f"detail_{self.args.data}"
            f"_sl{self.args.seq_len}_ll{self.args.label_len}_pl{self.args.pred_len}"
            f"_alpha{alpha_boundary}_w{welch_win_frac}_ov{welch_overlap}.csv"
        )
        df.to_csv(detail_path, index=False)

        chans = sorted(df["channel"].unique().tolist())
        summary_rows = []
        for c in chans:
            sub = df[df["channel"] == c]
            x = sub["MSE_lb"].to_numpy(dtype=float)
            y = sub["MSE_model"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]; y = y[m]
            if x.size >= 2:
                r = float(np.corrcoef(x, y)[0, 1])
                try:
                    a, b = np.polyfit(x, y, deg=1)
                except Exception:
                    a, b = np.nan, np.nan
                m_x, m_y = float(np.nanmean(x)), float(np.nanmean(y))
            else:
                r, a, b, m_x, m_y = np.nan, np.nan, np.nan, np.nan, np.nan

            mae_mean = float(np.nanmean(sub["MAE_model"].to_numpy(dtype=float))) if "MAE_model" in sub else np.nan
            nmse_mean = float(np.nanmean(sub["NMSE_model"].to_numpy(dtype=float))) if "NMSE_model" in sub else np.nan


            summary_rows.append({
                "channel": int(c),
                "n_samples": int(x.size),
                "pearson_r": r,
                "slope": a,
                "intercept": b,
                "MSE_lb_mean": m_x,
                "MSE_model_mean": m_y,
                "MAE_model_mean": mae_mean,   
                "NMSE_model_mean": nmse_mean,   
            })

        df_sum = pd.DataFrame(summary_rows)
        sum_path = root_out / (
            f"summary_channels_{self.args.data}"
            f"_sl{self.args.seq_len}_ll{self.args.label_len}_pl{self.args.pred_len}"
            f"_alpha{alpha_boundary}_w{welch_win_frac}_ov{welch_overlap}.csv"
        )
        df_sum.to_csv(sum_path, index=False)

        x_all = df["MSE_lb"].to_numpy(dtype=float)
        y_all = df["MSE_model"].to_numpy(dtype=float)
        m_all = np.isfinite(x_all) & np.isfinite(y_all)

        overall_r = float(np.corrcoef(x_all[m_all], y_all[m_all])[0, 1]) if m_all.any() else np.nan
        overall_mse_model_mean = float(np.nanmean(df["MSE_model"].to_numpy(dtype=float)))
        overall_mae_model_mean = float(np.nanmean(df["MAE_model"].to_numpy(dtype=float)))
        overall_nmse_model_mean = float(np.nanmean(df["NMSE_model"].to_numpy(dtype=float)))
        overall_mse_lb_mean    = float(np.nanmean(df["MSE_lb"].to_numpy(dtype=float)))
        overall_nmse_lb_mean   = float(np.nanmean(df["NMSE_lb"].to_numpy(dtype=float)))
        overall_plin_mean      = float(np.nanmean(df["P_lin"].to_numpy(dtype=float)))
        total_rows             = int(np.isfinite(df["MSE_model"]).sum())
        

        with open(root_out / "correlation_overall.txt", "w") as f:
            f.write(f"overall_pearson_r: {overall_r:.6f}\n")
            f.write(f"rows: {total_rows}\n")
            f.write(f"mse_model_mean_overall: {overall_mse_model_mean:.6f}\n")
            f.write(f"mae_model_mean_overall: {overall_mae_model_mean:.6f}\n")
            f.write(f"nmse_model_mean_overall: {overall_nmse_model_mean:.6f}\n")
            f.write(f"mse_lb_mean_overall: {overall_mse_lb_mean:.6f}\n")
            f.write(f"nmse_lb_mean_overall: {overall_nmse_lb_mean:.6f}\n")
            f.write(f"P_lin_mean_overall: {overall_plin_mean:.6f}\n")

        overall_csv_path = root_out / (
            f"dataset_overall_{self.args.data}"
            f"_sl{self.args.seq_len}_ll{self.args.label_len}_pl{self.args.pred_len}"
            f"_alpha{alpha_boundary}_w{welch_win_frac}_ov{welch_overlap}.csv"
        )
        df_overall = pd.DataFrame([{
            "data": self.args.data,
            "data_path": self.args.data_path,
            "model": self.args.model,
            "setting": setting,
            "seq_len": self.args.seq_len,
            "label_len": self.args.label_len,
            "pred_len": self.args.pred_len,
            "alpha": alpha_boundary,
            "welch_win_frac": welch_win_frac,
            "welch_overlap": welch_overlap,
            "rows": total_rows,
            "pearson_r": overall_r,
            "mse_model_mean_overall": overall_mse_model_mean,
            "mae_model_mean_overall": overall_mae_model_mean,  
            "nmse_model_mean_overall": overall_nmse_model_mean, 
            "mse_lb_mean_overall": overall_mse_lb_mean,
            "nmse_lb_mean_overall": overall_nmse_lb_mean,
            "P_lin_mean_overall": overall_plin_mean,
        }])
        df_overall.to_csv(overall_csv_path, index=False)

        print(f"[OK] detail CSV:   {detail_path}")
        print(f"[OK] summary CSV:  {sum_path}")
        print(f"[OK] overall r ->  {overall_r:.6f}")
        print(
            "[OK] dataset means -> "
            f"MSE_model: {overall_mse_model_mean:.6f}, "
            f"MAE_model: {overall_mae_model_mean:.6f}, "
            f"NMSE_model: {overall_nmse_model_mean:.6f}, "
            f"MSE_lb: {overall_mse_lb_mean:.6f}"
        )

    def evaluate_band_predictability_dualaxis(
        self,
        setting: str,
        channel: int = 0,
        bands: str = "auto:10",          
        welch_win_frac: float = 0.25,
        welch_overlap: float = 0.5,
        include_dc: bool = False,      
        max_samples: int | None = 3000,
        outdir: str = "./predictability_bands",
        load: bool = True,
    ):

        os.makedirs(outdir, exist_ok=True)

        test_data, test_loader = self._get_data(flag='test')
        if load:
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            else:
                print(f"[WARN] checkpoint not found: {ckpt_path} (use current weights)")

        # ---------- Welch + coherence ----------
        def _hann(n: int) -> np.ndarray:
            return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)

        def _welch_psd_and_csd(x: np.ndarray, y: np.ndarray, nperseg: int, noverlap: int, eps: float = 1e-12):
            N = len(x)
            nperseg = max(8, min(nperseg, N))
            step = max(1, nperseg - noverlap)
            if step <= 0:
                step = max(1, nperseg // 2)
            starts = np.arange(0, N - nperseg + 1, step, dtype=int)
            if len(starts) == 0:
                starts = np.array([max(0, N - nperseg)], dtype=int)
            win = _hann(nperseg).astype(np.float64)
            W = np.sum(win**2) + eps
            f = np.fft.rfftfreq(nperseg, d=1.0)
            Sxx = np.zeros_like(f, dtype=np.complex128)
            Syy = np.zeros_like(f, dtype=np.complex128)
            Sxy = np.zeros_like(f, dtype=np.complex128)
            for s in starts:
                X = np.fft.rfft(x[s:s+nperseg] * win)
                Y = np.fft.rfft(y[s:s+nperseg] * win)
                Sxx += X * np.conj(X)
                Syy += Y * np.conj(Y)
                Sxy += X * np.conj(Y)
            K = len(starts)
            Sxx = (Sxx / (K * W)).real
            Syy = (Syy / (K * W)).real
            Sxy = (Sxy / (K * W))
            return f, Sxx, Syy, Sxy

        def _coherence(x: np.ndarray, y: np.ndarray, win_frac: float, overlap: float):
            eps = 1e-12
            N = len(x)
            nperseg = max(8, int(win_frac * N))
            noverlap = int(overlap * nperseg)
            f, Sxx, Syy, Sxy = _welch_psd_and_csd(x, y, nperseg, noverlap, eps)
            gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy + eps)
            gamma2 = np.clip(gamma2.real, 0.0, 1.0)
            return f, Sxx, Syy, Sxy, gamma2

        def _parse_bands(bands_str: str, f_nyq: float = 0.5) -> list[tuple[float, float]]:
            if bands_str.startswith("auto:"):
                k = int(bands_str.split(":")[1])
                edges = np.linspace(0.0, f_nyq, k + 1)
                edges[0] = 0.0
                bands = [(edges[i], edges[i+1]) for i in range(k)]
                bands[0] = (1e-9, bands[0][1])  
                return bands
            res: list[tuple[float, float]] = []
            for seg in bands_str.split(","):
                a, b = seg.split("-")
                res.append((float(a), float(b)))
            return res

        band_edges = _parse_bands(bands, f_nyq=0.5)
        band_labels = [f"[{a:.3f},{b:.3f})" for (a, b) in band_edges]

        device = torch.device(f"cuda:{self.args.gpu}") if self.args.use_gpu else torch.device("cpu")
        self.model.eval()
        self.model.to(device)

        f_dim = -1 if self.args.features.upper() == 'MS' else 0

        Efrac_rows: list[list[float]] = []          
        PE_over_tot_rows: list[list[float]] = []   
        P_total_list: list[float] = []             
        mse_list: list[float] = []                 
        idx_list: list[int] = []
        seen = 0

        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # batch_x = batch[0].float().to(device)      # [B, T_x, C]
                # batch_y = batch[1].float().to(device)      # [B, T_y, C] （T_y = label_len + pred_len）



                y_pred = outputs[:, -self.args.pred_len:, f_dim:]  # [B, T, C]
                y_true = batch_y[:, -self.args.pred_len:, f_dim:]  # [B, T, C]
                # batch_x_eval = batch_x[:, :, f_dim:]                     # [B, Tx, C]

                x_np = batch_x.detach().cpu().numpy()            # [B, T_x, C]
                y_np = batch_y.detach().cpu().numpy()            # [B, T_y, C]
                ypred_np = y_pred.detach().cpu().numpy()         # [B, T_pred, C’]
                ytrue_np = y_true.detach().cpu().numpy()         # [B, T_pred, C’]


                B, T_x, Cx = x_np.shape
                _, T_y, Cy = y_np.shape
                pred_len = self.args.pred_len

                ypred_ch = y_pred[:, :, channel]
                ytrue_ch = y_true[:, :, channel]
                # print(torch.mean(ypred_ch))
                # print(torch.std(ypred_ch))
                # print(torch.mean(ytrue_ch))
                # print(torch.std(ytrue_ch))
        

                # print(ypred_ch)
                # print(ytrue_ch)
                # mse_batch = np.mean((ypred_ch - ytrue_ch) ** 2, axis=1)  # [B]
                mse_batch = torch.mean((ypred_ch - ytrue_ch) ** 2, dim=1).detach().cpu().numpy()  # [B]

                for b in range(B):
                    L = min(self.args.seq_len, pred_len, T_x, pred_len)
                    x_tail = x_np[b, T_x - L:T_x, channel].astype(np.float64)
                    y_head = y_np[b, T_y - pred_len:T_y, channel].astype(np.float64)[:L]

                    x0 = x_tail - np.mean(x_tail)
                    y0 = y_head - np.mean(y_head)

                    f, Sxx, Syy, Sxy, gamma2 = _coherence(x0, y0, welch_win_frac, welch_overlap)
                    if not include_dc:
                        m = f > 0
                        f, Syy, gamma2 = f[m], Syy[m], gamma2[m]

                    parts_S = []
                    for (fa, fb) in band_edges:
                        m = (f >= fa) & (f < fb)
                        parts_S.append(float(np.sum(Syy[m])))
                    total_S = float(np.sum(Syy) + 1e-12)
                    E_frac = [p / total_S for p in parts_S]

                    parts_PE = []
                    for (fa, fb) in band_edges:
                        m = (f >= fa) & (f < fb)
                        parts_PE.append(float(np.sum(gamma2[m] * Syy[m])))
                    PE_over_total = [p / total_S for p in parts_PE]
                    P_total = float(np.clip(sum(PE_over_total), 0.0, 1.0))

                    Efrac_rows.append(E_frac)
                    PE_over_tot_rows.append(PE_over_total)
                    P_total_list.append(P_total)
                    mse_list.append(float(mse_batch[b]))
                    idx_list.append(seen)
                    seen += 1

                    if max_samples is not None and seen >= max_samples:
                        break
                if max_samples is not None and seen >= max_samples:
                    break

        if seen == 0:
            raise RuntimeError("No Samples")

        xs = np.asarray(idx_list, dtype=int)
        Efrac_arr = np.asarray(Efrac_rows, dtype=np.float64)             # [N, Bn], sum=1
        PE_over_tot_arr = np.asarray(PE_over_tot_rows, dtype=np.float64) # [N, Bn], sum=P_total
        Ptot_arr = np.asarray(P_total_list, dtype=np.float64)            # [N]
        MSE_arr = np.asarray(mse_list, dtype=np.float64)                 # [N]

        def _save_csv(mat: np.ndarray, name: str):
            path = os.path.join(outdir, f"{name}.csv")
            header = ["sample_idx"] + [*band_labels]
            tbl = np.column_stack([xs, mat])
            np.savetxt(path, tbl, fmt="%.8f", delimiter=",", header=",".join(header), comments="")
            print(f"[OK] CSV -> {path}")

        _save_csv(Efrac_arr,        f"energy_frac_ch{channel}")
        _save_csv(PE_over_tot_arr,  f"pred_energy_over_total_frac_ch{channel}")
        path_pm = os.path.join(outdir, f"P_total_and_MSE_ch{channel}.csv")
        np.savetxt(path_pm, np.column_stack([xs, Ptot_arr, MSE_arr]), fmt="%.8f",
                delimiter=",", header="sample_idx,P_total,MSE", comments="")
        print(f"[OK] CSV -> {path_pm}")

        fig, axL = plt.subplots(figsize=(11, 5.5))
        axL.stackplot(xs, *[PE_over_tot_arr[:, j] for j in range(PE_over_tot_arr.shape[1])],
                    labels=band_labels, alpha=0.90)
        axL.plot(xs, Ptot_arr, linewidth=1.6, label="P_total", linestyle="-")
        axL.set_ylim(0.0, 1.02)
        axL.set_xlabel("Test sample index (in order)")
        axL.set_ylabel("Predictable energy share over TOTAL (stacked)")
        axL.set_title(f"Predictable energy over TOTAL + MSE (channel={channel})")

        axR = axL.twinx()
        axR.plot(xs, MSE_arr, linewidth=1.2, linestyle="-", label="MSE (model)", color="red")
        axR.set_ylabel("MSE (model)")

        linesL, labelsL = axL.get_legend_handles_labels()
        linesR, labelsR = axR.get_legend_handles_labels()
        axL.legend(linesL + linesR, labelsL + labelsR, loc="upper right", ncol=1, fontsize=9)
        axL.grid(True, ls="--", alpha=0.35)
        fig.tight_layout()
        png = os.path.join(outdir, f"predictable_stack_ch{channel}.png")
        fig.savefig(png, dpi=180)
        plt.close(fig)
        print(f"[OK] PNG -> {png}")

        return {
            "csv_energy_frac": os.path.join(outdir, f"energy_frac_ch{channel}.csv"),
            "csv_pred_over_total": os.path.join(outdir, f"pred_energy_over_total_frac_ch{channel}.csv"),
            "csv_ptotal_mse": os.path.join(outdir, f"P_total_and_MSE_ch{channel}.csv"),
            "png_dualaxis": png,
        }



    def evaluate_band_LUR(
        self,
        setting: str,
        channel: int = 0,
        bands: str = "auto:10",          
        welch_win_frac: float = 0.25,
        welch_overlap: float = 0.5,
        include_dc: bool = False,       
        alpha_boundary: float = 1.0,     
        max_samples: int | None = 3000,
        outdir: str = "./sce_bands",
        load: bool = True,
        ckpt_file: str | None = None,
    ):

        import os
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt

        dataname = str(self.args.data_path)[:-4]
        subdir = (
            f"{dataname}/"
            f"sl{self.args.seq_len}_ll{self.args.label_len}_pl{self.args.pred_len}/"
            f"w{welch_win_frac}_ov{welch_overlap}/"
            f"{self.args.model}"
        )
        root_out = Path(outdir) / subdir / setting
        root_out.mkdir(parents=True, exist_ok=True)

        test_data, test_loader = self._get_data(flag='test')

        if load:
            ckpt_path = ckpt_file or os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=self.device)
                if isinstance(state, dict):
                    sd = state.get('model') or state.get('state_dict') or state
                else:
                    sd = state
                from collections import OrderedDict
                new_sd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in sd.items())
                missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
                if missing or unexpected:
                    print(f"[WARN] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
                print(f"[OK] loaded checkpoint: {ckpt_path}")
            else:
                print(f"[WARN] checkpoint not found: {ckpt_path} (use current weights)")

        self.model.eval()
        self.model.to(self.device)

        # ---------- Welch + tools ----------
        def _hann(n: int) -> np.ndarray:
            return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)

        def _welch_psd_and_cpsd(x: np.ndarray, y: np.ndarray, nperseg: int, noverlap: int, eps: float = 1e-12):
            N = len(x)
            nperseg = max(8, min(nperseg, N))
            step = max(1, nperseg - noverlap)
            if step <= 0:
                step = max(1, nperseg // 2)
            starts = np.arange(0, N - nperseg + 1, step, dtype=int)
            if len(starts) == 0:
                starts = np.array([max(0, N - nperseg)], dtype=int)
            win = _hann(nperseg).astype(np.float64)
            W = np.sum(win**2) + eps
            f = np.fft.rfftfreq(nperseg, d=1.0)
            Sxx = np.zeros_like(f, dtype=np.complex128)
            Syy = np.zeros_like(f, dtype=np.complex128)
            Sxy = np.zeros_like(f, dtype=np.complex128)
            for s in starts:
                X = np.fft.rfft(x[s:s+nperseg] * win)
                Y = np.fft.rfft(y[s:s+nperseg] * win)
                Sxx += X * np.conj(X)
                Syy += Y * np.conj(Y)
                Sxy += X * np.conj(Y)
            K = len(starts)
            Sxx = (Sxx / (K * W)).real
            Syy = (Syy / (K * W)).real
            Sxy = (Sxy / (K * W))
            return f, Sxx, Syy, Sxy

        def _coherence(x: np.ndarray, y: np.ndarray, win_frac: float, overlap: float):
            eps = 1e-12
            N = len(x)
            nperseg = max(8, int(win_frac * N))
            noverlap = int(overlap * nperseg)
            f, Sxx, Syy, Sxy = _welch_psd_and_cpsd(x, y, nperseg, noverlap, eps)
            gamma2 = (np.abs(Sxy) ** 2) / (Sxx * Syy + eps)
            gamma2 = np.clip(gamma2.real, 0.0, 1.0)
            return f, Sxx, Syy, Sxy, gamma2

        def _parse_bands(bands_str: str, f_nyq: float = 0.5) -> list[tuple[float, float]]:
            if bands_str.startswith("auto:"):
                k = int(bands_str.split(":")[1])
                edges = np.linspace(0.0, f_nyq, k + 1)
                edges[0] = 0.0
                res = [(edges[i], edges[i+1]) for i in range(k)]
                res[0] = (1e-9, res[0][1])  
                return res
            out = []
            for seg in bands_str.split(","):
                a, b = seg.split("-")
                out.append((float(a), float(b)))
            return out

        band_edges = _parse_bands(bands, f_nyq=0.5)
        band_labels = [f"[{a:.3f},{b:.3f})" for (a, b) in band_edges]

        f_dim = -1 if self.args.features.upper() == 'MS' else 0

        rows = []
        sample_id = 0

        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]      # [B, T, C’]
                y_pred  = outputs[:, :, f_dim:]                    # [B, T, C’]
                y_true  = batch_y[:, -self.args.pred_len:, f_dim:] # [B, T, C’]
                x_hist  = batch_x[:, :, f_dim:]                    # [B, Tx, C’]

                if y_pred.shape[-1] == 1 and channel != 0:
                    raise ValueError(f"features=MS  channel={channel}")

                # numpy
                x_np = x_hist.detach().cpu().numpy()
                y_np = y_true.detach().cpu().numpy()
                h_np = y_pred.detach().cpu().numpy()

                B, Tx, _ = x_np.shape
                _, T, _  = y_np.shape
                Np = max(8, int(alpha_boundary * min(Tx, T)))

                for b in range(B):
                    x_tail = x_np[b, -Np:, channel].astype(np.float64)
                    y_head = y_np[b, :Np,   channel].astype(np.float64)
                    h_head = h_np[b, :Np,   channel].astype(np.float64)

                    x0 = x_tail - np.mean(x_tail)
                    y0 = y_head - np.mean(y_head)
                    h0 = h_head - np.mean(h_head)

                    # γ^2_{x,y} & Syy_base
                    f_xy, Sxx_xy, Syy_xy, Sxy_xy, gamma2_xy = _coherence(x0, y0, welch_win_frac, welch_overlap)
                    # γ^2_{y,ŷ}
                    f_yh, Syy_yh, Shh_yh, Syh_yh = _welch_psd_and_cpsd(y0, h0, int(welch_win_frac*Np), int(welch_overlap*int(welch_win_frac*Np)))
                    gamma2_yh = (np.abs(Syh_yh) ** 2) / (Syy_yh * Shh_yh + 1e-12)
                    gamma2_yh = np.clip(gamma2_yh.real, 0.0, 1.0)

                    Lmin = min(len(f_xy), len(f_yh))
                    f = f_xy[:Lmin]
                    Syy = Syy_xy[:Lmin]           
                    g_xy = gamma2_xy[:Lmin]
                    g_yh = gamma2_yh[:Lmin]

                    if not include_dc:
                        m = f > 0
                        f, Syy, g_xy, g_yh = f[m], Syy[m], g_xy[m], g_yh[m]

                    xr_vec, w_vec = [], []
                    for (fa, fb) in band_edges:
                        sel = (f >= fa) & (f < fb)
                        num = float(np.sum(Syy[sel] * g_yh[sel]))  
                        den = float(np.sum(Syy[sel] * g_xy[sel])) 
                        xr  = np.nan if den == 0.0 else (num / den)
                        xr_vec.append(xr)
                        w_vec.append(float(np.sum(Syy[sel])))

                    w_sum = float(np.sum(w_vec))
                    w_norm = [w / w_sum if w_sum > 0 else np.nan for w in w_vec]

                    rows.append({
                        "sample_id": sample_id,
                        "channel": int(channel),
                        "xr": xr_vec,
                        "weights": w_norm,
                    })
                    sample_id += 1

                    if max_samples is not None and sample_id >= max_samples:
                        break
                if max_samples is not None and sample_id >= max_samples:
                    break

        if not rows:
            print("[WARN] No rows collected.")
            return

        recs = []
        for r in rows:
            sid = r["sample_id"]; ch = r["channel"]
            for j, lab in enumerate(band_labels):
                recs.append({
                    "sample_id": sid, "channel": ch, "band": lab,
                    "XR_m": r["xr"][j], "weight": r["weights"][j],
                })
        df = pd.DataFrame(recs)
        detail_path = root_out / f"sce_bands_XR_detail_ch{channel}.csv"
        df.to_csv(detail_path, index=False)

        def _q(x, q): 
            v = x[np.isfinite(x)]
            return np.nan if v.size == 0 else np.quantile(v, q)
        sum_rows = []
        for j, lab in enumerate(band_labels):
            sub = df[df["band"] == lab]
            xr_vals = sub["XR_m"].to_numpy(dtype=float)
            w_vals  = sub["weight"].to_numpy(dtype=float)
            sum_rows.append({
                "band": lab,
                "XR_mean": float(np.nanmean(xr_vals)),
                "XR_std":  float(np.nanstd(xr_vals)),
                "XR_q25":  float(_q(xr_vals, 0.25)),
                "XR_med":  float(_q(xr_vals, 0.50)),
                "XR_q75":  float(_q(xr_vals, 0.75)),
                "weight_mean": float(np.nanmean(w_vals)),
            })
        df_sum = pd.DataFrame(sum_rows)
        sum_path = root_out / f"sce_bands_XR_summary_ch{channel}.csv"
        df_sum.to_csv(sum_path, index=False)

        x = np.arange(len(band_labels))
        xr_mean = df_sum["XR_mean"].to_numpy(dtype=float)
        w_mean  = df_sum["weight_mean"].to_numpy(dtype=float)

        fig, axL = plt.subplots(figsize=(11.5, 5.5))
        axL.bar(x, xr_mean, width=0.6, alpha=0.88, label="XR (mean per band)")
        axL.set_ylabel("XR (explained-power ratio)")
        axL.set_xlabel("Bands")
        axL.set_xticks(x)
        axL.set_xticklabels(band_labels, rotation=30, ha="right")
        axL.axhline(1.0, ls="--", lw=1.0, color="gray", alpha=0.7)  
        axL.set_ylim(bottom=0.0)  

        axR = axL.twinx()
        axR.plot(x, w_mean, lw=1.8, ls="-", marker="o", label="energy weight (mean)")
        axR.set_ylabel("Energy weight E_m / ΣE")

        hL, lL = axL.get_legend_handles_labels()
        hR, lR = axR.get_legend_handles_labels()
        axL.legend(hL + hR, lL + lR, loc="upper right")
        axL.grid(True, ls="--", alpha=0.35)
        fig.tight_layout()
        png = root_out / f"sce_bands_XR_weight_ch{channel}.png"
        fig.savefig(png, dpi=180); plt.close(fig)

        print(f"[OK] detail CSV:  {detail_path}")
        print(f"[OK] summary CSV: {sum_path}")
        print(f"[OK] FIG:         {png}")

        return {"detail_csv": str(detail_path), "summary_csv": str(sum_path), "png": str(png)}
