from logging import raiseExceptions
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ReNF_alpha, ReNF_beta, Model_test
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, ema_update, mean_filter
from utils.metrics import metric
from thop import profile

import torch
import torch.nn as nn
from torch import optim
# from torch.optim import lr_scheduler 
import pytorch_warmup

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import copy

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        _, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        B, T, V = next(iter(self.train_loader))[0].shape
        L = 1
        self.d_inputs = [T, L, V]
        self.model = self._build_model().to(self.device)

        input = torch.zeros(1, T, V).to(self.device)
        flops, params = profile(self.model, inputs=(input, ))
        print('Flops: % .4fG'%(flops / 1e9))
        print('params: % .4fM'% (params / 1e6)) 

    def _build_model(self):
        model_dict = {
            'Model_test': Model_test,
            'ReNF_alpha': ReNF_alpha,
            'ReNF_beta': ReNF_beta,
        }
        if 'ReNF' or 'test' in self.args.model:
            model = model_dict[self.args.model].Model(self.args, d_inputs=self.d_inputs).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, pred_len=None)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _update_model_ema(self, model_pre, model, decay=0.998):
        return ema_update(model_pre, model, decay)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # evaluate with the smoothed (EMA) model only
        self.model_pre.eval()
        self.model.eval()
        iter_count = 0
        time_now = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    raiseExceptions ('No amp Implements')
                else:
                    if 'test' in self.args.model or 'ReNF' in self.args.model:
                        outputs = self.model_pre(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if (i + 1) % 500 == 0:
                    speed = (time.time() - time_now) / iter_count
                    print('\tinference speed: {:.4f}s/iter'.format(speed))
                    iter_count = 0
                    time_now = time.time()

                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs[-1][:, -self.args.pred_len:, f_dim:].detach().cpu()
                true = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        # plot loss
    
        total_loss = np.mean(total_loss)
        # self.model.train()
        return total_loss
    
    def time_freq_loss(self, y_hat, y, alpha_freq):
        freq_loss = (torch.fft.rfft(y_hat, dim=1) - torch.fft.rfft(y, dim=1)).abs().mean()
        # weights = (20. / (torch.arange(y_hat.shape[1], dtype=torch.float32, device=y_hat.device) + 1.0)).unsqueeze(0).unsqueeze(-1)
        time_loss = (y_hat - y).abs().mean()
        return alpha_freq * freq_loss + (1 - alpha_freq) * time_loss
    
    # def time_freq_loss2(self, y_hat, y, alpha_freq):
    #     freq_loss = ((torch.fft.rfft(y_hat, dim=1) - torch.fft.rfft(y, dim=1)).abs()**2).mean()
    #     # weights = (20. / (torch.arange(y_hat.shape[1], dtype=torch.float32, device=y_hat.device) + 1.0)).unsqueeze(0).unsqueeze(-1)
    #     time_loss = ((y_hat - y)**2).mean()
    #     return alpha_freq * freq_loss + (1 - alpha_freq) * time_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_val = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=2, gamma=self.args.gamma)
        self.model.train()
        self.model_pre = copy.deepcopy(self.model).requires_grad_(False)
        valid_losses = []
        test_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
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
                    pass
                else:
                    if 'test' in self.args.model or 'ReNF' in self.args.model:
                            outputs = self.model(batch_x)
                            n = len(outputs)
                            k = self.args.pred_len // n
                    else:
                        pass
                    f_dim = -1 if self.args.features == 'MS' else 0

                    # HRL coefficients
                    # lamb = 0.8 # set to 0 to disable
                    # coefs_L = torch.ones_like(batch_y).cuda()
                    # n_c = 4 # better be a factor of pred_len
                    # for j in range(n_c):
                    #     m = self.args.pred_len // n_c
                    #     end = (j + 1) * m
                    #     coefs_L[:, j:end, f_dim:] *= (0.5 + lamb * (n_c - 1 - j))

                    loss = 0.
                    gamma = 20.
                    for j in range(n):
                        loss += gamma / (j + 1) * self.time_freq_loss(outputs[j], batch_y[:, :(j + 1) * k, f_dim:], self.args.alpha_freq)
                    # loss = self.time_freq_loss(outputs[-1], batch_y[:, :, f_dim:], self.args.alpha_freq)
                    train_loss.append(loss.item())
  
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    pass
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    model_optim.step()
                    with torch.no_grad():
                        self.model_pre = self._update_model_ema(self.model_pre, self.model, decay=self.args.r_ema)
                # store the weight of each layer
                # if (epoch == self.args.train_epochs - 1) and (i == train_steps - 5):
                #     weights = []
                #     import matplotlib.pyplot as plt
                #     for name, param in self.model.named_parameters():
                #         if 'weight' in name:
                #             weights.append(torch.norm(param.data.cpu(), p=2))
                #     plt.plot(weights)
                #     plt.savefig('weights.png')
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali(self.vali_data, self.vali_loader, criterion_val)  # use the same indicator as in training phase
            valid_losses.append(vali_loss)
            test_loss = self.vali(self.test_data, self.test_loader, criterion_val)   # use test indicator
            test_losses.append(test_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model_pre, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step()
            print('Updating learning rate to {:.6f}'.format(scheduler.get_last_lr()[0]))
        # import matplotlib.pyplot as plt
        # np.save('./toy_exp/valid_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.npy', valid_losses)
        # np.save('./toy_exp/test_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.npy', test_losses)
        # plt.plot()
        # plt.savefig('./toy_exp/valid_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.pdf')
        # plt.clf()
        # plt.plot()
        # plt.savefig('./toy_exp/test_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.pdf')
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def post_process(self, outputs, batch_y, iter, strategy='optimal', plot=True):
        # Keep tensors on GPU for computation, store original device
        device = batch_y.device
        
        n = len(outputs)
        k = self.args.pred_len // n
    
        if strategy == 'average':
            pred_com = torch.zeros_like(batch_y)
            weight_sum = torch.zeros_like(batch_y)
            
            for i, output in enumerate(outputs):
                # Each output has length (i+1) * k where k = pred_len // n
                output_len = output.shape[1]
                target_len = min(output_len, batch_y.shape[1])
                
                # Add this output's contribution to the overlapping region
                pred_com[:, :target_len, :] += output[:, :target_len, :]
                weight_sum[:, :target_len, :] += 1.0
            
            # Normalize by the number of outputs that contributed to each position
            pred_com = torch.where(weight_sum > 0, pred_com / weight_sum, pred_com)
        else:
            # Optimal combination
            if n == 0:
                pred_com = torch.zeros_like(batch_y)
            else:
                pred_com = torch.zeros_like(batch_y)
                seq_len = min(outputs[0].shape[1], batch_y.shape[1])
                pred_com[:, :seq_len, :] = outputs[0][:, :seq_len, :]
                
                # Compare with subsequent outputs one by one
                for i in range(1, n):
                    output = outputs[i]
                    seq_len = min(output.shape[1], batch_y.shape[1])
                    
                    # Calculate errors only for the relevant sequence length
                    pred_error = torch.abs(batch_y[:, :seq_len, :] - pred_com[:, :seq_len, :])
                    output_error = torch.abs(batch_y[:, :seq_len, :] - output[:, :seq_len, :])
                    
                    # Update where output is better
                    mask = output_error < pred_error
                    pred_com[:, :seq_len, :] = torch.where(mask, output[:, :seq_len, :], pred_com[:, :seq_len, :])
        
        # Only create plot when actually needed - move to CPU only for plotting
        if plot and iter == 10:
            # Move to CPU only for plotting
            batch_y_cpu = batch_y.detach().cpu()
            pred_com_cpu = pred_com.detach().cpu()
            outputs_cpu = [output.detach().cpu() for output in outputs]
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot forecasts
            for i, output_cpu in enumerate(outputs_cpu):
                ax.plot(output_cpu[0, :, 0].numpy(), label=f'forecast_{i}')
            
            # Plot ground truth and combined prediction
            ax.plot(batch_y_cpu[0, :, 0].numpy(), label='true observation')
            ax.plot(pred_com_cpu[0, :, 0].numpy(), label=strategy)
            
            # Optimized styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True,
                     framealpha=0.9, loc='upper left')
            
            # Set spine properties in batch
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
            
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            plt.tight_layout()
            plt.savefig(f'./forecast_visual/result_{self.args.model_id}_{iter}_{strategy}.pdf')
            plt.close()
            
        return pred_com

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # # Pre-calculate total samples for efficient array allocation
        # total_samples = len(self.test_loader.dataset)
        # pred_len = self.args.pred_len
        # n_features = self.test_loader.dataset[0][1].shape[-1] if hasattr(self.test_loader.dataset, '__getitem__') else 7
        
        # Pre-allocate arrays for better performance
        all_preds = []
        all_trues = []
        all_pred_coms = []
        # all_reps = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                    
                batch_x = batch_x.float().to(self.device)  # (B, T, V, L)
                batch_y = batch_y.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        raise Exception('amp not supported')
                else:
                    if 'test' in self.args.model or 'ReNF' in self.args.model:
                        outputs, reps = self.model(batch_x, return_rep=True)
                    else:
                        raise Exception('Model not supported')

                f_dim = -1 if self.args.features == 'MS' else 0
                
                pred = outputs[-1][:, :, f_dim:].detach().cpu().numpy()
                true = batch_y[:, :, f_dim:].detach().cpu().numpy()
                pred_com = self.post_process(outputs, batch_y, i).detach().cpu().numpy()
                # Collect batch results
                all_preds.append(pred)
                all_trues.append(true)
                all_pred_coms.append(pred_com)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        rep_store = {}
        for n in range(len(reps)):
            rep_store['rep'+str(n)] = reps[n].detach().cpu().numpy()
        np.save('./rep_store.npy', rep_store)
        # Efficient concatenation and flattening
        preds = np.concatenate(all_preds, axis=0).flatten()
        trues = np.concatenate(all_trues, axis=0).flatten()
        pred_coms = np.concatenate(all_pred_coms, axis=0).flatten()
        
        print(f"Results shape: {trues.shape}")
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse = metric(preds, trues)
        mae_com, mse_com = metric(pred_coms, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('mse_com:{}, mae_com:{}'.format(mse_com, mae_com))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # The following savings are for model analyses
        # torch.save(store, folder_path + 'store.pt')
        # torch.save(Ys, folder_path + 'Ys.pt')
        return
