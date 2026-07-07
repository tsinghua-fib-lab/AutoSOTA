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
# import matplotlib.pyplot as plt
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
        L = self.args.level_dim
        self.d_inputs = [T, L, V]
        self.d_folds = [V * L, self.args.d_compress[0] * V, self.args.d_compress[0] * self.args.d_compress[1]]
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
            model = model_dict[self.args.model].Model(self.args, d_inputs=self.d_inputs, d_folds=self.d_folds,
                                                      d_compress=self.args.d_compress).float()
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
        criterion = nn.L1Loss()
        return criterion

    def _update_model_ema(self, model_pre, model, decay=0.998):
        # The y_encoder is updated using EMA to slowly track the progress of x_encoder
        return ema_update(model_pre, model, decay)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # evaluate with the smoothed (EMA) model only
        self.model_pre.eval()
        self.model.eval()
        L = self.args.level_dim
        iter_count = 0
        time_now = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if L > 0:
                    batch_x = batch_x.float().to(self.device)  # (B, T, V)
                    batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    raiseExceptions ('amp not supported')
                else:
                    if 'test' in self.args.model or 'ReNF' in self.args.model:
                        outputs = self.model_pre(batch_x)
                    else:
                        raiseExceptions (f'{self.args.model} not supported')
                
                if (i + 1) % 500 == 0:
                    speed = (time.time() - time_now) / iter_count
                    print('\tinference speed: {:.4f}s/iter'.format(speed))
                    iter_count = 0
                    time_now = time.time()

                f_dim = -1 if self.args.features == 'MS' else 0

                if L > 0:
                    # pred = outputs[-1][:, -self.args.pred_len:, f_dim:].detach().cpu()
                    pred = outputs[:, -self.args.pred_len:, f_dim:].detach().cpu()
                    true = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu()
                else:
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

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

                    loss = self.time_freq_loss(outputs, batch_y, alpha_freq=0.0)
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
        import matplotlib.pyplot as plt
        np.save('./toy_exp/valid_loss_' + self.args.data + '_' + str(self.args.r_ema) + '.npy', valid_losses)
        np.save('./toy_exp/test_loss_' + self.args.data + '_' + str(self.args.r_ema) + '.npy', test_losses)
        plt.plot(valid_losses)
        plt.savefig('./toy_exp/valid_loss_' + self.args.data + '_' + str(self.args.r_ema) + '.pdf')
        plt.clf()
        
        plt.plot(test_losses)
        plt.savefig('./toy_exp/test_loss_' + self.args.data + '_' + str(self.args.r_ema) + '.pdf')
        # load the best EMA weights into the online model for downstream use
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        L = self.args.level_dim
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
                        outputs = self.model(batch_x)
                    else:
                        raise Exception('Model not supported')

                f_dim = -1 if self.args.features == 'MS' else 0
                
                pred = outputs[:, :, f_dim:].detach().cpu().numpy()
                # rebuild the final result
                # t1 = torch.zeros_like(batch_y)
                # t2 = torch.zeros_like(batch_y)
                # t3 = torch.zeros_like(batch_y)
                # y1, y2, y3 = outputs
                # t1[:, :y1.shape[1], :] = y1
                # t2[:, :y2.shape[1], :] = y2
                # t3[:, :y3.shape[1], :] = y3
                # average the three
                # pred[:, :y1.shape[1], :] = (t1[:, :y1.shape[1], :] + t2[:, :y1.shape[1], :] + t3[:, :y1.shape[1], :]) / 3
                # pred[:, y1.shape[1]:y2.shape[1], :] = (t2[:, y1.shape[1]:y2.shape[1], :] + t3[:, y1.shape[1]:y2.shape[1], :]) / 2
                # pred[:, y2.shape[1]:, :] = t3[:, y2.shape[1]:, :]
                # optimal combination
                # z1 = (batch_y - t1).abs()
                # z2 = (batch_y - t2).abs()
                # z3 = (batch_y - t3).abs()
                # pred_tmp = torch.where(z1 < z2, t1, t2)
                # z = (batch_y - pred_tmp).abs()
                # pred_com = torch.where(z < z3, pred_tmp, t3).detach().cpu().numpy()
                
                true = batch_y[:, :, f_dim:].detach().cpu().numpy()

                if i == 10:
                    import matplotlib.pyplot as plt
                    plt.plot(pred[0, :, 0], label='out_all')
                    # plt.plot(outputs[0][0, :, 0].detach().cpu().numpy(), label='out_0')
                    # plt.plot(outputs[1][0, :, 0].detach().cpu().numpy(), label='out_1')
                    plt.plot(true[0, :, 0], label='true')
                    # plt.plot(pred_com[0, :, 0], label='out_com')
                    plt.legend()
                    plt.savefig('result.png')
                    plt.close()
                # if i == 1:
                #     store = outputs[-1]
                #     Ys = outputs[-2]
                preds += list(pred.flatten())
                trues += list(true.flatten())

                # x_sta, x_mean, x_std = mean_filter(batch_x)
                # y_sta, y_mean, y_std = mean_filter(batch_y[:, -self.args.pred_len:, f_dim:], 5)
                # pre_sta, pre_mean, pre_std = mean_filter(outputs[0][:, -self.args.pred_len:, f_dim:], 5)

                # x_sta = x_sta.detach().cpu().numpy()
                # x_mean = x_mean.detach().cpu().numpy()
                # x_std = x_std.detach().cpu().numpy()

                # y_sta = y_sta.detach().cpu().numpy()
                # y_mean = y_mean.detach().cpu().numpy()
                # y_std = y_std.detach().cpu().numpy()

                # pre_sta = pre_sta.detach().cpu().numpy()
                # pre_mean = pre_mean.detach().cpu().numpy()
                # pre_std = pre_std.detach().cpu().numpy()

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                #     # sta
                #     gt_sta = np.concatenate((x_sta[0, :, -1], y_sta[0, :, -1]), axis=0)
                #     pd_sta = np.concatenate((x_sta[0, :, -1], pre_sta[0, :, -1]), axis=0)
                #     visual(gt_sta, pd_sta, os.path.join(folder_path, str(i) + 'sta.pdf'))

                #     # mean
                #     gt_mean = np.concatenate((x_mean[0, :, -1], y_mean[0, :, -1]), axis=0)
                #     pd_mean = np.concatenate((x_mean[0, :, -1], pre_mean[0, :, -1]), axis=0)
                #     visual(gt_mean, pd_mean, os.path.join(folder_path, str(i) + 'mean.pdf'))

                #     # std
                #     gt_std = np.concatenate((x_std[0, :, -1], y_std[0, :, -1]), axis=0)
                #     pd_std = np.concatenate((x_std[0, :, -1], pre_std[0, :, -1]), axis=0)
                #     visual(gt_std, pd_std, os.path.join(folder_path, str(i) + 'std.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        print(trues.shape)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
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

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
