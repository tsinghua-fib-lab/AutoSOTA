
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch)
from utils.tools import (EarlyStopping, adjust_learning_rate, ensure_path,
                         visual)

warnings.filterwarnings('ignore')

from thop import profile, clever_format


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.pred_len = args.pred_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

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

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        ensure_path(path)
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
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
                        # outputs shape: [B, P, D]
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = 0

                    # if self.args.rec_lambda:
                    #     loss_rec = criterion(outputs, batch_y)
                    #
                    #     loss += self.args.rec_lambda * loss_rec
                    #     if (i + 1) % 100 == 0:
                    #         print(f"\tloss_rec: {loss_rec.item()}")
                    #
                    #     self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss_rec, self.step)
                    # # 原时域损失
                    # loss_tmp = ((outputs - batch_y) ** 2).mean()
                    # # 所提频域损失
                    # loss_feq = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean()
                    if self.args.auxi_lambda:
                        # fft shape: [B, P, D]
                        if self.args.auxi_mode == "fft":
                            loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)
                        elif self.args.auxi_mode == "rfft":
                            if self.args.auxi_type == 'complex':
                                loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
                            elif self.args.auxi_type == 'complex-phase':
                                loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                            elif self.args.auxi_type == 'complex-mag-phase':
                                loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                                loss_auxi_phase = (
                                            torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                                loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                            elif self.args.auxi_type == 'phase':
                                loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y,
                                                                                                    dim=1).angle()
                            elif self.args.auxi_type == 'mag':
                                loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                            elif self.args.auxi_type == 'mag-phase':
                                loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y,
                                                                                                      dim=1).abs()
                                loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y,
                                                                                                          dim=1).angle()
                                loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                            else:
                                raise NotImplementedError
                        elif self.args.auxi_mode == "rfft-D":
                            loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)

                        elif self.args.auxi_mode == "rfft-2D":
                            loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)

                        elif self.args.auxi_mode == "legendre":
                            loss_auxi = leg_torch(outputs, self.args.leg_degree, device=self.device) - leg_torch(
                                batch_y, self.args.leg_degree, device=self.device)

                        elif self.args.auxi_mode == "chebyshev":
                            loss_auxi = chebyshev_torch(outputs, self.args.leg_degree,
                                                        device=self.device) - chebyshev_torch(batch_y,
                                                                                              self.args.leg_degree,
                                                                                              device=self.device)
                        elif self.args.auxi_mode == "hermite":
                            loss_auxi = hermite_torch(outputs, self.args.leg_degree,
                                                      device=self.device) - hermite_torch(batch_y, self.args.leg_degree,
                                                                                          device=self.device)

                        elif self.args.auxi_mode == "laguerre":
                            loss_auxi = laguerre_torch(outputs, self.args.leg_degree,
                                                       device=self.device) - laguerre_torch(batch_y,
                                                                                            self.args.leg_degree,
                                                                                            device=self.device)
                        else:
                            raise NotImplementedError

                        # if self.mask is not None:
                        #     loss_auxi *= self.mask

                        if self.args.auxi_loss == "MAE":
                            # MAE, 最小化element-wise error的模长
                            loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()
                            # check the dim of fft
                        elif self.args.auxi_loss == "MSE":
                            # MSE, 最小化element-wise error的模长
                            loss_auxi = (loss_auxi.abs() ** 2).mean() if self.args.module_first else (
                                        loss_auxi ** 2).mean().abs()
                        else:
                            raise NotImplementedError

                        loss += self.args.auxi_lambda * loss_auxi
                        if (i + 1) % 100 == 0:
                            print(f"\tloss_auxi: {loss_auxi.item()}")

                        self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', loss_auxi, self.step)

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
            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/test/loss', test_loss, self.epoch)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        ensure_path(folder_path)

        self.model.eval()
        metric_collector = create_metric_collector(device=self.device)
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
                outputs = outputs.detach()
                batch_y = batch_y.detach()
                if test_data.scale and self.args.inverse:
                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()

                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = torch.from_numpy(outputs).to(self.device)
                    batch_y = torch.from_numpy(batch_y).to(self.device)

                outputs = outputs[:, :, f_dim:].contiguous()
                batch_y = batch_y[:, :, f_dim:].contiguous()

                metric_collector.update(outputs, batch_y)

                if self.output_pred or self.output_vis:
                    inp = batch_x.cpu().numpy()
                    pred = outputs.cpu().numpy()
                    true = batch_y.cpu().numpy()

                    inputs.append(inp)
                    preds.append(pred)
                    trues.append(true)

                if i % 20 == 0 and self.output_vis:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.output_pred:
            inputs = np.array(inputs)
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        m = metric_collector.compute()
        mae, mse, rmse, mape, mspe = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"]
        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.close()

        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))
        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs)
            np.save(os.path.join(res_path, 'pred.npy'), preds)
            np.save(os.path.join(res_path, 'true.npy'), trues)

        return

# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)
#
#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()
#
#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model
#
#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader
#
#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
#         return model_optim
#
#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion
#
#     def vali(self, vali_data, vali_loader, criterion):
#         total_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()
#
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#
#                 pred = outputs.detach().cpu()
#                 true = batch_y.detach().cpu()
#
#                 loss = criterion(pred, true)
#
#                 total_loss.append(loss)
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss
#
#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')
#
#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         time_now = time.time()
#
#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
#
#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()
#
#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()
#
#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []
#
#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                     loss = criterion(outputs, batch_y)
#                     train_loss.append(loss.item())
#
#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()
#
#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()
#
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(vali_data, vali_loader, criterion)
#             test_loss = self.vali(test_data, test_loader, criterion)
#
#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#                 epoch + 1, train_steps, train_loss, vali_loss, test_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#             adjust_learning_rate(model_optim, epoch + 1, self.args)
#
#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))
#
#         return self.model
#
#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
#
#         preds = []
#         trues = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)
#
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, :]
#                 batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()
#                 if test_data.scale and self.args.inverse:
#                     shape = outputs.shape
#                     outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                     batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
#
#                 outputs = outputs[:, :, f_dim:]
#                 batch_y = batch_y[:, :, f_dim:]
#
#                 pred = outputs
#                 true = batch_y
#
#                 preds.append(pred)
#                 trues.append(true)
#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     if test_data.scale and self.args.inverse:
#                         shape = input.shape
#                         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#
#         preds = np.concatenate(preds, axis=0)
#         trues = np.concatenate(trues, axis=0)
#         print('test shape:', preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print('test shape:', preds.shape, trues.shape)
#
#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#
#         # dtw calculation
#         if self.args.use_dtw:
#             dtw_list = []
#             manhattan_distance = lambda x, y: np.abs(x - y)
#             for i in range(preds.shape[0]):
#                 x = preds[i].reshape(-1,1)
#                 y = trues[i].reshape(-1,1)
#                 if i % 100 == 0:
#                     print("calculating dtw iter:", i)
#                 d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
#                 dtw_list.append(d)
#             dtw = np.array(dtw_list).mean()
#         else:
#             dtw = 'not calculated'
#
#
#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
#         f = open("result_long_term_forecast.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
#         f.write('\n')
#         f.write('\n')
#         f.close()
#
#         np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + 'pred.npy', preds)
#         np.save(folder_path + 'true.npy', trues)
#
#         return
