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
import math
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_PIR(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_PIR, self).__init__(args)

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

    def vali(self, vali_data, vali_loader, criterion, mode):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mode=mode)
                if mode == 'refine':
                    outputs = outputs[0]
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
        train_data_retrieval, train_loader_retrieval = self._get_data(flag='retrieval')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_refine = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_optim_refine = optim.Adam(self.model.parameters(), lr=self.args.refine_lr)
        criterion = self._select_criterion()

        if self.args.load_pretrained_backbone == 1:
            print('*******Loading Pretrained PIR Framework*********')
            backbone_path = os.path.join(self.args.bakcbone_checkpoints, setting)
            pretrain_backbone_path = backbone_path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(pretrain_backbone_path, map_location=self.device))
            vali_loss = self.vali(vali_data, vali_loader, criterion, mode='pretrain')
            test_loss = self.vali(test_data, test_loader, criterion, mode='pretrain')
            print("Vali Loss: {0:.7f} Test Loss: {1:.7f}".format(vali_loss, test_loss))
        elif self.args.load_pretrained_backbone == 2:
            print('*******Loading Pretrained Backbone*********')
            backbone_path = os.path.join(self.args.checkpoints, setting)
            backbone_path = backbone_path.replace('PIR', self.args.backbone)
            pretrain_backbone_path = backbone_path + '/' + 'checkpoint.pth'
            self.model.model.load_state_dict(torch.load(pretrain_backbone_path, map_location=self.device))
            vali_loss = self.vali(vali_data, vali_loader, criterion, mode='pretrain')
            test_loss = self.vali(test_data, test_loader, criterion, mode='pretrain')
            print("Vali Loss: {0:.7f} Test Loss: {1:.7f}".format(vali_loss, test_loss))
        else:
            backbone_path = os.path.join(self.args.bakcbone_checkpoints, setting)
            if not os.path.exists(backbone_path):
                os.makedirs(backbone_path)
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mode='pretrain')
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (
                                (self.args.train_epochs + self.args.refine_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    model_optim.step()

                print("Pretrain Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion, mode='pretrain')
                test_loss = self.vali(test_data, test_loader, criterion, mode='pretrain')

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, backbone_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = backbone_path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.prepare_retrieval_index(train_data_retrieval, train_loader_retrieval)
        person_sim = torch.FloatTensor(train_data.get_person_similarity()).to(self.device)
        criterion_MAE = nn.L1Loss()
        criterion_quality = nn.MSELoss(reduction='none')

        for epoch in range(self.args.refine_epochs):
            iter_count = 0
            train_loss = []
            quality_loss_all = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                iter_count += 1
                model_optim_refine.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                index = index.to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, index, mode='refine')
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs, quality = outputs
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                intermediate_results_loss = (criterion_quality(quality[0], batch_y).detach()
                                             .permute(0, 2, 1).mean(-1, keepdim=True))
                pred_loss = criterion(outputs, batch_y)
                quality_loss = criterion_MAE(quality[1], intermediate_results_loss)
                # Weight quality loss lower to focus optimization on predictions
                loss = pred_loss + 0.1 * quality_loss

                train_loss.append(pred_loss.item())
                quality_loss_all.append(quality_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | pred loss: {2:.7f}, quality loss: {3:.3f}"
                          .format(i + 1, epoch + 1, np.average(train_loss),np.average(quality_loss_all),))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.refine_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim_refine.step()

            print("Refine Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, mode='refine')
            test_loss = self.vali(test_data, test_loader, criterion, mode='refine')

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping_refine(vali_loss, self.model, path)
            if early_stopping_refine.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim_refine, epoch + 1, self.args, total_epochs=self.args.refine_epochs)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def prepare_retrieval_index(self, train_data, train_loader):
        print('*******Constructing the Retrieval Indexes*********')
        time_now = time.time()
        train_steps = len(train_loader)
        self.model.construct_index(len(train_data))
        with torch.no_grad():
            for epoch in range(1):
                iter_count = 0
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    index = index.to(self.device)
                    iter_count += 1
                    self.model.add_key_value(batch_x, batch_y[:, -self.args.pred_len:, :], index)

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1}".format(i + 1, epoch + 1))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((1 - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

        print('*******Finishing the Retrieval Indexes*********')
        self.model.value_permute = self.model.values.permute(2,0,1)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            train_data_retrieval, train_loader_retrieval = self._get_data(flag='retrieval')
            self.prepare_retrieval_index(train_data_retrieval, train_loader_retrieval)
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mode='refine')
                outputs, quality = outputs

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # gt_loss.append(intermediate_results_loss.detach().cpu().numpy())
                # pred_loss.append(quality[1].detach().cpu().numpy())
                # if i % 1 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.svg'))

        preds = np.array(preds, dtype=object)
        trues = np.array(trues, dtype=object)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        #
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'gt_loss.npy', gt_loss)
        # np.save(folder_path + 'pred_loss.npy', pred_loss)
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return mae, mse
