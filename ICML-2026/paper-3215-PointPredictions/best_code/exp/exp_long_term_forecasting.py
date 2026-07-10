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
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)


        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


        self.total_params = total_params
        self.trainable_params = trainable_params


        print(">>> Model: {}".format(self.args.model))
        print(">>> Total params: {:.2f} M".format(total_params / 1e6))
        print(">>> Trainable params: {:.2f} M".format(trainable_params / 1e6))
        print(">>> Seq_len = {}, Pred_len = {}, d_model = {}, e_layers = {}, d_ff = {}".format(
            self.args.seq_len, self.args.pred_len, getattr(self.args, 'd_model', None),
            getattr(self.args, 'e_layers', None), getattr(self.args, 'd_ff', None)
        ))
        # ============================
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
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

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

            self.model.train()
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

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():

                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                        f_dim = -1 if self.args.features == 'MS' else 0

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)


                        if hasattr(self.model, 'module'):
                            if hasattr(self.model.module, 'aux_loss'):
                                loss += self.model.module.aux_loss
                        elif hasattr(self.model, 'aux_loss'):
                            loss += self.model.aux_loss

                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                    if hasattr(self.model, 'module'):
                        if hasattr(self.model.module, 'aux_loss'):
                            loss += self.model.module.aux_loss
                    elif hasattr(self.model, 'aux_loss'):
                        loss += self.model.aux_loss
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


        state = {
            'model': self.model.state_dict(),
            'args': self.args
        }

        torch.save(state, best_model_path)

        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model'])

        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []


        total_time = 0.0
        total_batches = 0
        warmup = 3


        if self.args.use_gpu:
            torch.cuda.reset_peak_memory_stats()


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


                if i >= warmup:
                    if self.args.use_gpu:
                        torch.cuda.synchronize()
                    start = time.perf_counter()


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if i >= warmup:
                    if self.args.use_gpu:
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    total_time += (end - start)
                    total_batches += 1


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
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
            avg_time_per_batch = total_time / total_batches
            avg_time_per_sample = avg_time_per_batch / self.args.batch_size
            throughput = self.args.batch_size / avg_time_per_batch

            print(">>> Inference (test) avg time: {:.3f} ms / batch, {:.3f} ms / sample".format(
                avg_time_per_batch * 1000, avg_time_per_sample * 1000
            ))
            print(">>> Inference throughput: {:.1f} samples / s".format(throughput))

            self.avg_time_per_batch = avg_time_per_batch
            self.avg_time_per_sample = avg_time_per_sample
            self.throughput = throughput
        else:
            print(">>> Not enough batches for timing (batches <= warmup).")


        if self.args.use_gpu:
            max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(">>> Max GPU memory during test: {:.1f} MB".format(max_mem))
            self.max_mem = max_mem


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
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        '''
         f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        '''
        with open("result_long_term_forecast_baseline_Crossformer.txt", "a") as f:

            f.write(setting + "  \n")


            f.write(
                "Seq_len = {}, Label_len = {}, Pred_len = {}, "
                "d_model = {}, e_layers = {}, d_ff = {}, "
                "enc_in = {}, c_out = {}\n".format(
                    self.args.seq_len,
                    self.args.label_len,
                    self.args.pred_len,
                    getattr(self.args, "d_model", None),
                    getattr(self.args, "e_layers", None),
                    getattr(self.args, "d_ff", None),
                    getattr(self.args, "enc_in", None),
                    getattr(self.args, "c_out", None),
                )
            )


            f.write(
                "mse:{}, mae:{}, dtw:{}, "
                "params:{:.2f}M, trainable_params:{:.2f}M, "
                "latency:{:.3f}ms/sample, throughput:{:.1f} samples/s, "
                "max_mem:{:.1f}MB,"
                "macs:{:.4f}G\n\n".format(
                    mse,
                    mae,
                    dtw,
                    getattr(self, "total_params", 0) / 1e6,
                    getattr(self, "trainable_params", 0) / 1e6,
                    getattr(self, "avg_time_per_sample", 0.0) * 1000,
                    getattr(self, "throughput", 0.0),
                    getattr(self, "max_mem", 0.0),
                    getattr(self, "macs", 0.0) / 1e9,
                )
            )


        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
