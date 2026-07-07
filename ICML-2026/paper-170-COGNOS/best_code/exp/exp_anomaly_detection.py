from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.losses import GWNRLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import adjusted_metric, affiliation_metric, auc_vus_metric, UCR_evaluate
from utils.post_processing import apply_moving_average_filter,apply_lowpass_filter, KalmanSmoothing

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def spot(normal_energy,test_energy, q=0.001):

    from utils.spot import SPOT_Fast

    s = SPOT_Fast(q)
    s.fit(normal_energy, test_energy)
    s.initialize(verbose=True)
    ret = s.run()
    threshold = np.mean(ret['thresholds'])

    return threshold

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.training_speed = 0.0
        self.inference_time = 0.0
        self.filter_time = 0.0
        self.COGNOS_time = 0.0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.lradj == 'cosine_warmup':
            init_lr = self.args.learning_rate * 0.1  # Warmup starts at 10%
        else:
            init_lr = self.args.learning_rate
        model_optim = optim.Adam(self.model.parameters(), lr=init_lr)
        return model_optim

    def _select_criterion(self):
        if self.args.use_Gaussian_regularization:
            MMD_BANDWIDTHS = [0.05, 0.1, 0.5, 1.0, 2.0]  # Median Heuristic
            criterion = GWNRLoss(
                                 num_features=self.args.c_out,
                                 bandwidths=MMD_BANDWIDTHS,alphaGR=1)
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                if self.args.model == 'CrossAD':
                    ms_x_dec, ms_gt = self.model(batch_x, None, None, None)
                    pred = ms_x_dec
                    true = ms_gt

                else:
                    outputs = self.model(batch_x, None, None, None)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    pred = outputs
                    true = batch_x
                loss = criterion(pred, true)
                total_loss.append(loss.cpu())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self.anomaly_criterion = nn.MSELoss()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # CODE-01: Initialize epoch loss tracker
        open(path + '/epoch_losses.txt', 'w').close()

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        all_training_speed = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                if self.args.use_Gaussian_regularization:
                    input_x = batch_x
                    # Normalization
                    means = input_x.mean(1, keepdim=True).detach()
                    input_x = input_x.sub(means)
                    stdev = torch.sqrt(
                        torch.var(input_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                    input_x = input_x.div(stdev)

                    if self.args.model == 'CrossAD':
                        ms_x_dec, _ = self.model(input_x, None, None, None)
                        ms_x_dec = self.model.ms_interpolate(ms_x_dec)
                        outputs = ms_x_dec
                    else:
                        outputs = self.model(input_x, None, None, None)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, :, f_dim:]

                    # De-Normalization
                    outputs = outputs.mul(
                        (stdev[:, 0, :].unsqueeze(1).repeat(
                            1, self.args.seq_len, 1)))
                    outputs = outputs.add(
                        (means[:, 0, :].unsqueeze(1).repeat(
                            1, self.args.seq_len, 1)))

                    loss = criterion(outputs, batch_x, self.model)
                else:
                    if self.args.model == 'CrossAD':
                        ms_x_dec, ms_gt = self.model(batch_x, None, None, None)
                        outputs = ms_x_dec
                        batch_x = ms_gt
                    else:
                        outputs = self.model(batch_x, None, None, None)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, :, f_dim:]
                    loss = criterion(outputs, batch_x)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    all_training_speed.append(speed)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # CODE-04: Global gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, self.anomaly_criterion)
            test_loss = self.vali(test_data, test_loader, self.anomaly_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # CODE-01: Save per-epoch checkpoint and record test loss
            torch.save(self.model.state_dict(),
                       path + '/' + 'checkpoint_epoch_' + str(epoch + 1) + '.pth')
            with open(path + '/epoch_losses.txt', 'a') as elf:
                elf.write(str(epoch + 1) + ',' + str(test_loss) + '\n')
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.training_speed = np.average(all_training_speed)

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open(folder_path + "result_anomaly_detection.csv", 'a+')
        f.write("Training speed (s/iter), {:0.4e} \n".format(
            self.training_speed) + "  \n")


    def test(self, setting,test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        num_test_sample = test_data.test.shape[0]
        print('test_data.test.shape', num_test_sample)

        print('loading '+f'_GR:{self.args.use_Gaussian_regularization}_'+'model')
        # Use EarlyStopping best checkpoint (not per-epoch: test recon loss != anomaly F1)
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))


        if self.args.use_KalmanSmoothing:
            kalmansmoothing = KalmanSmoothing(self.args)

            total_train_length = len(train_data)
            total_test_length = len(test_data)

            cache_dir = "residual_cache/"+ setting + '/'
            cache_file_path = os.path.join(cache_dir, "residual_data.npz")

            os.makedirs(cache_dir, exist_ok=True)

            infer_time = 0.0


            if self.args.use_Gaussian_regularization:
                if os.path.exists(cache_file_path):
                    print(f"Cache file  exists：'{cache_file_path}'Load from cache...")
                    cached_data = np.load(cache_file_path)
                    anomaly_scores_KF = cached_data['anomaly_scores_KF']
                    test_labels_np = cached_data['test_labels_np']
                    test_len = cached_data['test_len']

                else:
                    print("Cache not exist ...")
                    train_residual_series = kalmansmoothing.process_residual_observations_per_channel(
                       self.model, train_loader, total_train_length, is_train_set=True
                    )

                    time_now = time.time()
                    test_residual_series, test_labels_np = kalmansmoothing.process_residual_observations_per_channel(
                        self.model, test_loader, total_test_length, is_train_set=False
                    )
                    infer_time = (time.time()-time_now) / num_test_sample

                    print("GRInference time: {:0.4e} s/point".format(infer_time))


                    norm_anomaly_scores_KF = kalmansmoothing.apply_adaptive_kalman_filter(train_residual_series)

                    time_now = time.time()
                    test_anomaly_scores_KF = kalmansmoothing.apply_adaptive_kalman_filter(test_residual_series)
                    KF_time = (time.time() - time_now) / num_test_sample
                    self.filter_time = KF_time
                    print("KF time: {:0.4e} s/point".format(KF_time))

                    self.COGNOS_time = infer_time + KF_time
                    print("COGNOS time: {:0.4e} s/point".format(self.COGNOS_time))

                    anomaly_scores_KF = np.concatenate((norm_anomaly_scores_KF, test_anomaly_scores_KF), axis=0)

                    test_len = test_residual_series.shape[0]

                    print(f"Saving cache file：'{cache_file_path}'...")
                    np.savez(
                        cache_file_path,
                        anomaly_scores_KF=anomaly_scores_KF,
                        test_labels_np=test_labels_np,
                        test_len=test_len,
                    )

                self.assemble_metrics(setting, anomaly_scores_KF, test_len, test_labels_np, flag='Kalman')

                folder_path = './test_results/' + setting + '/'
                f = open(folder_path + "result_anomaly_detection.csv", 'a+')
                f.write(
                    "GRInference time (ms/point), {:0.4e} \nKF time (ms/point), {:0.4e} \nCOGNOS time (ms/point), {:0.4e}".format(
                        infer_time * 1000, self.filter_time * 1000, self.COGNOS_time * 1000) + "  \n")

            else:
                if os.path.exists(cache_file_path):
                    print(f"Cache file  exists：'{cache_file_path}'Load from cache...")
                    cached_data = np.load(cache_file_path)
                    anomaly_scores_KF = cached_data['anomaly_scores_KF']
                    test_labels_np = cached_data['test_labels_np']
                    test_len = cached_data['test_len']

                else:
                    print("Cache not exist ...")
                    train_residual_series = kalmansmoothing.process_residual_observations_per_channel(
                        self.model, train_loader, total_train_length, is_train_set=True
                    )

                    test_residual_series, test_labels_np = kalmansmoothing.process_residual_observations_per_channel(
                        self.model, test_loader, total_test_length, is_train_set=False
                    )

                    norm_anomaly_scores_KF = kalmansmoothing.apply_adaptive_kalman_filter(train_residual_series)

                    test_anomaly_scores_KF = kalmansmoothing.apply_adaptive_kalman_filter(test_residual_series)

                    anomaly_scores_KF = np.concatenate((norm_anomaly_scores_KF, test_anomaly_scores_KF), axis=0)

                    test_len = test_residual_series.shape[0]
                    anomaly_scores_KF = anomaly_scores_KF.reshape(-1, self.args.c_out)
                    anomaly_scores_KF = np.sum(anomaly_scores_KF, axis=1)

                    print(f"Saving cache file：'{cache_file_path}'...")
                    np.savez(
                        cache_file_path,
                        anomaly_scores_KF=anomaly_scores_KF,
                        test_labels_np=test_labels_np,
                        test_len=test_len,
                    )

                self.assemble_metrics(setting, anomaly_scores_KF, test_len, test_labels_np, flag='vanilla_KF')

        else:

            attens_energy = []
            self.model.eval()
            self.anomaly_criterion = nn.MSELoss(reduce=False)
            # (1) stastic on the train set
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    batch_x = batch_x.float().to(self.device)
                    # reconstruction
                    if self.args.model == 'CrossAD':
                        ms_x_dec, ms_gt = self.model(batch_x, None, None, None)
                        score = self.model.ms_anomaly_score(ms_x_dec, ms_gt).mean(dim=-1)
                    else:

                        outputs = self.model(batch_x, None, None, None)
                        # criterion
                        score = self.anomaly_criterion(batch_x, outputs).mean(dim=-1)
                    score = score.detach().cpu().numpy()
                    attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)

            # (2) find the threshold
            attens_energy = []
            test_labels = []
            time_now = time.time()
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                if self.args.model == 'CrossAD':
                    ms_x_dec, ms_gt = self.model(batch_x, None, None, None)
                    score = self.model.ms_anomaly_score(ms_x_dec, ms_gt).mean(dim=-1)
                else:
                    outputs = self.model(batch_x, None, None, None)
                    # criterion
                    score = self.anomaly_criterion(batch_x, outputs).mean(dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)
            infer_time = (time.time() - time_now) / num_test_sample
            self.inference_time = infer_time
            print("Vanilla Inference time: {:0.4e}".format(infer_time))

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            test_len = test_energy.shape[0]
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)

            combined_energy = np.concatenate([train_energy, test_energy], axis=0)

            self.assemble_metrics(setting, combined_energy, test_len, test_labels, flag='vanilla')

            folder_path = './test_results/' + setting + '/'
            f = open(folder_path + "result_anomaly_detection.csv", 'a+')
            f.write("Vanilla Inference time (ms/point), {:0.4e} \n".format(
            self.inference_time *1000)+ "  \n")

            if self.args.use_Gaussian_regularization:
                anomaly_scores_MA = apply_moving_average_filter(combined_energy)

                anomaly_scores_LPF = apply_lowpass_filter(combined_energy)

                self.assemble_metrics(setting, anomaly_scores_MA, test_len, test_labels, flag='Moving Average')
                self.assemble_metrics(setting, anomaly_scores_LPF, test_len, test_labels, flag='Low Pass')


    def assemble_metrics(self, setting, anomaly_scores, test_len, test_labels_np, flag='Kalman'):


        if flag == 'Kalman':
            test_labels_flatten = test_labels_np.flatten()
            test_labels_per_point = test_labels_np[:, 0]
            anomaly_scores_channel = anomaly_scores.reshape(-1, self.args.c_out)
            agg_anomaly_scores = np.sum(anomaly_scores_channel, axis=1)
            test_anomaly_scores = agg_anomaly_scores.reshape(-1)[-test_len:]
            test_energy = anomaly_scores[-test_len * self.args.c_out:]
            train_energy = anomaly_scores[:-test_len * self.args.c_out]
        else:
            test_energy = anomaly_scores[-test_len:]
            test_anomaly_scores = test_energy
            test_labels_flatten = test_labels_np
            test_labels_per_point = test_labels_np
            train_energy = anomaly_scores[:-test_len]

        folder_path = './test_results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.anomaly_ratio == 0:
            threshold = spot(train_energy, test_energy)
        else:
            threshold = np.percentile(anomaly_scores, 100 - self.args.anomaly_ratio)

        print("Threshold :", threshold)

        pred = (test_energy > threshold).astype(int)
        gt = test_labels_flatten.astype(int)

        # adjusted_metric
        precision, recall, f_score = adjusted_metric(pred, gt)
        # affiliation_metric
        Aff_precision, Aff_recall, Aff_F_score = affiliation_metric(pred, gt)

        f = open(folder_path + "result_anomaly_detection.csv", 'a+')
        f.write(setting +f"_{flag}_"+ "  \n")
        f.write("Precision, {:0.4f} \nRecall, {:0.4f} \nF-score, {:0.4f} ".format(
            precision, recall, f_score)+ "  \n")
        f.write("Aff_precision, {:0.4f} \nAff_recall, {:0.4f} \nAff_F_score, {:0.4f}".format(
            Aff_precision, Aff_recall, Aff_F_score)+ "  \n")

        if self.args.eval_vus:
            auc_vus_storage = {}
            auc_vus_metric(auc_vus_storage, ['auc', 'r_auc','vus'], labels=test_labels_per_point, score=test_anomaly_scores, version='opt_mem', thre=250)
            for key, value in auc_vus_storage.items():
                f.write(key + ', {:0.4f}'.format(value) + '\n')
                print(key, value)

        if self.args.data == 'UCR':
            UCR_storage = {}
            UCR_evaluate(UCR_storage, test_labels_per_point, test_anomaly_scores)
            for key, value in UCR_storage.items():
                f.write(key + ', {:0.4f}'.format(value) + '\n')
                print(key, value)

        f.write('\n')
        f.write('\n')
        f.close()




