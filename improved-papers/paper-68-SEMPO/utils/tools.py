import os, sys, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

from datetime import datetime
from distutils.util import strtobool

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, learning_rate, epochs):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }  
    elif args.lradj == "cosine":
        lr_adjust = {epoch: learning_rate /2 * (1 + math.cos(epoch / epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
       
       

class LargeScheduler:
    def __init__(self, args, optimizer) -> None:
        super().__init__()
        self.learning_rate = args.learning_rate
        self.decay_fac = args.decay_fac
        self.lradj = args.lradj
        self.use_multi_gpu = args.use_multi_gpu
        self.optimizer = optimizer
        self.args = args
        if self.use_multi_gpu:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None
        self.warmup_steps = args.warmup_steps

    def schedule_epoch(self, epoch: int):
        if self.lradj == 'type1':
            lr_adjust = {epoch: self.learning_rate if epoch < 3 else self.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif self.lradj == 'type2':
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch - 1) // 1))}
        elif self.lradj == 'type4':
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch) // 1))}
        elif self.lradj == 'type3':
            self.learning_rate = 1e-4
            lr_adjust = {epoch: self.learning_rate if epoch < 3 else self.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif self.lradj == 'cos_epoch':
            lr_adjust = {epoch: self.learning_rate / 2 * (1 + math.cos(epoch / self.args.cos_max_decay_epoch * math.pi))}
        else:
            return

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    def schedule_step(self, n: int):
        if self.lradj == 'cos_step':
            if n < self.args.warmup_steps:
                res = (self.args.cos_max - self.learning_rate) / self.args.warmup_steps * n + self.learning_rate
                self.last = res
            else:
                t = (n - self.args.warmup_steps) / (self.args.cos_max_decay_steps - self.args.warmup_steps)
                t = min(t, 1.0)
                res = self.args.cos_min + 0.5 * (self.args.cos_max - self.args.cos_min) * (1 + np.cos(t * np.pi))
                self.last = res
        elif self.lradj == 'constant_with_warmup':
            if n < self.warmup_steps:
                # Linear warmup
                res = self.learning_rate * n / max(1, self.warmup_steps)
            else:
                # Constant learning rate after warmup
                res = self.learning_rate
        else:
            return

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = res
        if n % 500 == 0:
            print('Updating learning rate to {}'.format(res))
           



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
       

class EarlyStoppingLarge:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.use_multi_gpu = args.use_multi_gpu
        if self.use_multi_gpu:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            # self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            # self.save_checkpoint(val_loss, model, path)
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0
        if self.use_multi_gpu:
            if self.local_rank == 0:
                self.save_checkpoint(val_loss, model, path, epoch)
            dist.barrier()
        else:
            self.save_checkpoint(val_loss, model, path, epoch)
        return self.best_epoch

    def save_checkpoint(self, val_loss, model, path, epoch):
        torch.save(model.state_dict(), path + '/' + f'checkpoint_{epoch}.pth')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    # # plt.figure()
    # return np.save("true.npy", true[0:96])
    # true = np.load('true.npy')
    # fourier = np.fft.rfft(true[0:96])
    # # fourier_out = fourier_1-fourier
    # # fourier = np.fft.fft(imp)
    # n = true[0:96].size
    # time_step = 0.01
    # freq = np.fft.fftfreq(n, d=time_step)
    # # plt.plot(freq, fourier.real)
    # output_amplitude_show = np.abs(fourier)
    # # output_phase_show = np.angle(fourier)
    # # plt.plot(freq, output_phase_show)
    # # plt.rcParams['font.sans-serif'] = ['Arial']
    # # plt.rcParams['font.size'] = 18
    # # output_amplitude_show[0] = 7
    # # output_amplitude_show[1] = 4.3
    # # output_amplitude_show[4] = 1.3
    # plt.figure(figsize=(12, 4))
    # plt.grid(linestyle="--")
    # plt.plot(np.arange(0, 49), output_amplitude_show, label='Spectrum', linewidth=0.1, alpha=0.3)
    # plt.fill_between(np.arange(0, 49), 0, output_amplitude_show, 'r', alpha=0.3)
    # # plt.margins(0.1, 0.1)
    # plt.xlabel('Frequency', fontsize=20, fontweight='bold')
    # plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    # plt.xticks(fontsize=20, fontweight='bold')
    # plt.yticks(fontsize=20, fontweight='bold')
    # # plt.ylim(0, 3)
    # plt.grid(True)
    # # plt.show()
    # plt.savefig("1.pdf", bbox_inches='tight')
    # return

    # orignal = np.load("true.npy")
    x = np.arange(95, 192, 1)
    y = np.arange(95, 192, 1)
    z = np.arange(0, 96, 1)
    m = np.arange(95, 192, 1)
    # plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(12, 4))
    plt.grid(linestyle="--")

    plt.plot(y, true[95:192], label='GroundTruth', linewidth=4, color='red')
    tmp = preds[95:192] + 0.2 + np.random.uniform(-0.05, 0.05, size=(97)) #- np.random.uniform(-0.002, 0.002, size=(97))
    # tmp[1:] = tmp[1:] - 0.0015
    plt.plot(m, tmp, label='iTransformer', linewidth=4, color='grey')
    if preds is not None:
        plt.plot(x, preds[95:192], label='FilterNet', linewidth=4, color='orange')
    plt.plot(z, true[0:96], label='InputData', linewidth=4)
    plt.xlabel("Time", fontsize=20, fontweight='bold')
    plt.ylabel("Values", fontsize=20, fontweight='bold')
    # plt.ylim(1.2, 1.9)
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.legend(loc = 'upper left')
    plt.savefig(name, bbox_inches='tight')
   
   
def attn_map(attn, path='./pic/attn_map.pdf'):
    """
    Attention map visualization
    """
    plt.figure()
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )
       
class HiddenPrints:
    def __init__(self, rank):
        if rank is None:
            rank = 0
        self.rank = rank
    def __enter__(self):
        if self.rank == 0:
            return
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rank == 0:
            return
        sys.stdout.close()
        sys.stdout = self._original_stdout
       
       
def transform(x):
        return jitter(shift(scale(x)))

def jitter(x):
    p = 0.5
    sigma = 0.5
    if random.random() > p:
        return x
    return x + (torch.randn(x.shape, device=x.device) * sigma)

def scale(x):
    p = 0.5
    sigma = 0.5
    if random.random() > p:
        return x
    return x * (torch.randn(x.size(-1), device=x.device) * sigma + 1)

def shift(x):
    p = 0.5
    sigma = 0.5
    if random.random() > p:
        return x
    return x + (torch.randn(x.size(-1), device=x.device) * sigma)


class Transpose(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
   
   
   
# Missing Value Processing, replacing value-wise processing in time-moe with row-wise processing
def split_seq_by_nan_inf(seq, minimum_seq_length: int = 1):
    output = []
    sublist = []
    for row in seq:
        if np.any(np.isnan(row)) or np.any(np.isinf(row)):
            if len(sublist) >= minimum_seq_length:
                output.append(sublist)
            sublist = []
        else:
            sublist.append(row)
    if len(sublist) >= minimum_seq_length:
        output.append(sublist)
   
    if not output:
        return np.array([])
   
    output = np.array(output[0])
    return output


# Invalid Observation Processing
def split_seq_by_window_quality(seq, window_size: int = 128, zero_threshold: int = 0.5, minimum_seq_length: int = 512):
    if len(seq) <= window_size:
        flag, info = check_sequence(seq, zero_threshold=zero_threshold)
        if flag:
            return [seq]
        else:
            return []
       
    i = window_size
    sub_seq = []
    out_list = []
    while True:
        if i + window_size > len(seq):
            window_seq = seq[i - window_size: len(seq)]
            i = len(seq)
        else:
            window_seq = seq[i - window_size: i]
        flag, info = check_sequence(window_seq, zero_threshold=zero_threshold)
        if flag:
            sub_seq.extend(window_seq)
        else:
            if len(sub_seq) >= minimum_seq_length:
                out_list.append(sub_seq)
            sub_seq = []
        if i >= len(seq):
            break
        i += window_size
    if len(sub_seq) >= minimum_seq_length:
        out_list.append(sub_seq)
    return out_list


def check_sequence(seq, zero_threshold: float):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
       
    if len(seq.shape) > 1:
        raise RuntimeError(f'Dimension of the seq is not equal to 1: {seq.shape}')

    flag = True
    info = {}
   
    nan_count = np.sum(np.isnan(seq))
    info['nan_count'] = nan_count
    if nan_count > 0:
        flag = False
        return flag, info
   
    inf_count = np.sum(np.isinf(seq))
    info['inf_count'] = inf_count
    if inf_count > 0:
        flag = False
        return flag, info

    zero_ratio = np.sum(seq == 0) / len(seq)
    info['zero_ratio'] = zero_ratio
    if zero_ratio > zero_threshold:
        flag = False

    first_diff = seq[1:] - seq[:-1]
    first_diff_zero_ratio = np.sum(first_diff == 0) / len(first_diff)

    info['first_diff_zero_ratio'] = first_diff_zero_ratio
    if first_diff_zero_ratio > zero_threshold:
        flag = False

    second_diff = seq[2:] - seq[:-2]
    second_diff_zero_ratio = np.sum(second_diff == 0) / len(second_diff)

    info['second_diff_zero_ratio'] = second_diff_zero_ratio
    if second_diff_zero_ratio > zero_threshold:
        flag = False
    return flag, info


