from tqdm import tqdm
from math import ceil
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from attack_backtime_trigger_generator import TgrGCN


def fft_compress(raw_data_seq, n_components=200):
    """
    compress the time series data using fft to have global representation for each variable.
    """
    if len(raw_data_seq.shape) == 2:
        raw_data_seq = raw_data_seq[:, :, None]
    data_seq = raw_data_seq[:, :, 0:1] # Just one channel :) 
    # data_seq: (l, n, c)
    l, n, c = data_seq.shape
    data_seq = data_seq.reshape(l, -1).transpose()
    # use fft to have the amplitude, phase, and frequency for each time series data
    fft_data = np.fft.fft(data_seq, axis=1)
    amplitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    frequency = np.fft.fftfreq(l)

    # choose the top n_components frequency components
    top_indices = np.argsort(amplitude, axis=1)[::-1][:, :n_components]
    amplitude_top = amplitude[np.arange(amplitude.shape[0])[:, None], top_indices]
    phase_top = phase[np.arange(phase.shape[0])[:, None], top_indices]
    frequency_top = frequency[top_indices]
    feature_top = np.concatenate([amplitude_top, phase_top, frequency_top], axis=1)
    return feature_top


class Attacker:
    def __init__(self, dataset, channel_features, atk_vars, config, target_pattern, device='cuda'):
        """
        the attacker class is used to inject triggers and target patterns into the dataset.
        the attacker class has the full access to the dataset and the trigger generator.
        """
        self.config = config
        self.device = device
        self.dataset = dataset

        self.target_pattern = target_pattern
        self.atk_vars = atk_vars

        self.trigger_generator = TgrGCN(config, sim_feats=channel_features, atk_vars=atk_vars, device=device)
        self.trigger_len = config.trigger_len
        self.pattern_len = config.pattern_len
        self.bef_tgr_len = config.bef_tgr_len  # the length of the data before the trigger to generate the trigger

        self.fct_input_len = config.seq_len  # the length of the input for the forecast model
        self.fct_output_len = config.pred_len  # the length of the output for the forecast model
        self.alpha_t = config.alpha_t
        self.alpha_s = config.alpha_s
        self.temporal_poison_num = ceil(self.alpha_t * len(self.dataset))

        self.trigger_generator = self.trigger_generator.to(device)
        
        self.attack_optim = optim.Adam(self.trigger_generator.parameters(), 
                                       lr=config.attack_lr)
        
        self.atk_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.attack_optim, 
                                                                  milestones=[20, 40], 
                                                                  gamma=0.9)

        self.lam_norm = config.lam_norm

    def state_dict(self):
        attacker_state = {
            'target_pattern': self.target_pattern.cpu().detach().numpy(),
            'trigger_generator': self.trigger_generator.state_dict(),
            'trigger_len': self.trigger_len,
            'pattern_len': self.pattern_len,
            'bef_tgr_len': self.bef_tgr_len,
            'fct_input_len': self.fct_input_len,
            'fct_output_len': self.fct_output_len,
            'alpha_t': self.alpha_t,
            'alpha_s': self.alpha_s,
            'temporal_poison_num': self.temporal_poison_num,
            'lam_norm': self.lam_norm,
            'attack_optim': self.attack_optim.state_dict(),
            'atk_scheduler': self.atk_scheduler.state_dict(),
            'atk_ts': self.atk_ts.cpu().detach().numpy() if hasattr(self, 'atk_ts') else None,
            'atk_vars': self.atk_vars.cpu().detach().numpy() if hasattr(self, 'atk_vars') else None,
        }
        return attacker_state

    def load_state_dict(self, attacker_state):
        self.trigger_len = attacker_state['trigger_len']
        self.pattern_len = attacker_state['pattern_len']
        self.bef_tgr_len = attacker_state['bef_tgr_len']
        self.fct_input_len = attacker_state['fct_input_len']
        self.fct_output_len = attacker_state['fct_output_len']
        self.alpha_t = attacker_state['alpha_t']
        self.alpha_s = attacker_state['alpha_s']
        self.temporal_poison_num = attacker_state['temporal_poison_num']
        self.lam_norm = attacker_state['lam_norm']

        self.trigger_generator.load_state_dict(attacker_state['trigger_generator'])
        self.attack_optim.load_state_dict(attacker_state['attack_optim'])
        self.atk_scheduler.load_state_dict(attacker_state['atk_scheduler'])
        self.target_pattern = torch.from_numpy(attacker_state['target_pattern'])
        self.atk_ts = torch.from_numpy(attacker_state['atk_ts']) if attacker_state['atk_ts'] is not None else None
        self.atk_vars = torch.from_numpy(attacker_state['atk_vars']) if attacker_state['atk_vars'] is not None else None

        self.trigger_generator = self.trigger_generator.to(self.device)
        self.target_pattern = self.target_pattern.to(self.device)
        if self.atk_ts is not None:
            self.atk_ts = self.atk_ts.to(self.device)
        if self.atk_vars is not None:
            self.atk_vars = self.atk_vars.to(self.device)

    def eval(self):
        self.trigger_generator.eval()

    def train(self):
        self.trigger_generator.train()

    def set_atk_timestamp(self, atk_ts):
        """
        set the attack timestamp for the attacker.
        """
        self.atk_ts = atk_ts

    def set_atk_variables(self, atk_var):
        """
        set the attack variables for the attacker.
        """
        self.atk_vars = atk_var

    def set_atk(self, atk_ts, atk_var):
        self.set_atk_timestamp(atk_ts)
        self.set_atk_variables(atk_var)

    def sparse_inject(self):
        """
        Inject the trigger and target pattern into all the variables at the attack timestamp.
        """
        assert hasattr(self, 'atk_vars'), 'Please set the attack variable first.'
        assert hasattr(self, 'atk_ts'), 'Please set the attack timestamp first.'
        self.dataset.init_poison_data()

        n, c, T = self.dataset.data.shape
        n = len(self.atk_vars)
        trigger_len = self.trigger_generator.output_dim
        pattern_len = self.target_pattern.shape[-1]

        for beg_idx in self.atk_ts.tolist():
            data_bef_tgr = self.dataset.data[self.atk_vars, 0:1, beg_idx - self.trigger_generator.input_dim:beg_idx]
            data_bef_tgr = self.dataset.normalize(data_bef_tgr)
            data_bef_tgr = data_bef_tgr.reshape(-1, self.trigger_generator.input_dim)

            triggers = self.trigger_generator(data_bef_tgr)[0]
            triggers = self.dataset.denormalize(triggers).reshape(n, 1, -1)

            # inject the trigger and target pattern
            self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx:beg_idx + trigger_len] = triggers.detach()
            self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx + trigger_len:beg_idx + trigger_len + pattern_len] = \
                self.target_pattern + self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx - 1:beg_idx]

    def predict_trigger(self, data_bef_trigger):
        """
        predict the trigger using the trigger generator.
        n = number of samples, c = number of variables, l = length of the data
        :param data_bef_trigger: the data before the trigger, shape: (n, c, l).
        :return: the predicted trigger, shape: (n, c, trigger_len)
        """
        c, l = data_bef_trigger.shape[-2:]
        data_bef_trigger = self.dataset.normalize(data_bef_trigger)
        data_bef_trigger = data_bef_trigger.view(-1, self.trigger_generator.input_dim)
        triggers, perturbations = self.trigger_generator(data_bef_trigger)
        triggers = self.dataset.denormalize(triggers).reshape(-1, c, self.trigger_len)
        return triggers, perturbations

    def get_trigger_slices(self, bef_len, aft_len):
        """
        A easy implementation to limit the range for soft identification.
        find all the sliced time window that contains the trigger.
        :return: a list of slices. time range: [idx - bef_len, idx + aft_len], where idx is the start index of triggers
        """
        slices = []
        timestamps = []

        for idx in self.atk_ts.tolist():
            if idx + aft_len < self.dataset.poisoned_data.shape[-1] and idx - bef_len >= 0:
                slices.append(self.dataset.poisoned_data[..., idx - bef_len:idx + aft_len].detach())
                if self.dataset.use_timestamps:
                    timestamps.append(self.dataset.timestamps[idx - bef_len:idx + aft_len])

        if not self.dataset.use_timestamps:
            return slices
        return slices, timestamps

    def select_atk_timestamp(self, poison_metrics):
        """
        select the attack timestamp using the poison metrics (clean MAE). poison_metrics: a list of [mae, idx]
        """
        select_pos_mark = torch.zeros(len(self.dataset), dtype=torch.int)
        poison_metrics = poison_metrics.to(self.device)
        sort_idx = torch.argsort(poison_metrics[:, 0], descending=True).detach().cpu().numpy()
        # ensure the distance between two poison indices is larger than trigger length + pattern length, avoid overlap
        valid_idx = []
        for i in range(len(sort_idx)):
            # use greedy algorithm to select the valid indices with the largest poison metrics
            beg_idx = int(poison_metrics[sort_idx[i], 1])
            end_idx = beg_idx + self.trigger_len + self.pattern_len + 8  # 8: the magic number to avoid overlap
            if torch.sum(select_pos_mark[beg_idx:end_idx]) == 0 and \
                    end_idx < len(self.dataset) and beg_idx > self.bef_tgr_len:
                valid_idx.append(sort_idx[i])
                select_pos_mark[beg_idx:end_idx] = 1
            if len(valid_idx) > 2 * self.temporal_poison_num:
                print('break due to enough valid indices')
                break

        valid_idx = np.array(valid_idx)
        # random select the temporal poison indices. add randomness to avoid overfitting
        top_sort_idx = np.random.choice(valid_idx, min(self.temporal_poison_num, valid_idx.shape[0]), replace=False)
        top_sort_idx = torch.from_numpy(top_sort_idx).to(self.device)
        atk_ts = poison_metrics[top_sort_idx, 1].long()
        # sort poison indices
        atk_ts = torch.sort(atk_ts)[0]
        self.set_atk_timestamp(atk_ts)

    def update_trigger_generator(self, net, epoch, epochs, use_timestamps):
        """
        update the trigger generator using the soft identification.
        """
        if not use_timestamps:
            tgr_slices = self.get_trigger_slices(self.fct_input_len - self.trigger_len,
                                                 self.trigger_len + self.pattern_len + self.fct_output_len)
        else:
            tgr_slices, tgr_timestamps = self.get_trigger_slices(self.fct_input_len - self.trigger_len,
                                                                 self.trigger_len + self.pattern_len + self.fct_output_len)
            
        pbar = tqdm(tgr_slices, desc=f'Attacking data {epoch}/{epochs}', dynamic_ncols=True)
        for slice_id, slice in enumerate(pbar):
            slice = slice.to(self.device)
            slice = slice[:, 0:1, :]
            data_bef = slice[self.atk_vars, :,
                       self.fct_input_len - self.trigger_len - self.bef_tgr_len:self.fct_input_len - self.trigger_len]
            data_bef = data_bef.reshape(-1, self.bef_tgr_len)

            triggers, perturbations = self.predict_trigger(data_bef)

            # add the trigger to the slice. x[t-trigger_len:x] = trigger
            triggers = triggers.reshape(self.atk_vars.shape[0], -1, self.trigger_len)
            slice[self.atk_vars, :, self.fct_input_len - self.trigger_len:self.fct_input_len] = triggers

            # add the pattern to the slice. x[t:t+ptn_len] = x[t-1-trigger_len] + target_pattern
            slice[self.atk_vars, :, self.fct_input_len:self.fct_input_len + self.pattern_len] = \
                self.target_pattern + slice[self.atk_vars, :, self.fct_input_len - self.trigger_len - 1].unsqueeze(-1)

            # mimic the soft identification, i.e., the input and output only contain a part of the trigger and pattern
            batch_inputs_bkd = [slice[..., i:i + self.fct_input_len] for i in range(self.pattern_len)]
            batch_labels_bkd = [slice[..., i + self.fct_input_len:i + self.fct_input_len + self.fct_output_len].detach()
                                for i in range(self.pattern_len)]
            batch_inputs_bkd = torch.stack(batch_inputs_bkd, dim=0)
            batch_labels_bkd = torch.stack(batch_labels_bkd, dim=0)

            batch_inputs_bkd = batch_inputs_bkd[:, :, 0:1, :]
            batch_labels_bkd = batch_labels_bkd[:, :, 0, :]
            batch_inputs_bkd = self.dataset.normalize(batch_inputs_bkd)

            # calculate eta in the soft identification to reweight the loss
            loss_decay = (self.pattern_len - torch.arange(0, self.pattern_len, dtype=torch.float32).to(
                self.device)) / self.pattern_len

            self.attack_optim.zero_grad()

            batch_inputs_bkd = batch_inputs_bkd.squeeze(2).permute(0, 2, 1)
            batch_labels_bkd = batch_labels_bkd.permute(0, 2, 1)

            if use_timestamps:
                batch_x_mark = [tgr_timestamps[slice_id][i:i + self.fct_input_len] 
                                                        for i in range(self.pattern_len)]
                batch_y_mark = [tgr_timestamps[slice_id][i + self.fct_input_len:i + self.fct_input_len + self.fct_output_len] 
                                                        for i in range(self.pattern_len)]
                batch_x_mark = torch.stack(batch_x_mark, dim=0)
                batch_y_mark = torch.stack(batch_y_mark, dim=0)
            else:
                batch_x_mark = torch.zeros(batch_inputs_bkd.shape[0], batch_inputs_bkd.shape[1], 4).to(self.device)

            x_des = torch.zeros_like(batch_labels_bkd)

            outputs_bkd = net(batch_inputs_bkd, batch_x_mark, x_des, None) 
            outputs_bkd = self.dataset.denormalize(outputs_bkd)
            loss_bkd = F.mse_loss(outputs_bkd[:, :, self.atk_vars], batch_labels_bkd[:, :, self.atk_vars], reduction='none')
            loss_bkd = torch.mean(loss_bkd, dim=(1, 2))
            loss_bkd = torch.sum(loss_bkd * loss_decay)  # reweight the loss
            loss_norm = torch.abs(torch.sum(perturbations, dim=1)).mean()
            loss = loss_bkd + self.lam_norm * loss_norm
            loss.backward()
            self.attack_optim.step()
        self.atk_scheduler.step()
