"""
TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting
"""

import os 
import random 
import argparse
import yaml
import time
from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np
import torch 
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors

from forecast_models import TimesNet, FEDformer, SimpleTM
from dataset_attack import load_raw_data
from dataset_defense import TimeDataset, AttackEvaluationSetLoad, TimeDatasetwithWeight
from utils.misc import get_current_datetime
from utils.distances import to_euclidean_from_pearson, weights_forecast_gaussian


MODEL_MAP = {"FEDformer": FEDformer,
            "TimesNet": TimesNet,
            "SimpleTM": SimpleTM}

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # To disable hash randomization and ensure the reproducibility of experiments.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parser_args():
    current_time = get_current_datetime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense_config_path", type=str, required=True)
    args = parser.parse_args()
    print(args)

    # load default configs/default_config.yaml
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)

    # load defense config
    config = yaml.load(open(args.defense_config_path), Loader=yaml.FullLoader)['Defense']

    # load dataset 
    config['Dataset'] = default_config['Dataset'][config['dataset']]

    # load models
    config['Model'] = default_config['Model'][config['model_name']]
    config['Model']['c_out'] = config['Dataset']['num_of_vertices']
    config['Model']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Model']['dec_in'] = config['Dataset']['num_of_vertices']
    config['Model']['token_len'] = config['token_len']
    config['Model']['seq_len'] = config['seq_len']
    config['Model']['label_len'] = config['label_len']
    config['Model']['pred_len'] = config['pred_len']

    # load training settings
    args_dict = vars(args)
    config.update(args_dict)
    config['current_time'] = current_time
    return edict(config)

class TimeGuardDefender:
    def __init__(self, config,
                 mean, std,  
                 train_data_seq, test_data_seq, 
                 train_data_stamps, test_data_stamps,
                 device):
        
        self.config = config
        self.device = device
        
        self.mean = mean
        self.std = std
        self.train_data_seq = train_data_seq
        self.test_data_seq = test_data_seq

        self.train_data_stamps = train_data_stamps
        self.test_data_stamps = test_data_stamps

        self.train_set = TimeDataset(raw_data=train_data_seq, 
                                     mean=mean,
                                     std=std, 
                                     device=device,
                                     num_for_hist=config.seq_len,
                                     num_for_futr=config.pred_len,
                                     timestamps=train_data_stamps)
        
        
        self.test_set = TimeDataset(raw_data=test_data_seq,
                                    mean=mean,
                                    std=std, 
                                    device=device,
                                    num_for_hist=config.seq_len,
                                    num_for_futr=config.pred_len,
                                    timestamps=test_data_stamps)


        dataset_dict = torch.load(os.path.join(config.attack_save_folder, 'test_attacked_data.pth'), map_location="cpu")
        print("Loading test poison data from", os.path.join(config.attack_save_folder, 'test_attacked_data.pth'))


        self.poison_test_set = AttackEvaluationSetLoad(dataset_dict, 
                                                        mean=mean,
                                                        std=std,
                                                        device=device)


        self.forecaster = MODEL_MAP[config.model_name](config.Model).to(device)

        config.Model_Backcaster = config.Model.copy()
        config.Model_Backcaster["seq_len"] = config.pred_len
        config.Model_Backcaster["pred_len"] = config.seq_len
        self.backcaster = MODEL_MAP[config.model_name](config.Model_Backcaster).to(device)


        self.batch_size = config.batch_size
        self.use_timestamps = self.train_set.use_timestamps

        ## For neighborhood_distance_calculation
        self._all_sample_data = None # (N, C, T)
        self._all_sample_data_euclid = None
        self._pearson_weighted = weights_forecast_gaussian(L=config.seq_len, F=config.pred_len)

    def _get_all_sample_data(self):

        # Avoid recomputing
        if self._all_sample_data is not None and self._all_sample_data_euclid is not None:
            return self._all_sample_data, self._all_sample_data_euclid

        all_sample = []
        train_loader = DataLoader(self.train_set, batch_size=512, shuffle=False)  # bigger batch, fewer iters
        with torch.no_grad():
            for enc, lab, *rest in train_loader:
                lookback = self.train_set.denormalize(enc).cpu().numpy()   # (B, C, L) 
                lab = lab.cpu().numpy()
                all_sample.append(np.concatenate((lookback, lab), axis=-1))

        all_sample = np.concatenate(all_sample, axis=0).astype(np.float32, copy=False)  # (N, C, T)

        N, C, T = all_sample.shape
        euclid = np.empty_like(all_sample, dtype=np.float32)
        for ch in range(C):
            euclid[:, ch, :] = to_euclidean_from_pearson(all_sample[:, ch, :], weights=self._pearson_weighted).astype(np.float32, copy=False)

        self._all_sample_data = all_sample
        self._all_sample_data_euclid = euclid

        return all_sample, euclid

    def _precompute_knn_graph(self, Kmax):
        """
        For Stage II Efficient Neighborhood Distance Calculation
        """
        _, all_euclid = self._get_all_sample_data()
        N, C, T = all_euclid.shape
        self._knn_graph = {"ind": [], "dist": []}
        for ch in tqdm(range(C), desc="Precompute KNN graph"):
            X = all_euclid[:, ch, :]
            nn = NearestNeighbors(n_neighbors=min(Kmax+1, N), 
                                  algorithm="brute", 
                                  metric="euclidean", 
                                  n_jobs=-1).fit(X)
            dist, ind = nn.kneighbors(X, return_distance=True)
            self._knn_graph["ind"].append(ind[:, 1:])
            self._knn_graph["dist"].append(dist[:, 1:])
        self._knn_graph["ind"] = [a.astype(np.int32, copy=False) for a in self._knn_graph["ind"]]
        self._knn_graph["dist"] = [a.astype(np.float32, copy=False) for a in self._knn_graph["dist"]]

    def neighborhood_distance_from_graph(self, neighbor_indexes=None):
        N = len(self.train_set)
        C = len(self._knn_graph["ind"])
        k = self.config.k_nn
        out = np.empty((N, C), dtype=np.float32)

        for ch in range(C):
            ind  = self._knn_graph["ind"][ch]    # (N, Kmax)
            dist = self._knn_graph["dist"][ch]   # (N, Kmax)

            if neighbor_indexes is None:
                # No mask: just take first k
                out[:, ch] = dist[:, :k].mean(axis=1).astype(np.float32)
                continue

            allow = np.zeros(N, dtype=bool)
            allow[np.asarray(neighbor_indexes[ch], dtype=np.int64)] = True

            # mask of allowed neighbors per row: (N, Kmax)
            mask = allow[ind]

            # Set disallowed distances to +inf, then take k-smallest per row
            allowed_dist = np.where(mask, dist, np.inf).astype(np.float32, copy=False)

            # np.partition gives k-smallest (unordered). Handles rows with <k finite by leaving infs.
            # If k > Kmax, partition would error; ensure Kmax >= k in precompute, or clamp here.
            k_eff = min(k, allowed_dist.shape[1])
            part = np.partition(allowed_dist, kth=k_eff-1, axis=1)[:, :k_eff]  # (N, k_eff)

            finite = np.isfinite(part)
            sums   = np.where(finite, part, 0.0).sum(axis=1)
            counts = finite.sum(axis=1)

            # Rule:
            # - if counts >= 1: mean over available neighbors
            # - if counts == 0: set result = inf
            mean_vals = np.full(N, np.inf, dtype=np.float32)   # (N,)
            has_neighbors = counts > 0
            mean_vals[has_neighbors] = (sums[has_neighbors] / counts[has_neighbors]).astype(np.float32)

            out[:, ch] = mean_vals

        return out

    def warm_up_backcaster(self):
        ## training backward model 
        optimizer = optim.Adam(self.backcaster.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.SmoothL1Loss(reduction='mean')
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.config.t_b):
            pbar = tqdm(train_loader, desc=f"Training Backcaster: Epoch {epoch+1}/{self.config.t_b}",  unit="batch", dynamic_ncols=True)
            self.backcaster.train()
            for batch_data in pbar:
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = None
                    y_mark = None
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data 
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)
                
                encoder_inputs_original = torch.squeeze(encoder_inputs, dim=2).float().to(self.device).permute(0, 2, 1)
                labels_original = labels.float().to(self.device).permute(0, 2, 1)
                
                encoder_inputs = labels_original.flip(dims=[1])
                labels = encoder_inputs_original.flip(dims=[1])
                encoder_inputs = self.train_set.normalize(encoder_inputs)
                labels = self.train_set.denormalize(labels)

                optimizer.zero_grad()
                if not self.use_timestamps:
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[1], 4).to(self.device)
                x_des = torch.zeros_like(labels)

                if self.config.Model.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.backcaster(encoder_inputs, x_mark, x_des, None) # x_des and y_mark are useless for AutoTimes
                else:
                    outputs = self.backcaster(encoder_inputs, x_mark, x_des, None) # x_des and y_mark are useless for AutoTimes
                    
                outputs = self.train_set.denormalize(outputs)
                loss = criterion(outputs, labels)

                pbar.set_postfix(stage_1_backcaster_loss=f'{loss.item():.3f}')
            
                loss.backward()
                optimizer.step()

        criterion = torch.nn.SmoothL1Loss(reduction='none')
        self.backcaster.eval()
        losses_record = []

        example_data_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=False)
        pbar = tqdm(example_data_loader, desc="Backcaster: Collect Loss Value", dynamic_ncols=True, leave=False)
        for batch_data in pbar:
            with torch.no_grad():
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = None
                    y_mark = None
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)
            
                encoder_inputs_original = torch.squeeze(encoder_inputs, dim=2).float().to(self.device).permute(0, 2, 1)
                labels_original = labels.float().to(self.device).permute(0, 2, 1)

                encoder_inputs = labels_original.flip(dims=[1])
                labels = encoder_inputs_original.flip(dims=[1])
                encoder_inputs = self.train_set.normalize(encoder_inputs)
                labels = self.train_set.denormalize(labels)


                if not self.use_timestamps:
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[1], 4).to(self.device)

                x_des = torch.zeros_like(labels)

                outputs = self.backcaster(encoder_inputs, x_mark, x_des, None)
                outputs = self.train_set.denormalize(outputs)
                loss = criterion(outputs, labels).mean(dim=(1,))
                losses_record.append(loss.cpu().numpy())

        losses_record = np.concatenate(losses_record, axis=0)
        return losses_record
    
    def neighborhood_distance_calculation(self, neighbor_indexes=None):
        """
        Compared each samples to the neighbors specified by indexes from the warm up backcasters and decide which samples to keep
        neighbor_indexes the neighbors are all the datasets size C X M 
        return: list of distances for all samples
        """
        _, all_euclid = self._get_all_sample_data()  # (N, C, T)
        N, C, T = all_euclid.shape
        k = self.config.k_nn

        results_all = np.empty((N, C), dtype=np.float32)

        for ch in tqdm(range(C), desc="KNN per channel (vectorized)"):
            X = all_euclid[:, ch, :] 

            if neighbor_indexes is None:
                pool_idx = None
                Y = X  # neighbors = all samples
                k_query = k + 1  # account for self-neighbor
                in_subset = None
            else:
                pool_idx = np.asarray(neighbor_indexes[ch], dtype=np.int64)
                Y = X[pool_idx]  # restricted neighbor pool
                k_query = k + 1  # we'll slice properly below
                in_subset = np.zeros(N, dtype=bool)
                in_subset[pool_idx] = True

            nbrs = NearestNeighbors(
                    n_neighbors=k_query,
                    algorithm="brute",
                    metric="euclidean",
                    n_jobs=-1).fit(Y)

            # Query all points against the chosen pool
            distances, indices = nbrs.kneighbors(X, return_distance=True)

            # If the query is also in the pool, the first neighbor is itself -> drop it.
            # Otherwise keep the first k.
            if in_subset is None:
                # neighbor pool == all -> drop self uniformly
                valid = distances[:, 1:k+1]
            else:
                # Select per-row either drop-first or not, vectorized
                cand_in  = distances[:, 1:k+1]    # for rows where in_subset==True
                cand_out = distances[:, :k]       # for rows where in_subset==False
                # Build output by stacking and choosing with mask
                valid = np.empty((N, k), dtype=np.float32)
                valid[in_subset]  = cand_in[in_subset]
                valid[~in_subset] = cand_out[~in_subset]

            results_all[:, ch] = valid.mean(axis=1, dtype=np.float32)

        return results_all
  
    def select_reliable_seed(self):
        ## training backward model 

        # warm_up_backcaster_record N x C  

        ## RCF
        warm_up_backcaster_loss_record = self.warm_up_backcaster()
        warm_up_backcaster_loss_record_sorted = np.argsort(warm_up_backcaster_loss_record, axis=0)
        backcaster_seed_id = warm_up_backcaster_loss_record_sorted[:int(warm_up_backcaster_loss_record_sorted.shape[0] * self.config.alpha)]
        
        ## NDF
        distance_records = self.neighborhood_distance_calculation(neighbor_indexes=None)
        distance_record_sorted = np.argsort(distance_records, axis=0)
        neighborhood_distance_seed_id = distance_record_sorted[-int(distance_record_sorted.shape[0] * self.config.alpha):]

        final_seed_all_ids = []
        for channel_idx in range(distance_records.shape[1]):
            backcaster_seed_channel_id = backcaster_seed_id[:, channel_idx]
            neighborhood_distance_seed_channel_id = neighborhood_distance_seed_id[:, channel_idx]
            final_seed_channel_id = list(set(backcaster_seed_channel_id) & set(neighborhood_distance_seed_channel_id))
            final_seed_all_ids.append(final_seed_channel_id)

        return final_seed_all_ids

    def train_with_weights(self, net, training_weights, 
                            learning_rate, training_epochs=1):
        # training forecaster
        train_set = TimeDatasetwithWeight(raw_data=self.train_data_seq, 
                                        mean=self.mean,
                                        std=self.std, 
                                        device=self.device,
                                        num_for_hist=self.config.seq_len,
                                        num_for_futr=self.config.pred_len,
                                        timestamps=self.train_data_stamps,
                                        weights=training_weights) 
        

        criterion = torch.nn.SmoothL1Loss(reduction='none')
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        poison_loader = DataLoader(self.poison_test_set, batch_size=self.batch_size, shuffle=False)

        for epoch in range(training_epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_epochs}" if training_epochs > 1 else None, unit="batch", dynamic_ncols=True)
            net.train()
            for batch_data in pbar:
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, weights, idx  = batch_data
                    x_mark = None
                    y_mark = None
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, weights, idx  = batch_data
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)
            
                encoder_inputs_original = torch.squeeze(encoder_inputs, dim=2).float().to(self.device).permute(0, 2, 1)
                labels_original = labels.float().to(self.device).permute(0, 2, 1)
                
                encoder_inputs = encoder_inputs_original
                labels = labels_original
                

                weights = weights.to(self.device)

                optimizer.zero_grad()

                if not self.use_timestamps:
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[1], 4).to(self.device)

                x_des = torch.zeros_like(labels)
              
                outputs = net(encoder_inputs, x_mark, x_des, None)

                outputs = self.train_set.denormalize(outputs)
                loss = criterion(outputs, labels).mean(dim=(1,))
                loss = (loss * weights).sum() / (weights.sum() + 1e-8) 
            
                pbar.set_postfix(loss=f'{loss.item():.3f}')
                
                loss.backward()
                optimizer.step()

            val_results = self.validate_step(net, epoch, test_loader, poison_loader)
        
        return val_results

    def mine_reliable_samples(self, training_weight, gamma):        
        self.forecaster.eval()

        criterion = torch.nn.SmoothL1Loss(reduction='none')
        all_train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=False)

        # Loss selection
        pbar = tqdm(all_train_loader, desc="Calculate Forecaster Loss Value", dynamic_ncols=True, leave=False)
        losses_record_f = []

        for batch_data in pbar:
            with torch.no_grad():
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = None
                    y_mark = None
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)
            
                encoder_inputs_f = torch.squeeze(encoder_inputs, dim=2).float().to(self.device).permute(0, 2, 1)
                labels_f = labels.float().to(self.device).permute(0, 2, 1)

                if not self.use_timestamps:
                    x_mark = torch.zeros(encoder_inputs_f.shape[0], encoder_inputs_f.shape[1], 4).to(self.device)

                x_des = torch.zeros_like(labels_f)

                if self.config.Model.use_amp:
                    with torch.cuda.amp.autocast():
                        output_f = self.forecaster(encoder_inputs_f, x_mark, x_des, None)
                else:
                    output_f = self.forecaster(encoder_inputs_f, x_mark, x_des, None)

                output_f = self.train_set.denormalize(output_f)
                loss_f = criterion(output_f, labels_f).mean(dim=(1,))
                losses_record_f.append(loss_f.cpu().numpy())

        losses_record_f = np.concatenate(losses_record_f, axis=0)
        losses_record = losses_record_f

        # NDF
        neighbor_seed_id = []
        for ch_idx in range(training_weight.shape[0]):
            false_indices = np.where(~training_weight[ch_idx])[0]
            neighbor_seed_id.append(false_indices)

        distance_records = self.neighborhood_distance_from_graph(neighbor_indexes=neighbor_seed_id)
        distance_record_sorted = np.argsort(distance_records, axis=0)

        neighborhood_distance_seed_id_to_include = distance_record_sorted[-int(distance_record_sorted.shape[0] * min(self.config.pi * gamma, 1.0)):]


        training_weight = np.zeros_like(training_weight, dtype=bool)
        for ch_idx in range(losses_record.shape[1]): 
            num_samples_to_mine = int(len(self.train_set) * gamma)

            loss_channel = losses_record[:, ch_idx]  # samples x 1
            neighborhood_distance_channel = neighborhood_distance_seed_id_to_include[:, ch_idx]
            loss_channel_redundancy = loss_channel[neighborhood_distance_channel]
            indices_to_mine = neighborhood_distance_channel[np.argsort(loss_channel_redundancy)[:num_samples_to_mine]]
            training_weight[ch_idx, indices_to_mine] = True  # Update the training weight to include the mined samples

        return training_weight

    def validate_step(self, net, epoch, test_loader, poison_loader):
        net.eval()
        cln_info = atk_info = ''
        with torch.no_grad():
            cln_preds = []
            atk_preds = []
            cln_targets = []
            atk_targets = []

            pbar = tqdm(test_loader, desc="Validating on clean dataset", unit="batch", dynamic_ncols=True, leave=False)
            for batch_data in pbar:
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = None
                    y_mark = None
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)

                encoder_inputs_original = torch.squeeze(encoder_inputs, dim=2).float().to(self.device).permute(0, 2, 1)
                labels_original = labels.float().to(self.device).permute(0, 2, 1)
                
                encoder_inputs = encoder_inputs_original
                labels = labels_original
            

                if not self.use_timestamps:
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[1], 4).to(self.device)
                x_des = torch.zeros_like(labels)
                outputs = net(encoder_inputs, x_mark, x_des, None).detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                outputs = self.train_set.denormalize(outputs)

                cln_targets.append(labels)
                cln_preds.append(outputs)
            
            cln_targets = np.concatenate(cln_targets, axis=0)
            cln_preds = np.concatenate(cln_preds, axis=0)
            cln_mae = mean_absolute_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1))
            cln_info = f' | clean MAE: {cln_mae}'
            pbar.close()

            pbar = tqdm(poison_loader, desc="Validating on poison dataset", unit="batch", dynamic_ncols=True, leave=False)
            for batch_data in pbar:
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = None
                    y_mark = None
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)

                encoder_inputs_original = torch.squeeze(encoder_inputs, dim=2).float().to(self.device).permute(0, 2, 1)
                labels_original = labels.float().to(self.device).permute(0, 2, 1)
                
                encoder_inputs = encoder_inputs_original
                labels = labels_original

                if not self.use_timestamps:
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[1], 4).to(self.device)
                x_des = torch.zeros_like(labels)
               
                outputs = net(encoder_inputs, x_mark, x_des, None).detach().cpu().numpy()
                
                labels = labels.detach().cpu().numpy()
                outputs = self.train_set.denormalize(outputs)

                outputs = outputs[:, :self.poison_test_set.pattern_len, self.poison_test_set.atk_vars]
                labels = labels[:, :self.poison_test_set.pattern_len, self.poison_test_set.atk_vars]

                atk_targets.append(labels)
                atk_preds.append(outputs)
            
            atk_preds = np.concatenate(atk_preds, axis=0)
            atk_targets = np.concatenate(atk_targets, axis=0)
            atk_mae = mean_absolute_error(atk_targets.reshape(-1, 1), atk_preds.reshape(-1, 1))
            atk_info = f' | attack MAE: {atk_mae}'

            pbar.close()
        
        info = 'Epoch {}'.format(epoch+1) + cln_info + atk_info
        print(info)

        return {"cln_mae": cln_mae, "atk_mae": atk_mae}

    def mitigate(self):
        start_time = time.perf_counter()

        # Implement the mitigation logic here 
        print("-"*20 + " Stage I: Time-aware Reliable Pool Initialization " + "-"*20)

        print("Pool Initialization via RCF and NDF...")
        pool_seed_ids = self.select_reliable_seed()

        training_weights = np.zeros((len(pool_seed_ids), len(self.train_set)), dtype=bool)
        for channel_idx in range(len(pool_seed_ids)):
            training_weights[channel_idx, pool_seed_ids[channel_idx]] = True

        if self.config.t_1 > 0:
            print("Warm-up Training Forecaster on Initial Reliable Pool...")
            self.train_with_weights(net=self.forecaster, 
                                    training_weights=training_weights,
                                    learning_rate=self.config.learning_rate,
                                    training_epochs=self.config.t_1)
            
        print("-"*20 + " Stage II: Distance-Regularized Loss Selection " + "-"*20)

        print("Precomputing KNN graph for efficient neighborhood distance calculation...")
        self._precompute_knn_graph(Kmax=self.config.k_nn_max)  # Precompute KNN graph for fast dynamic selection
        val_loss_min = np.Inf

        if not hasattr(self.config, "learning_rate_phase_2"):
            self.config.learning_rate_phase_2 = self.config.learning_rate # TIPS: tune this x1 or x10 learning_rate for phase 2

        print("Progressive Training with DRLS...") 
        for epoch_idx in range(self.config.t_2):
            print(f"Starting epoch {epoch_idx+1}/{self.config.t_2} of stage 2.")
            gamma = self.config.alpha + (self.config.beta - self.config.alpha) / (self.config.t_2-1) * epoch_idx
            training_weights = self.mine_reliable_samples(training_weight=training_weights, gamma=gamma)

            results = self.train_with_weights(net=self.forecaster,
                                            training_weights=training_weights, 
                                            learning_rate=self.config.learning_rate_phase_2)
                
            mae_result = results["cln_mae"]
            if mae_result < val_loss_min:
                val_loss_min = mae_result
                final_results = results
            
        print("Done.")

        end_time = time.perf_counter()
        print(f"Total defense time: {end_time - start_time:.4f} seconds")

        return final_results

if __name__ == "__main__":
    config = parser_args()

    # check if there is existing attack_save_folder folder first
    if not os.path.isdir(config.attack_save_folder):
        raise ValueError(f"Attack save folder {config.attack_save_folder} does not exist. Please check the path.")

    # Set device
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        DEVICE = torch.device('cuda')
        print("CUDA:", USE_CUDA, DEVICE, "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    else:
        DEVICE = torch.device("cpu")
        print("!!! CUDA IS NOT AVAILABLE, USING", DEVICE)
        
    config.Model.device = DEVICE
    seed_torch()
    data_config = config.Dataset
    data_config.data_filename = os.path.join(config.attack_save_folder, "poisoned_dataset.csv")
    if not data_config.use_timestamps:
        mean, std, train_data_seq, test_data_seq = load_raw_data(data_config)
        train_data_stamps, test_data_stamps = None, None
    else:
        mean, std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(data_config)
    
    # set defense variables
    defender = TimeGuardDefender(config,
                                 mean, std,
                                 train_data_seq, test_data_seq, 
                                 train_data_stamps, test_data_stamps, 
                                 DEVICE)
    seed_torch()
    final_results = defender.mitigate()
    print("Final results:", final_results)
