import time
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch

from tqdm import tqdm
from typing import Dict

from runner.trainer.trainer_utils import update_logging_dict
from utils.dist_training import gather_tensors
from utils.visual import plot_sample_qualitative_stock, plot_results_per_label_stock


def move_forward_one_epoch_mjd(model, optimizer, ema_helper, dist_helper, dataloader, logger_dict: Dict[str, dict], mode: str, dataset_nm: str, plot_save_dir=None, huber_delta=None, writer=None) -> None:
    """
    Iterate one full epoch over a dataloader.

    Works for both training and evaluation modes, updates `logger_dict` with
    per-iteration metrics, and optionally plots/records results.
    """
    assert mode in ['train', 'val', 'test']

    if dataset_nm.startswith('sp500'):
        _gnn_sp500_trainer(model, optimizer, ema_helper, dist_helper, dataloader, logger_dict, mode, dataset_nm, plot_save_dir, huber_delta, writer)
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_nm))


def _gnn_sp500_trainer(model, optimizer, ema_helper, dist_helper, dataloader, logger_dict, mode: str, dataset_nm: str, plot_save_dir=None, huber_delta=None, writer=None) -> None:
    logger_dict[mode]['time_start'] = time.time()
    flag_plot_ = (logger_dict['epoch'] == 0 and mode == 'train') or mode in ['val', 'test']
    flag_plot_qualitative = flag_plot_ == True

    device = next(model.parameters()).device
    
    static_num_to_str_dict = dataloader.dataset.int_to_ticker_dict

    for i_iter, batch_data in tqdm(enumerate(dataloader), desc='MJD trainer at mode: {}'.format(mode)):

        """Init"""
        batch_data = batch_data.to(device)

        # Input data
        dyn_feat = batch_data.dyn_feat                  # [B, 2, Past]
        static_feat = batch_data.static_feat            # [B, 1]
        edge_index = batch_data.edge_index              # [2, E], E=N*N
        edge_attr = batch_data.edge_attr                # [E, 1]
        price_past_raw = batch_data.price_past_raw      # [B, Past]

        price_norm_coef = batch_data.price_norm_coef    # [B]
        key_ego = batch_data.key_ego                    # [B]
        if isinstance(key_ego, list):
            key_ego = sum(key_ego, [])  # flatten the list

        # static feature is the factorized ticker (integer)
        ticker_factorize = static_feat[:, 0].long()

        # Target data
        price_future_norm = batch_data.price_future_norm  # [B, Future]
        price_future_raw = batch_data.price_future_raw    # [B, Future]

        # convert to dense format with mask, [TotalNodes, ...] -> [B, N, ...] or [B, N, N, ...]
        node_past_dyn_data, node_mask = to_dense_batch(dyn_feat, batch=batch_data.batch)                            # [B, N, 2, Past],  [B, N]
        node_past_static_data, _ = to_dense_batch(static_feat, batch=batch_data.batch)                              # [B, N, 1]
        adj_matrix = to_dense_adj(edge_index, batch=batch_data.batch)                                               # [B, N, N]
        edge_attr_dist = to_dense_adj(edge_index, batch=batch_data.batch, edge_attr=edge_attr)                      # [B, N, N, 1]
        node_norm_coef, _ = to_dense_batch(price_norm_coef, batch=batch_data.batch)                                 # [B, N, 1]
        node_target, _ = to_dense_batch(price_future_raw, batch=batch_data.batch)                                   # [B, N, Future]

        # build spatial position matrix
        assert set(adj_matrix.long().unique().tolist()).issubset({0, 1, 2}), "Adjacency matrix must be a subset of {0, 1, 2}"

        spatial_pos = torch.full_like(adj_matrix, -1)   # [B, N, N], range is [-1, n_neighbors - 1]
        spatial_pos[adj_matrix == 0] = 2                # no direct connection between two nodes, their path must go across the center node
        spatial_pos[adj_matrix == 1] = 1                # direct connection between two nodes
        spatial_pos[:, torch.arange(spatial_pos.size(1)), torch.arange(spatial_pos.size(1))] = 0  # self connection
        spatial_pos = spatial_pos.masked_fill(~node_mask.unsqueeze(-1), -1)  # [B, N, N]
        spatial_pos = spatial_pos.masked_fill(~node_mask.unsqueeze(-2), -1)  # [B, N, N]
        assert spatial_pos.max() <= 2, "Spatial position must be less than or equal to 2 for ego-graph"
        assert spatial_pos.min() >= -1, "Spatial position must be greater than or equal to -1"

        # build node type
        node_type = torch.full_like(node_mask.long(), -1)               # [B, N]
        node_type[node_mask] = 1                                        # 1 for valid nodes

        node_type = node_type.masked_fill(~node_mask, 0)                # 0 for padding nodes
        node_type = node_type.unsqueeze(-1)                             # [B, N, 1]
        assert set(node_type.long().unique().tolist()).issubset({0, 1, 2}), "Node type must be a subset of {0, 1, 2}"

        batched_data = {
                        'node_type': node_type.long(),                                      # [B, N, 1]
                        'in_degree': adj_matrix.sum(dim=-1).long(),                         # [B, N]              
                        'out_degree': adj_matrix.sum(dim=-2).long(),                        # [B, N]
                        'spatial_pos': spatial_pos.long(),                                  # [B, N, N]
                        'edge_attr': edge_attr_dist.float(),                                # [B, N, N, 1]
                        'adj_matrix': adj_matrix.long(),                                    # [B, N, N]
                        'node_mask': node_mask,                                             # [B, N]
                        'node_past_dyn_data': node_past_dyn_data.float(),                   # [B, N, 2, Past]
                        'node_past_static_data': node_past_static_data.float(),             # [B, N, 1]
                    }

        norm_coef = {}
        norm_coef['node_norm_coef'] = node_norm_coef
        norm_coef['data_norm'] = dataloader.dataset.data_norm
        if dataloader.dataset.data_norm == 'minmax':
            norm_coef['node_norm_min'] = to_dense_batch(batch_data.price_norm_min, batch=batch_data.batch)[0]
        batched_data.update(norm_coef)

        batched_data['huber_delta'] = huber_delta

        """Network forward pass"""
        cond_mean_loss, likelihood_loss, outputs = model(batched_data, target=node_target, flag_sample=mode != 'train')   # training loss
        node_target = node_target[node_mask]

        training_loss = cond_mean_loss + likelihood_loss
        training_loss = training_loss[node_mask]
        loss = training_loss.mean()


        """Network backward pass"""
        if mode == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)  # clip gradient
            optimizer.step()
            if ema_helper is not None:
                # we maintain a list EMA helper to handle multiple EMA coefficients
                [ema.update() for ema in ema_helper]

            output = outputs[node_mask]
        else:
            output_all, output_winner, output_prob = outputs
            output_all = output_all[node_mask]          # [Z, R, Future]
            output = output_all.mean(dim=-2)            # [Z, Future], averaged over runs
            output_winner = output_winner[node_mask]    # [Z, Future], winner-take-all selection
            output_prob = output_prob[node_mask]        # [Z, Future], probabilistic selection


        """Record training result per iteration"""
        # aggregate results on the mian GPU process
        dist_helper.ddp_sync()
        if dist_helper.is_ddp:
            output = gather_tensors(output, cat_dim=0, device=device, auto_max_shape=True).cpu()
            node_target = gather_tensors(node_target, cat_dim=0, device=device, auto_max_shape=True).cpu()
            training_loss = gather_tensors(training_loss, cat_dim=0, device=device, auto_max_shape=True).cpu()

            if mode != 'train':
                output_all = gather_tensors(output_all, cat_dim=0, device=device, auto_max_shape=True).cpu()
                output_winner = gather_tensors(output_winner, cat_dim=0, device=device, auto_max_shape=True).cpu()
                output_prob = gather_tensors(output_prob, cat_dim=0, device=device, auto_max_shape=True).cpu()
        
        # update the logging dictionary
        if mode == 'train':
            # use partial data to save memory
            slice_num_dp = 32
            slice_step_size = max(1, len(output) // slice_num_dp)
            slice_dp = slice(None, None, slice_step_size)
            update_logging_dict(logger_dict, mode, 
                                training_loss=training_loss.mean(), mae_loss=(output - node_target).abs().mean(),
                                cond_mean_loss=cond_mean_loss.mean(), likelihood_loss=likelihood_loss.mean(),
                                output=output[slice_dp], target=node_target[slice_dp], ticker_factorize=ticker_factorize[slice_dp])
        else:
            # use full data for accurate evaluation
            update_logging_dict(logger_dict, mode, 
                                training_loss=training_loss, mae_loss=(output - node_target).abs(),
                                cond_mean_loss=cond_mean_loss, likelihood_loss=likelihood_loss,
                                output=output, target=node_target, ticker_factorize=ticker_factorize,
                                output_winner=output_winner, output_prob=output_prob)
        
        """Plot the input and output sequence occasionally"""      
        if flag_plot_qualitative and plot_save_dir:
            flag_plot_qualitative = False  # only plot once
            price_future_pred = None if mode == 'train' else output.detach()  # not plot the training data model prediction

            plot_sample_qualitative_stock(price_past_raw, price_future_raw, price_future_pred,
                                          ticker_factorize, static_num_to_str_dict, key_ego,
                                          plot_save_dir, logger_dict['epoch'], tag=mode)
            
    """At the end of the epoch"""        
    if flag_plot_ and plot_save_dir:
        # plot the results by category at the end of the epoch
        all_future_price_gt = np.concatenate(logger_dict[mode]['target'], axis=0)           # [N, Future]
        all_future_price_pred = np.concatenate(logger_dict[mode]['output'], axis=0)         # [N, Future]
        all_tickers_num = np.concatenate(logger_dict[mode]['ticker_factorize'], axis=0)     # [N]
        plot_results_per_label_stock(all_future_price_gt, all_future_price_pred, all_tickers_num, static_num_to_str_dict, 
                                     plot_save_dir, logger_dict['epoch'], tag=mode)

