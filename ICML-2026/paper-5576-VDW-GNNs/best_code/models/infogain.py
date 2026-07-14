"""
Functions for 'InfoGain Wavelets', an unsupervised method for selecting 
(lazy random walk-based) graph diffusion wavelet scales.

Author: Anonymous

TODO [ideas]
    [ ] ensure no two t cutoffs are the same index? right now it
        can happen, which effectively drops a wavelet: (P^3 - P^3)x = 0,
        but this means features extracted by wavelets reflect the same
        infogain quantile, where some channels have more than that quantile's
        worth of infogain (the t diffusion steps aren't fine enough)
    [ ] outlier control in divergence calcs?
    [ ] some form of regression target 'imbalance' correction?
"""
import os
import warnings
import sys
sys.path.insert(0, '../')
from typing import Tuple, List, Dict, Optional, Iterable, Literal

import models.nn_utilities as nnu
# import pyg_utilities as pygu
from data_processing.process_pyg_data import get_P
from utilities import generate_random_integers
import matplotlib.pyplot as plt

from numpy.random import RandomState
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

DIVERGENCE_TYPES_MAP = {
    "kl": ("kl", "kld", "kl_divergence"),
    "l1": ("l1", "l1_diff", "tv", "total_variation"),
    "ot": ("ot", "wasserstein", "emd", "earth_mover")
}


def generate_scaled_index(
    batch: Batch,
    reps: int,
) -> Batch:
    """
    Adds an index attribute where each graph index is repeated
    `reps * num_nodes_in_graph` times.
    """
    # grab current device
    device = batch.batch.device
    
    # Count nodes per graph
    node_counts = torch.bincount(batch.batch)

    # Scale the repeat count
    repeat_counts = reps * node_counts  # size: [num_graphs]

    # Build repeated index vector
    new_index = torch.repeat_interleave(
        torch.arange(batch.num_graphs).to(device), 
        repeat_counts
    ) # .to(device)

    return new_index


def get_batch_idx(
    batch: Batch,
    vector_feat_dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Returns the batch index tensor for a batch, handling vector features if needed.
    Always returns a tensor, even for single-graph batches (all zeros).
    """
    if hasattr(batch, 'num_graphs') and batch.num_graphs > 1:
        if vector_feat_dim is not None:
            return generate_scaled_index(batch, vector_feat_dim)
        else:
            return batch.batch
    else:
        # Single graph: all nodes belong to graph 0
        return torch.zeros(batch.x.shape[0], dtype=torch.long, device=batch.x.device)


def compute_shortest_path_cost_matrix(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    nodes_idx: torch.Tensor,
    device: str,
):
    """
    Compute an all-pairs shortest-path cost matrix for a subgraph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index of the *full* batch graph (2, E).
    edge_weight : torch.Tensor | None
        Optional edge weights for the full graph.
    nodes_idx : torch.Tensor
        Indices of nodes that belong to the subgraph (1-D tensor).
    device : str
        Target device for the returned cost matrix.

    Returns
    -------
    torch.Tensor
        Dense [n_nodes, n_nodes] tensor of shortest-path distances on
        the chosen device.
    """
    from torch_geometric.utils import subgraph as tg_subgraph, to_scipy_sparse_matrix
    from scipy.sparse.csgraph import shortest_path as cs_shortest_path

    edge_index_sub, edge_weight_sub = tg_subgraph(
        subset=nodes_idx,
        edge_index=edge_index,
        edge_attr=edge_weight,
        relabel_nodes=True,
    )
    # Build SciPy CSR adjacency matrix (weights if provided, else 1s)
    adj = to_scipy_sparse_matrix(
        edge_index_sub,
        edge_weight_sub,
        num_nodes=nodes_idx.shape[0],
    ).astype(float)

    dist = cs_shortest_path(adj, directed=False, unweighted=edge_weight_sub is None)
    return torch.from_numpy(dist).float().to(device)


def extract_graph_Ptx_is(
    Ptxs: torch.Tensor,
    batch_idx: Optional[torch.Tensor],
    g_i: Optional[int],
    multiple_graphs: bool,
    device: str,
    verbosity: int = 0,
) -> torch.Tensor:
    """
    Extracts and moves the diffusion steps tensor Ptx_is for 
    graph `g_i` to the GPU.

    Args:
        Ptxs: Diffusion results tensor, of shape (num_diffusion_steps,
            total_num_nodes, num_channels).
        batch_idx: Tensor indicating graph membership for each node in the batch 
            (for multiple_graphs == True only).
        g_i: Index of the graph to extract (for multiple_graphs == True only).
        multiple_graphs: Whether `Ptxs` contains multiple graphs.
        device: string key for device, e.g. 'cuda'.
        verbosity: Verbosity level for printing debug info.

    Returns:
        Tensor (Ptx_is) of diffusion results for on graph, of shape
        (num_diffusion_steps, graph_num_nodes, num_channels), moved 
        to cuda device if applicable.
    """ 
    if multiple_graphs:
        x_i_mask = (batch_idx == g_i)
        Ptx_is = Ptxs[:, x_i_mask, :]
    else:
        Ptx_is = Ptxs
    Ptx_is = Ptx_is.to(device)
    if verbosity > 0:
        print('Ptx_is.shape:', Ptx_is.shape)
        print('Ptx_is.device:', Ptx_is.device)
    return Ptx_is

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def calc_infogain_wavelet_scales(
    pyg_graphs: Dict | Data, # either a dict with 'train' keying a DataLoader object, or a single Data object (for single-graph tasks)
    task: str,
    device: str,
    divergence_metric: Literal["kl", "l1", "ot", "wasserstein", "emd"] = "kl",
    divergence_metric_kwargs: Dict = {},
    diff_op_key: str = 'P',
    feat_to_diffuse_key: str = 'x',
    vector_feat_dim: Optional[int] = None,
    num_cuda_streams: int = 0,
    T: int = 16,
    cmltv_divergence_quantiles: Iterable[float] = (0.2, 0.4, 0.6, 0.8),
    start_from_t2: bool = True,
    # fixed_above_zero_floor: Optional[float] = None, # moved into divergence_metric_kwargs
    failure_value: int | float = -1,
    reweight_divs_for_target_imbal: bool = True,
    data_subsample_prop: Optional[int] = None,
    data_subsample_random_state: Optional[RandomState] = None,
    auto_process_uninformative_channels: bool = True,
    uninformative_channel_strategy: str = 'drop',
    savepath_divergence_by_channel_plot: Optional[str] = None,
    plot_dyadic_kld_curve: bool = False,
    divergence_by_channel_plot_name: str = "infogain_wavelet_scales_plot",
    divergence_by_channel_plot_label: str = "",
    verbosity: int = 0
) -> Tuple[torch.Tensor]:
    r"""
    Calculates the scales for 'custom' wavelets (non-dyadic,
    with unique scales for each channel of signal on graphs); the
    output 'channels_t_is' (calculated once over all training data) 
    can then be used in 'get_Batch_P_Wjxs'.

    Give a diffusion operator $P$ and powers $P^t$ for $t \in 0...T$,
    and a graph channel signal $x$, we take normalized $P^T x$ as our 
    reference distribution, and calculate relative entropy (KL divergence/
    information gain) of each $P^t x$ versus this reference distribution.
    We then select P-wavelet scales based on $t$ cutoffs uniquely for
    each channel, based on which powers of $t$ cross the cumulative
    KL divergence thresholds passed in the 'cmltv_kld_quantiles' argument. 
    (Here, we know that relative entropies relative to $P^T x$ decrease
    with increasing powers of $t$, so each channel has a slowing cumulative
    sum; if 'start_from_t2' is true, we also automatically keep t <= 2 as
    scale cutoffs, as the greatest values of relative entropy are expected
    in these lowest powers, and corresponding wavelets should be kept by 
    default).

    Thus, instead of dyadic-scale wavelets (where $W_j x = P^{2^{j-1}} 
    - P^{2^j}) x$), we obtain, for example, wavelets unique to each channel,
    such as (P^3 - P^5)x in one, but (P^4 - P^7)x in another, with both
    capturing the same volume change in relative entropy against their 
    channel's steady state diffusion (P^T x) at the wavelet index.

    Notes on KL divergence / relative entropy: 
        - If any entry in a channel is <=0, a NaN will be created
        by log, and that NaN is then part of a sum -> sum = NaN
        - Thus we normalize each P^t x into a probability vector
        first (i.e. with range 0-1 and sum = 1), and this prevents
        zeros, since KLD can't handle them.
        - We also prevent skewing relative KLDs by replacing zeros 
        with too tiny of a value by replacing zeros with the value
        halfway between the (pre-normalized) minimum channel value
        and second-lowest channel value
    
    Args:
        graphs: either dictionary with 'train' keying a 
            pytorch geometric DataLoader object 
            containing the test set graphs in batches 
            (multiple graphs); or single-graph pytorch
            geometric Data object with a 'train' mask.
        task: string description of the modeling task, e.g.,
            'binary_classification'.
        device: string key for the torch device, e.g. 'cuda'.
        divergence_metric: string key for the divergence metric 
            to use, e.g., 'kl' (K-L divergence); 'l1' (L1 distance); 
            'ot', 'wasserstein', 'emd' (1-Wasserstein, aka Earth Mover's
            Distance).
        divergence_metric_kwargs: optional dictionary of kwargs for the
            divergence metric.
            - 'ot': cost_matrix (scipy sparse matrix)
            - 'wasserstein': cost_matrix (scipy sparse matrix)
            - 'emd': cost_matrix (scipy sparse matrix)
        diff_op_key: if P_sparse (diffusion operator matrix
            has already been calculated and attached to batch,
            pass the string key for the attribute here.
        feat_to_diffuse_key: the Batch attribute string key
            for the feature to diffuse (defaults to 'x').
        vector_feat_dim: if a vector feature is being diffused,
            set its dimension (e.g., 3 for an x, y, z coordinate 
            feature) here; else leave as None for scalar features.
        use_cuda_streams: whether to use CUDA streams 
            in processing (if using CUDA).
        T: max power of $P$, for $P^t$ where $t \in 1...T$.
        cmltv_divergence_quantiles: iterable of cumulative KLD 
            quantiles/percentile cutoffs, which powers of P
            must reach to be a wavelet scale boundary $P^t$.
        start_from_t2: boolean whether to keep filters with 
            $P^1$ and $P^2$, and choose subsequent scales
            (and ignore their contribution to cumulative KLD;
            calc from $P^3...P^T$ instead). This is useful since
            these lowest powers of $t$ generally cover the largest
            steps in KLD, and perhaps should be included scale
            steps in all channels by default.
        fixed_above_zero_floor: [deprecated] optional fixed float value to
            replace zeros in original features with. If None, the linear
            midpoint between 0 and the next lowest value is used. Note
            zeros must be replaced since KLD uses logarithms. Provide via
            `divergence_metric_kwargs['above_zero_floor']`.
        failure_value: value to return in final indices tensor
            in place of NaN, etc., in case computation fails (e.g.
            in the case of features/channels without sufficient
            information change over diffusion steps.
        reweight_divs_for_target_imbal: boolean whether to re-weight 
            each graph's contribution to a channels' total (sum) KLD 
            loss, e.g. to rebalance KLD for unbalanced target classes.
        data_subsample_prop: if not None, a random sample of size
            int(data_subsample_prop * num_graphs) will be taken of 
            the graphs (in a multi-graph training dataset) and 
            used to calculate the wavelet scales, instead of the 
            full train set.
        data_subsample_random_state: optional np.RandomState generating
            data_subsample_n when fitting infogain on a subset of the 
            training data.
        auto_process_uninformative_channels: whether to automatically
            process uninformative channels by the strategy in 
            'uninformative_channel_strategy'.
        uninformative_channel_strategy: 'drop' (to remove channels) 
            or 'average' to replace channels with the median scales from
            informative channels.
        savepath_divergence_by_channel_plot: optional save path for a plot
            of cumulative KLDs by channel. Set to None to skip creation
            of the plot.
        divergence_by_channel_plot_name: filename ('.png' added automatically)
            for the optional KLD by channel plot.
        plot_dyadic_kld_curve: whether to plot a line on the 'KLD by channel
            plot' showing the dyadic scale KLD curve. [CURRENTLY BROKEN.]
        divergence_by_channel_plot_label: string label for the plot
        verbosity: integer controlling volume of print output as
            the function runs.
    Returns:
        2-tuple of torch tensors: (1) optional tensor of the indices of 
        uninformative channels found (if autoprocessing them here); and
        (2) tensor containing indices of wavelet scale $t$s (which also 
        happen to be their values in $P^t x, t \in 0...T$) for each channel 
        in the graph dataset; shape (n_channels, n_ts).
    """
    Ptx_i_start = 2 if start_from_t2 else 0

    # get DataLoader or single Data object
    if isinstance(pyg_graphs, dict):
        graphs = pyg_graphs['train']
        multiple_graphs = True
    elif isinstance(pyg_graphs, DataLoader):
        graphs = pyg_graphs
        multiple_graphs = True
    elif isinstance(pyg_graphs, Data):
        # make iterable of 1 batch of 1 graph
        graphs = (pyg_graphs, )
        multiple_graphs = False
    
    # -------------------------------------------------------------------------
    # Loop over batches in train set
    # -------------------------------------------------------------------------
    klds_by_x_t_chan = []
    targets_by_xi = []
    for batch_i, batch in enumerate(graphs):
        
        # Get batch attributes
        batch = batch.to(device)
        x = batch[feat_to_diffuse_key]
        # If vector features are provided as (N, d), flatten to (N*d, 1)
        if (vector_feat_dim is not None) and (x.ndim == 2) and (x.shape[1] == vector_feat_dim):
            x = x.reshape(-1, 1)
        # Ensure 2-D: (N, C)
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)
        print(f"\tbatch {batch_i + 1} (shape={x.shape})")
        n_channels = x.shape[1]
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight \
            if hasattr(batch, 'edge_weight') \
            else None
        
        # Get number of graphs in batch
        if multiple_graphs:
            num_graphs = batch.num_graphs
            # use 'extend' since we are populating by batch
            targets_by_xi.extend(batch.y)
            # batch_index = batch.batch
        else: # 1 graph in single Data object
            # note that in a node-level graph task,
            # we don't mask any node signals for train vs. valid
            # set until loss calculation / evaluation time
            num_graphs = 1
            # use 'extend' since we are populating by batch
            targets_by_xi.extend(batch.y[batch.train_mask])

        if (num_graphs > 1) and (data_subsample_prop is not None):
            graphs_loop_idx = generate_random_integers(
                n=int(data_subsample_prop * num_graphs),
                max_val=num_graphs,
                random_state=data_subsample_random_state
            )
        else:
            graphs_loop_idx = range(num_graphs)

        # grab or create batch indexing for scalar/vector feature
        batch_idx = get_batch_idx(batch, vector_feat_dim)
        
        # get P_sparse: calc if it is not already an attribute of the
        # batched graphs Batch object
        if diff_op_key is None:
            P_sparse = get_P(
                data=batch,
                lazy=True,
                device=device
            )
        else:
            P_sparse = batch[diff_op_key]

        # ---------------------------------------------------------------------
        # Diffusion
        # ---------------------------------------------------------------------
        # calc P^t x for t \in 1...T
        # make each Ptx in list dense here so tensor slicing below works
        # Ensure dense tensor for sparse.mm; keep a dense working copy
        x_dense = x.to_dense() if getattr(x, 'is_sparse', False) else x
        Ptx = x_dense
        Ptxs = [None] * (T + 1)
        Ptxs[0] = x_dense
        for j in range(1, T + 1):
            Ptx = torch.sparse.mm(P_sparse, Ptx)
            Ptxs[j] = Ptx.to_dense()
        Ptxs = torch.stack(Ptxs, dim=0) # shape (T, N, C)
        
        # --- Option 1: Parallelize graphs across cuda streams ---
        if (num_cuda_streams > 0) and ('cuda' in device):
            num_streams = min(num_cuda_streams, len(graphs_loop_idx))
            streams = [torch.cuda.Stream() for _ in range(num_streams)]
            
            # initialize correct-size list; stream results must be inserted
            # at the right (graph_i) index since they could append out of order
            streams_channel_klds = [None] * num_graphs
            
            for g_i in graphs_loop_idx:
                stream = streams[g_i % num_streams]
                    
                with torch.cuda.stream(stream):
                    Ptx_is = extract_graph_Ptx_is(
                        Ptxs=Ptxs,
                        batch_idx=batch_idx,
                        g_i=g_i,
                        multiple_graphs=(num_graphs > 1),
                        device=device,
                        verbosity=verbosity,
                    )  # shape (T, n_i, C)

                    # Build divergence-specific kwargs (e.g., cost matrix for OT)
                    divergence_metric_kwargs_graph = dict(divergence_metric_kwargs)
                    if divergence_metric.lower() in DIVERGENCE_TYPES_MAP["ot"]:
                        nodes_idx = torch.where(batch_idx == g_i)[0]
                        cost_matrix = compute_shortest_path_cost_matrix(
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                            nodes_idx=nodes_idx,
                            device=device,
                        )
                        divergence_metric_kwargs_graph["cost_matrix"] = cost_matrix

                    channel_klds = get_divergence_by_channel(
                        Ptx_is,
                        Ptx_i_start,
                        vector_feat_dim,
                        divergence_metric,
                        divergence_metric_kwargs_graph,
                    )  # shape (T - 1 - Ptx_i_start, C)
                    
                    streams_channel_klds[g_i] = channel_klds
            
            # after all streams have finished, sync
            torch.cuda.synchronize()

            # ...append 'kld_all_ts_by_chan' for each graph in each batch to 'klds_by_x_t_chan',
            # a growing list of eventual length n_graphs, where each list element is a tensor 
            # of shape (num_t_powers, n_channels)'
            klds_by_x_t_chan.extend(streams_channel_klds)
            
        # --- Option 2: Process graphs sequentially ---
        else: # cpu or 1 cuda stream
            for g_i in graphs_loop_idx:

                Ptx_is = extract_graph_Ptx_is(
                    Ptxs=Ptxs,
                    batch_idx=batch_idx,
                    g_i=g_i,
                    multiple_graphs=(num_graphs > 1),
                    device=device,
                    verbosity=verbosity,
                )

                # Build divergence-specific kwargs (e.g., cost matrix for OT)
                divergence_metric_kwargs_graph = dict(divergence_metric_kwargs)

                if divergence_metric.lower() in DIVERGENCE_TYPES_MAP["ot"]:
                    nodes_idx = torch.where(batch_idx == g_i)[0]
                    divergence_metric_kwargs_graph["cost_matrix"] = compute_shortest_path_cost_matrix(
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        nodes_idx=nodes_idx,
                        device=device,
                    )

                channel_klds = get_divergence_by_channel(
                    Ptx_is,
                    Ptx_i_start,
                    vector_feat_dim,
                    divergence_metric,
                    divergence_metric_kwargs_graph,
                )
                klds_by_x_t_chan.append(channel_klds)

    # -------------------------------------------------------------------------
    # Collect divergence values
    # -------------------------------------------------------------------------
    # after all graphs in all batches:
    # stack KLD values for all graphs x t powers x channels into a tensor
    klds_by_x_t_chan = torch.stack(klds_by_x_t_chan) # shape (n_graphs, num_t_powers, n_channels)
    if verbosity > 0:
        print('klds_by_x_t_chan.shape:', klds_by_x_t_chan.shape)
    targets_by_xi = torch.stack(targets_by_xi) # shape (n_graphs, )

    # quantify relative KLD over all graphs, by t and channel
    '''
    TODO
    - if taking sum over all xs, beware of outliers contributing
        a huge amount of KLD: only consider 10th-90th percentiles?
    - define regression target imbalance and reweight for that?
    '''
    # -------------------------------------------------------------------------
    # [Optional] Re-weight divergences for target class balance
    # -------------------------------------------------------------------------
    if reweight_divs_for_target_imbal:
        # 0s get weight 1, 1s get relative weight 'pos_class_wt'
        if 'bin' in task.lower() and 'class' in task.lower():
            ct_1s = targets_by_xi.sum()
            pos_class_wt = (targets_by_xi.shape[0] - ct_1s) / ct_1s
            div_weights = torch.ones(len(targets_by_xi))
            div_weights[targets_by_xi == 1] = pos_class_wt
        else:
            # raise NotImplementedError()
            warnings.warn(f"Reweighting KLDs not implemented for task='{task}'.")
            div_weights = None
    
        # reweight KLDs
        if div_weights is not None:
            klds_by_x_t_chan = torch.einsum(
                'bTc,b->bTc',
                klds_by_x_t_chan,
                div_weights
            )
        
    # -------------------------------------------------------------------------
    # Sum divergences across graphs, by t and channel
    # -------------------------------------------------------------------------
    # sum (reweighted) KLD across graphs, by t and channel
    divs_by_t_chan = torch.sum(klds_by_x_t_chan, dim=0) # shape (T - Ptx_i_start, n_channels)
    
    # get cumulative sums as t increases
    cmltv_divs_by_t_chan = torch.cumsum(divs_by_t_chan, dim=0) # shape (T - Ptx_i_start, n_channels)

    # if there's only one channel, we likely have a 1-d tensor (but need 2-d)
    if cmltv_divs_by_t_chan.ndim == 1:
        cmltv_divs_by_t_chan = cmltv_divs_by_t_chan.unsqueeze(dim=-1)
    
    # minmax scale channel cumulative KLDs, for cross-channel comparison (i.e.
    # so all channels' cumulative KLDs range from 0 to 1)
    for c in range(n_channels):
        # klds_cum_by_t_chan[:, c] = (klds_cum_by_t_chan[:, c] - min_kld) / (max_kld - min_kld)
        chan_rescaled_cum_kld = nnu.minmax_scale_tensor(
            v=cmltv_divs_by_t_chan[:, c], 
            min_v=cmltv_divs_by_t_chan[:, c][0], # min cmltv kld is at start
            max_v=cmltv_divs_by_t_chan[:, c][-1] # max cmltv kld is at end
        )   
        # if min-max scaling doesn't work (likely no variance in channel), None is returned
        # for the 'chan_rescaled_cum_kld' vector -> insert tensor of -1s instead into
        # 'klds_cum_by_t_chan', this will lead to 'failure_value' in 'channels_t_is' below
        if chan_rescaled_cum_kld is None:
            cmltv_divs_by_t_chan[:, c] = -torch.ones(cmltv_divs_by_t_chan.shape[0])
        else:
            cmltv_divs_by_t_chan[:, c] = chan_rescaled_cum_kld

    # -------------------------------------------------------------------------
    # [Optional] Plot divergence curves
    # -------------------------------------------------------------------------
    if savepath_divergence_by_channel_plot is not None:
        plot_divergence_curves(
            cmltv_divs_by_t_chan,
            Ptx_i_start,
            T,
            savepath_divergence_by_channel_plot,
            plot_name=divergence_by_channel_plot_name,
            plot_dyadic_kld_curve=plot_dyadic_kld_curve,
            title=divergence_by_channel_plot_label
        )

    # -------------------------------------------------------------------------
    # Find wavelet scales
    # -------------------------------------------------------------------------
    # for each channel, find (indexes of) t-integer scale cutoffs, 
    # following the quantiles of cmltv KLD in 'cmltv_kld_quantiles'
    channels_t_is = torch.stack([
        torch.stack([
            # adjust t indexes returned for the 'Ptx_i_start' index
            # also make sure an index is found; else return failure value
            (torch.tensor((y[0, 0] if y.ndim > 1 else y[0]).item() + Ptx_i_start, device=device) \
             if ((y := torch.argwhere(cmltv_divs_by_t_chan[:, c] >= q)).numel() > 0) \
             else torch.tensor(failure_value, device=device)) \
            for q in cmltv_divergence_quantiles
        ]) \
        for c in range(n_channels)
    ]) # shape (n_channels, n_quantiles)

    # calc wavelet filters, by (P^t - P^u)x = (P^t)x - (P^u)x
    # uniquely for each channel, following 'channels_t_is'
    # (all P^ts and P^us calc'd above for all xs and channels:
    # just need to subtract using specific ts and us for each channel)
    if start_from_t2:
        # Build fixed prefix (t = 0, 1, 2) and suffix (t = T) for every channel
        prefix = torch.arange(3, device=device).unsqueeze(0).repeat(n_channels, 1)  # (n_channels, 3)
        suffix = torch.full((n_channels, 1), T, device=device)  # (n_channels, 1)

        # In rare cases (e.g., vector track with a single informative channel),
        # channels_t_is may unintentionally have an extra singleton dimension
        # (shape = [n_channels, n_ts, 1]).  Squeeze/flatten to ensure 2-D.
        if channels_t_is.ndim > 2:
            channels_t_is = channels_t_is.view(channels_t_is.shape[0], -1)

        # Concatenate along the feature dimension
        channels_t_is = torch.cat((prefix, channels_t_is.to(device), suffix), dim=1).long()

        # OLD logic
        # channels_t_is = torch.concatenate((
        #     torch.stack((
        #         torch.zeros(n_channels), 
        #         torch.ones(n_channels), 
        #         torch.ones(n_channels) * 2
        #     ), dim=-1).to(device),
        #     channels_t_is,
        #     (torch.ones(n_channels) * T).unsqueeze(dim=1).to(device)
        # ), dim=-1).to(torch.long)

    if verbosity > 0:
        print('channels_t_is.shape:', channels_t_is.shape) # shape (n_channels, n_quantiles)
        print('channels_t_is\n', channels_t_is)

    # -------------------------------------------------------------------------
    # [Optional] Process uninformative channels
    # -------------------------------------------------------------------------
    # find rows where channels_t_is contains failure_value
    mask = (channels_t_is == failure_value)        
    failure_row_indices = torch.unique(torch.argwhere(mask)[:, 0])

    # if failures are found without autoprocess strategy, raise warning
    if (not auto_process_uninformative_channels) \
    and (len(failure_row_indices) > 0):
        fail_idx_l = [i.item() for i in failure_row_indices]
        warnings.warn(
            f"The channels/features at indexes {fail_idx_l}"
            f" failed to generate valid diffusion scales. Consider"
            f" dropping these features?"
        )

    # check if all channels' scales (rows) are the same
    # if so, return 1-d tensor (more efficient wavelet filtrations
    # can be done during training)
    # note that if this is infogain for vector features, we only have 1 channel
    if (vector_feat_dim is None) \
    and (channels_t_is[0] == channels_t_is).all():
        warnings.warn(
            f"All channels' custom scales were found to be equal:"
            f" returning 1-d tensor of scales instead."
        )
        channels_t_is = channels_t_is[0]

    # Return indices of uninformative channels, and wavelet scales
    if auto_process_uninformative_channels:
        uninform_chan_is, channels_t_is = process_uninformative_channels(
            channels_t_is=channels_t_is,
            strategy=uninformative_channel_strategy,
            T=T
        )
        return uninform_chan_is, channels_t_is
    else:
        return None, channels_t_is


def process_uninformative_channels(
    channels_t_is: torch.Tensor,
    strategy: str = "drop",
    T: int = 32
) -> Tuple[torch.Tensor]:
    """
    If uninformative channels are found, rows in
    'channels_t_is' will have -1 values. This method
    processes 'channels_t_is' according to the
    desired strategy. 
    
    Args:
        channels_t_is: tensor of integer indices
            for wavelet scale boundaries. Shape:
            (n_channels, n_ts).
        strategy: how the uninformative channels are 
            to be handled: 'drop' (remove them), 
            'average' (replace with the median scales
            from informative channels), or 'dyadic'
            (replace with zero-padded dyadic scales).
        T: max diffusion step (e.g. 16 or 32).
    Returns:
        2-tuple of tensors: (1) indices of uninformative
        channels found; (2) wavelet scale t cutoffs by
        channel with uninformative channels removed.
    """
    # find where the tensor equals -1
    mask = (channels_t_is == -1)
    # use argwhere to get row indices where any column contains -1
    uninform_channels_is = torch.unique(torch.argwhere(mask)[:, 0])

    if strategy == 'drop':
        incl_mask = torch.ones(len(channels_t_is), dtype=torch.bool)
        incl_mask[uninform_channels_is] = False
        channels_t_is = channels_t_is[incl_mask]
        print(
            f"\tDropped {len(uninform_channels_is)} uninformative channels:"
            f" {uninform_channels_is}"
        )
    elif strategy == 'average' or strategy == 'avg':
        median_scales = get_avg_infogain_wavelet_scales(channels_t_is)
        channels_t_is[uninform_channels_is] = median_scales
    elif strategy == 'dyadic':
        T_to_J_lookup = {8: 3, 16: 4, 32: 5, 64: 6} # ugly patch
        n_ts = channels_t_is.shape[1]
        dyadic_scales = torch.cat((
            torch.tensor([0]),
            2 ** torch.arange(T_to_J_lookup[T] + 1)
        ))
        if n_ts > len(dyadic_scales):
            # left-pad with 0s if there are more ts than in a true 
            # dyadic scale
            pad_zeros = torch.zeros(n_ts - len(dyadic_scales))
            dyadic_scales = torch.cat((pad_zeros, dyadic_scales))
            channels_t_is[uninform_channels_is] = dyadic_scales
        
    return uninform_channels_is, channels_t_is
    

def get_avg_infogain_wavelet_scales(
    channels_t_is: torch.Tensor,
    average_method: str = 'median'
) -> torch.Tensor:
    r"""
    Averages each custom P-wavelet scale indices across
    channels. Useful for MFCN networks with more than one
    filter cycle; can use the average custom scales found
    here for all new (recombined feature) channels, instead
    of recomputing custom scales in each second or further
    cycle of each training epoch (as the model learns new
    channel-filter combinations).

    Args:
        channels_t_is: Torch tensor containing indices of 
            $t$s (which also happen to be their values in
            $P^t x, t \in 0...T$) for each channel in the
            graph dataset; shape (n_channels, n_ts).
        average_method: string key for the averaging method 
            used, e.g. 'median' (which uses the integer floor
            in the case of #.5s).
    Returns:
        Tensor of channel's averaged $t$ indices; shape
        (n_ts, ).
    """
    if average_method == 'median':
        return torch.median(channels_t_is, dim=0).values.to(torch.long)
    else:
        raise NotImplementedError(
            f"Averaging method '{average_method}' not"
            f" implemented."
        )


def get_floor(
    tensor: torch.Tensor, 
    above_zero_floor: Optional[float] = 1e-2
) -> torch.Tensor | float:
    """
    Utility function to replace zeros in a
    diffusion results tensor with another
    minimum value, `above_zero_floor`.
    """
    if above_zero_floor is not None:
        floor = above_zero_floor
    else:
        floor = nnu.get_mid_btw_min_and_2nd_low_vector_vals(tensor)
    if (floor is not None) \
    and (not isinstance(floor, str)) \
    and (floor <= 0):
        floor = -floor
    return floor


def normalize_channels(
    x: torch.Tensor,
    above_zero_floor: Optional[float] = None
) -> torch.Tensor:
    """
    Apply per-channel normalization using nnu.norm_1d_tensor_to_prob_mass.
    Supports vectorized input: x shape (T, N, C) or (N, C).
    """
    if x.ndim == 3:
        # x shape: (T, N, C)
        T, N, C = x.shape
        normed = torch.empty_like(x)
        for c in range(C):
            for t in range(T):
                chan = x[t, :, c]
                floor = get_floor(chan, above_zero_floor)
                normed[t, :, c] = nnu.norm_tensor_to_prob_mass(chan, above_zero_floor=floor)
        return normed
    elif x.ndim == 2:
        # x shape: (N, C)
        N, C = x.shape
        normed_channels = [
            nnu.norm_tensor_to_prob_mass(x[:, c], above_zero_floor=get_floor(x[:, c], above_zero_floor))
            for c in range(C)
        ]
        return torch.stack(normed_channels, dim=1)
    else:
        raise ValueError(f"normalize_channels expects 2D or 3D tensor, got shape {x.shape}")


def get_divergence_by_channel(
    Ptx_is: torch.Tensor,
    Ptx_i_start: int = 2,
    vector_feat_dim: Optional[int] = None,
    divergence_metric: str = "kl",
    divergence_metric_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """
    For one graph, compute KL divergence values for each signal
    channel, of each diffusion step against the final diffusion
    step calculated (at time T), starting with diffusion step
    `Ptx_i_start`.

    PyTorch KL divergence ref:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html

    Args:
        Ptx_is: one graph's diffusion results tensor, of 
            shape (num_diffusion_steps, num_nodes, num_channels).
        Ptx_i_start: the first diffusion step to calculate 
            (earlier steps may be skipped if included as scales
            automatically).
        above_zero_floor: zero-replacement value used only when
            `divergence_metric=='kl'`.  Provide via
            `divergence_metric_kwargs['above_zero_floor']`.
        vector_feat_dim:
    Returns:
        Tensor of KL divergence values for one graph, of shape
        (T - 1 - Ptx_i_start, num_channels).
    """
    if divergence_metric_kwargs is None:
        divergence_metric_kwargs = {}
    above_zero_floor = divergence_metric_kwargs.get("above_zero_floor")

    T, N, C = Ptx_is.shape
    PTx = Ptx_is[-1]  # shape: (N, C)
    # Normalize final and intermediate diffusion results into probability vectors
    PTx_normed = normalize_channels(PTx, above_zero_floor)  # (N, C)
    Ptx_t_normed = normalize_channels(
        Ptx_is[Ptx_i_start : T - 1], above_zero_floor
    )  # (T-1-Ptx_i_start, N, C)

    # Expand the final step distribution to match intermediate steps
    PTx_normed_exp = PTx_normed.unsqueeze(0).expand(Ptx_t_normed.shape)

    divergence_metric = divergence_metric.lower()
    if divergence_metric in DIVERGENCE_TYPES_MAP["kl"]:
        # KL divergence: input is log-probabilities, target is probabilities
        vals = F.kl_div(
            input=torch.log(Ptx_t_normed + 1e-12),  # avoid log(0)
            target=PTx_normed_exp,
            reduction="none",
            log_target=False,
        )  # (T-1-Ptx_i_start, N, C)
    elif divergence_metric in DIVERGENCE_TYPES_MAP["l1"]:
        # L1 distance between distributions (optionally scaled by 0.5 for TV; omit for now)
        vals = torch.abs(Ptx_t_normed - PTx_normed_exp)  # (T-1-Ptx_i_start, N, C)
    elif divergence_metric in DIVERGENCE_TYPES_MAP["ot"]:
        cost_matrix = divergence_metric_kwargs.get("cost_matrix")
        if cost_matrix is None:
            raise ValueError(
                "For 'ot' divergence_metric, a 'cost_matrix' key must be provided in divergence_metric_kwargs."
            )
        import ot  # type: ignore
        cost_np = cost_matrix.detach().cpu().numpy()
        vals_rows = []
        for t_idx in range(Ptx_t_normed.shape[0]):
            row_vals = []
            for c_idx in range(C):
                a = Ptx_t_normed[t_idx, :, c_idx].detach().cpu().numpy()
                b = PTx_normed[:, c_idx].detach().cpu().numpy()
                emd_val = ot.emd2(a, b, cost_np)
                row_vals.append(emd_val)
            vals_rows.append(row_vals)
        vals = torch.tensor(vals_rows, dtype=PTx_normed.dtype, device=PTx_normed.device)  # (T-1-Ptx_i_start, C)
    else:
        raise ValueError(
            f"Unsupported divergence_metric '{divergence_metric}'. Choose 'kl', 'l1', or 'ot'."
        )

    # Reduce over nodes to obtain per-diffusion-step per-channel values
    if divergence_metric in ("ot", "wasserstein", "emd", "earth_mover"):
        # Already aggregated over nodes; vals shape = (T-1-Ptx_i_start, C)
        channel_vals = vals
    else:
        if vector_feat_dim is not None:
            channel_vals = vals.sum(dim=1, keepdim=True)  # shape (T-1-Ptx_i_start, 1)
        else:
            channel_vals = vals.sum(dim=1)  # shape (T-1-Ptx_i_start, C)

    return channel_vals


def plot_divergence_curves(
    cmltv_div_by_t_chan: torch.Tensor,
    Ptx_i_start: int,
    T: int,
    savepath: str,
    plot_name: str = "infogain_wavelet_scales_plot",
    plot_dyadic_kld_curve: bool = False,
    operator_str: str = "P",
    title: str = ""
) -> None:
    """
    Saves a plot of cumulative divergence values by channel.
    """
    from numpy import linspace, log2
    os.makedirs(savepath, exist_ok=True)
    for c in range(cmltv_div_by_t_chan.shape[1]):
        chan_vals = cmltv_div_by_t_chan[:, c].cpu().numpy()
        if chan_vals[0] > -1:  # don't plot uninformative channels
            plt.plot(range(Ptx_i_start, T), chan_vals)
    if plot_dyadic_kld_curve and ((T == 16) or (T == 32)):
        pass  # (left as before)
    title_prefix = f"{title} - " if title else ""
    plt.title(
        f"{title_prefix}Normalized cumulative (all-node) sums of divergences of ${operator_str}^t x$"
        f"\nfor $t \in ({Ptx_i_start},\ldots,(T-1))$ from ${operator_str}^T x$, by channel"
    )
    plt.xlabel('$t$ in ${operator_str}^t x$')
    plt.xticks([0] + [2 ** p for p in range(0, int(log2(T)) + 1)])
    plt.yticks(linspace(0, 1, 11))
    plt.ylabel('cumulative divergence from ${operator_str}^T x$')
    plt.grid()
    plt.savefig(f"{savepath}/{plot_name}.png")
    plt.clf()


def process_custom_wavelet_scales_type(scales):
    """
    Convert various user-provided formats to a tensor or None.

    Args:
        scales: 'dyadic', list[int], list[list[int]], or torch.Tensor.
    Returns:
        torch.LongTensor or None (for 'dyadic').
    """
    if isinstance(scales, str):
        # The default string ('dyadic') signals to use dyadic scales.
        return None
    if isinstance(scales, torch.Tensor):
        return scales.long()
    if isinstance(scales, list):
        try:
            return torch.tensor(scales, dtype=torch.long)
        except Exception:
            # Fallback: let torch infer dtype (handles nested list lengths)
            return torch.tensor(scales)
    # Unsupported type
    else:
        raise ValueError(f"process_wavelet_scales_type: Unsupported scale type: {type(scales)}")


def get_wavelet_count_from_scales(
    scales: List[int] | torch.Tensor,
    include_lowpass: bool = True,
) -> int:
    """
    Get the number of wavelets from a list/tensor of scales.
    """
    # Ensure scales is a tensor
    t = process_custom_wavelet_scales_type(scales)
    if t.ndim == 1:
        ct = int(t.numel()) - 1
    else:
        ct = int(t.shape[0]) - 1
    if include_lowpass:
        ct += 1
    return ct