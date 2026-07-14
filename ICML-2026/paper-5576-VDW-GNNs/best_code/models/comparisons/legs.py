"""
LEGS module and accessory methods
Original author: Alex Tong
Edited by: Anonymous

DJ key changes:
1. Generalized to to use 32 or any (dyadic) number of diffusion steps.
2. Made final 1st- and 2nd-order scattering features optionally absolute value.
(This loses information where -x is conflated with x).
3. Swapped normalized/statistical moments for unnormalized moments in pooling
step. Added further pooling options (mean, max).
4. Consolidated tensor multiplications and reshaping into torch.einsum operations;
replaced list appends with insertions; replaced tensor concatenations in new
dimension with torch.stack operations.
5. Added method to save state of best selector matrix tensor, for inspection
after model training.

TO DO
[ ] implement sparse P in recursive P^t x diffusion calculations?

LEGS reference paper: "Data-Driven Learning of Geometric Scattering Networks"
IEEE Machine Learning for Signal Processing Workshop 2021

Original LEGS code repo:
https://github.com/KrishnaswamyLab/LearnableScattering/blob/main/models/LEGS_module.py
"""
import numpy as np
import torch
# from torch.nn import Linear
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
# from torch_geometric.utils import degree
from torch_geometric.transforms import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add # scatter_mean
try:
    from torch_sparse import matmul
except Exception:
    matmul = None  # type: ignore

from typing import List, Optional, Literal, Union
from pyg_utilities import moments_channel_pool


class LEGSModel(torch.nn.Module):
    """
    Implementation of the LEGS Model.
    """
    def __init__(
        self,
        in_channels: int,
        output_dim: int = 1,
        *,
        feature_attr: str = 'x',
        J: int = 4,
        n_moments: Optional[int] = 4,
        apply_modulus_to_scatter: bool = True, # AT's default
        trainable_laziness: bool = False,
        save_best_selector_matrix: bool = True,
        selector_matrix_save_path: Optional[str] = None,
        verbosity: int = 0,
        # Comparison-model extensions
        pool_types: Union[
            Literal['max', 'mean', 'sum', 'moments'],
            List[Literal['max', 'mean', 'sum', 'moments']]
        ] = 'sum',
        predict_per_node: bool = False,
        mlp_head_hidden_dims: Optional[List[int]] = [128, 64, 32, 16],
        node_mlp_dim: int = 128,
        activation: Literal['relu', 'swish'] = 'relu',
        mlp_dropout_p: float = 0.0,
    ):
        super().__init__()
        # Comparison-model flags
        self.predict_per_node = predict_per_node
        self.out_dim = int(output_dim)
        self.feature_attr = str(feature_attr)
        self.save_best_selector_matrix = save_best_selector_matrix
        # self.selector_matrix_save_path = selector_matrix_save_path
        self.verbosity = verbosity
        self.in_channels = int(in_channels)
        self.mlp_head_hidden_dims = mlp_head_hidden_dims
        self.activation_name = str(activation).lower()
        self.mlp_dropout_p = mlp_dropout_p
        """
        Pooling configuration: allow single or multiple pool types and
        compute the resulting readout input multiplier.
        """
        self.n_moments = n_moments
        # Normalize pool_types to a list of lowercase strings
        if isinstance(pool_types, (list, tuple)):
            self.pool_types_list = [str(pt).lower() for pt in pool_types]
        else:
            self.pool_types_list = [str(pool_types).lower()]
        # Validate pool types
        _allowed_pools = {'sum', 'mean', 'max', 'moments'}
        for pt in self.pool_types_list:
            if pt not in _allowed_pools:
                raise ValueError(
                    f"Unsupported pool_types '{pool_types}'. Use elements from ['sum','mean','max','moments']."
                )
        # If moments is requested, ensure n_moments is provided
        if ('moments' in self.pool_types_list) and (self.n_moments is None):
            raise ValueError("n_moments must be provided when 'moments' is in pool_types")
        # Multiplier for concatenated pooled representations at readout
        num_basic = sum(pt in {'sum', 'mean', 'max'} for pt in self.pool_types_list)
        num_mom = (self.n_moments if ('moments' in self.pool_types_list) else 0)
        self.readout_multiplier = int(num_basic + num_mom)
        # For backward compatibility keep a primary pool_type (first entry)
        self.pool_type = self.pool_types_list[0]
        self.J = J
        self.apply_modulus_to_scatter = apply_modulus_to_scatter
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = Diffuse(
            in_channels,
            in_channels,
            trainable_laziness
        )
        self.diffusion_layer2 = Diffuse(
            in_channels, 
            in_channels, 
            trainable_laziness
        )
        self.wavelet_constructor = torch.nn.Parameter(
            self._get_dyadic_scales_matrix_tensor(self.J),    
        )
        self.n_wavelets = self.wavelet_constructor.shape[0] # 4 / 5
        self.n_diffusion_steps = self.wavelet_constructor.shape[1] # 17 / 33
        self.feng_filters_2nd_order = self._feng_filters_2nd_order()
        # Pooling (graph-level) configuration
        self._pool_map = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }
        # For single-pool fast-path convenience
        if (len(self.pool_types_list) == 1) and (self.pool_type in self._pool_map):
            self.pool = self._pool_map[self.pool_type]
        else:
            self.pool = None

        # Heads: graph-level MLP and node-level 2-layer MLP
        flat_node_dim = self.out_shape()  # j_total * in_channels (pre-pooling)
        act_module = torch.nn.ReLU if self.activation_name == 'relu' else torch.nn.SiLU

        # Unified MLP for node-level or graph-level prediction
        layers: List[torch.nn.Module] = []
        in_dim = flat_node_dim if self.predict_per_node else (flat_node_dim * self.readout_multiplier)
        for hidden in self.mlp_head_hidden_dims:
            layers.append(torch.nn.Linear(in_dim, int(hidden)))
            if self.mlp_dropout_p > 0.0:
                layers.append(torch.nn.Dropout(self.mlp_dropout_p))
            layers.append(act_module())
            in_dim = int(hidden)
        layers.append(torch.nn.Linear(in_dim, self.out_dim))
        self.mlp_head = torch.nn.Sequential(*layers)


    def on_best_model(self, save_path: str, fold_i: Optional[int]) -> None:
        """
        Save the state of the wavelet scales selector
        matrix of the best model (as a numpy array).
        If saved during k-folds cross validation, save
        best for each fold by overwriting npy file with
        the filename index for the fold in its name.

        Note that if no matrix is saved, the model
        failed to learn / improved beyond initial weights,
        meaning the dyadic initial matrix was 'best'.
        """
        if self.save_best_selector_matrix:
            # to save folds' matrix 'npy' files in a subdirectory:
            # save_path = f"{save_path}/selector_matrix" \
            #     if (fold_i is not None) else f"{save_path}"
            # Path(save_path).mkdir(exist_ok=True)
            filepath = f"{save_path}/best_selector_matrix"
            if (fold_i is not None):
                filepath += f"_{fold_i}"
            m = self.wavelet_constructor.cpu().detach().numpy()
            np.save(filepath, m)
            

    def out_shape(self):
        # Base per-node embedding length prior to any graph pooling.
        # length = [n filters across orders] * [in_channels]
        # n_filters_all_orders = 11 for J = 4; 16 for J = 5
        n_filters_all_orders = 1 \
            + self.n_wavelets \
            + int(self.n_wavelets * (self.n_wavelets - 1) / 2) # 2nd-order
        return n_filters_all_orders * self.in_channels

    
    def _get_dyadic_scales_matrix_tensor(self, J: int) -> torch.Tensor:
        """
        DJ: added this method instead of hardcoded selector 
        matrix dyadic initialization.
        - If J = 4, get T = 16
        - If J = 5, get T = 32
        """
        m = torch.zeros((J, 2 ** J + 1))
        # in the original LEGS code, -1s were left of (smaller index than)
        # +1s in hardcoded scales selector matrix: I think this is 
        # backwards (we want to subtract greater powers)
        # cf. Eq 4 in LEGS paper
        # I've corrected things here, so -1s are at greater indices than +1s
        ones_idx = [2 ** i for i in range(0, J)]
        neg_ones_idx = [2 ** i for i in range(1, J + 1)]
        
        for j in range(J):
            m[j, ones_idx[j]] = 1.0
            m[j, neg_ones_idx[j]] = -1.0
        return m.to(torch.float)

    
    def _feng_filters_2nd_order(self):
        """
        DJ: these Feng filters (index subset) are to only
        include 2nd-order filters where a higher-power 
        (lower-frequency) wavelet is applied to all lower-power
        (higher frequency) 1st order filtrations; that is, 
        ensure j < j'. (Also generalized this method for 
        arbitrary J.)
        """
        idx = [self.J]
        for j in range(2, self.J):
            for jprime in range(0, j):
                idx.append(self.J * j + jprime)
        # example: idx = [4, 8, 9, 12, 13, 14] if self.J = 4
        return idx


    def _pool_node_features(
        self, x: torch.Tensor, 
        batch_index: Optional[torch.Tensor], 
        num_graphs: Optional[int] = None
    ) -> torch.Tensor:
        if self.pool_type == 'moments':
            if num_graphs is None:
                num_graphs = 1 if batch_index is None else int(torch.max(batch_index).item()) + 1
            # Expect x shaped (N, J_total, C). For pooling we need the unflattened x_all.
            # Caller will pass x_all for 'moments' mode.
            return moments_channel_pool(x, batch_index, num_graphs)
        # sum/mean/max on flattened per-node embeddings
        if batch_index is None:
            if self.pool_type == 'sum':
                return x.sum(dim=0, keepdim=True)
            if self.pool_type == 'mean':
                return x.mean(dim=0, keepdim=True)
            if self.pool_type == 'max':
                return x.max(dim=0, keepdim=True).values
        else:
            return self.pool(x, batch_index)
        raise ValueError(f"Unsupported pool type '{self.pool_type}'")

    def forward(self, data):
        """
        einsums key:
            j: number of wavelet filters
            p: number of wavelet filters, repeated for outer product
            t: max diffusion step + 1 (= num. cols in selector matrix)
            n: total number of nodes in batch
            c: number of (node signal) channels
        """
        # Prepare node signals x (N, F_in). If missing, use learnable bias.
        edge_index = data.edge_index
        x = getattr(data, self.feature_attr, None)
        if x is None:
            raise ValueError(
                f"LEGSModel expected feature '{self.feature_attr}' on data but it was missing."
            )
        else:
            if isinstance(x, torch.Tensor) and x.is_sparse:
                x = x.to_dense()
            if not isinstance(x, torch.Tensor) or x.dim() != 2:
                raise ValueError(f"Feature '{self.feature_attr}' must be a 2D torch.Tensor of shape (N, F)")
            if x.shape[-1] != self.in_channels:
                raise ValueError(
                    f"LEGSModel in_channels={self.in_channels} but feature '{self.feature_attr}' has dim {x.shape[-1]}. "
                    f"Please set in_channels accordingly or pre-process features to expected size."
                )

        
        # 0th-order scattering (don't take modulus of x)
        s0 = x.unsqueeze(dim=1) # shape 'n1c'


        # 1st-order scattering: |Wjx| or Wjx for 1 <= j <= J
        avgs = [None] * self.n_diffusion_steps
        # P^0 x = x
        avgs[0] = s0
        for i in range(self.n_diffusion_steps - 1):
            # recursive diffusion (powers of P^t @ x) starting at t = 1
            avgs[i+1] = self.diffusion_layer1(avgs[i], edge_index)
        diffusion_levels = torch.stack(avgs) # shape 'tnc1'

        s1 = torch.einsum(
            'jt,tnc->njc',
            self.wavelet_constructor,
            diffusion_levels.squeeze() 
        )
        # optional: take modulus of s1 (loses information: -x is conflated with x)
        # to use as the final s1 output features
        if self.apply_modulus_to_scatter:
            s1 = torch.abs(s1)

        
        # 2nd-order scattering: |Wj'|Wjx|| or Wj'|Wjx| for 1 <= j < j' <= J
        avgs = [None] * self.n_diffusion_steps
        # take modulus of s1 if not taken already before applying Wj'
        avgs[0] = s1 if self.apply_modulus_to_scatter else torch.abs(s1)
        for i in range(self.n_diffusion_steps - 1):
            avgs[i+1] = self.diffusion_layer2(avgs[i] , edge_index)
        # take modulus of 1st-order filtrations
        diffusion_levels_2 = torch.stack(avgs)

        # note: pj dimension is squared/outer product of j filters: each filter 
        # gets applied to every other in AT's approach, and then subsetted out 
        # to keep only where j > j'
        s2 = torch.einsum(
            'pt,tncj->npjc', 
            self.wavelet_constructor,
            diffusion_levels_2 
        )
        # flatten pj into one dimension ('njc') and subset to where Wj'|Wjx| has 
        # j < j' only
        s2 = s2.reshape(s2.shape[0], -1, self.in_channels)
        s2 = s2[:, self.feng_filters_2nd_order, :]
        if self.apply_modulus_to_scatter:
            s2 = torch.abs(s2)

        # concatenate 0th-, 1st-, and (feng-filtered) 2nd-order scattering coeffs
        # in the 'j' dim (all should have shape 'njc')
        x_all = torch.cat((s0, s1, s2), dim=1)  # (N, j_total, C)

        if self.verbosity > 0:
            print('x_all.shape (node features before heads):', x_all.shape)

        # Build per-node embedding by flattening filter and channel dims
        h_node = x_all.reshape(x_all.shape[0], -1)  # (N, flat_node_dim)
        batch_index = getattr(data, 'batch', None)

        # Node-level vs graph-level predictions
        if self.predict_per_node:
            return self.mlp_head(h_node)
        else:  # graph-level prediction
            # Support multiple pool types with concatenation
            pooled_parts: List[torch.Tensor] = []
            num_graphs = 1 if (batch_index is None) else data.num_graphs
            for pt in self.pool_types_list:
                if pt == 'moments':
                    pooled_m = self._pool_node_features(x_all, batch_index, num_graphs)
                    if pooled_m.dim() > 2:
                        pooled_m = pooled_m.reshape(pooled_m.shape[0], -1)
                    pooled_parts.append(pooled_m)
                else:
                    if batch_index is None:
                        if pt == 'sum':
                            pooled = h_node.sum(dim=0, keepdim=True)
                        elif pt == 'mean':
                            pooled = h_node.mean(dim=0, keepdim=True)
                        elif pt == 'max':
                            pooled = h_node.max(dim=0, keepdim=True).values
                        else:
                            raise ValueError(f"Unsupported pool type '{pt}'")
                    else:
                        pooled = self._pool_map[pt](h_node, batch_index)
                    pooled_parts.append(pooled)
            pooled_concat = pooled_parts[0] if len(pooled_parts) == 1 else torch.cat(pooled_parts, dim=-1)
            return self.mlp_head(pooled_concat)
        

class LazyLayer(torch.nn.Module):
    r""" 
    AT: Currently a single elementwise multiplication with one laziness parameter per
    channel. This is run through a softmax so that this is a real laziness parameter.

    DJ: This optional layer creates a trainable weight in an alternative construction of 
    the diffusion operator $P$, $P_{\alpha}$. See "Relaxed geometric scattering" on
    p. 2 of https://ml4molecules.github.io/papers2020/ML4Molecules_2020_paper_63.pdf
    (The 'Lazy' refers to modification of the lazy random walk matrix)
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
    

def gcn_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    add_self_loops=False,
    dtype=None
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1), ), 
            dtype=dtype,
            device=edge_index.device
        )

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    return edge_index, deg_inv_sqrt[row] * edge_weight


class Diffuse(MessagePassing):
    """ 
    Implements low pass walk with optional weights.
    """
    def __init__(
        self, 
        in_channels,
        out_channels, 
        trainable_laziness=False,
        fixed_weights=True
    ):
        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None):

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        # turn off this step for simplicity
        if not self.fixed_weights:
            x = self.lin(x)

        # Step 3: Compute normalization
        edge_index, edge_weight = gcn_norm(
            edge_index, 
            edge_weight, 
            num_nodes=x.size(self.node_dim), 
            dtype=x.dtype
        )

        # Step 4-6: Start propagating messages.
        propogated = self.propagate(
            x=x,
            edge_index=edge_index, 
            edge_weight=edge_weight,
            size=None,
        )
        if not self.trainable_laziness:
            return 0.5 * (x + propogated)

        return self.lazy_layer(x, propogated)


    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j


    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out