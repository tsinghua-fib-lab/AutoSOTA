import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from typing import List, Tuple, Optional, Dict
import h5py
import os

# Global cache for HDF5 file handles (not pickled with dataset)
_HDF5_FILE_CACHE = {}


class VDWData(Data):
    operator_keys = ('P', 'P_line', 'Q')
    def __init__(self, *args, operator_keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        if operator_keys is not None:
            self.operator_keys = operator_keys

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key:
            return 1
        elif key in self.operator_keys:
            # Set block-diagonal batching for operator tensors
            return (0, 1)
        else:  
            return 0

    def to(self, device, *args, **kwargs):
        data = super().to(device, *args, **kwargs)
        # Move all tensor attributes to the correct device, if not already on device
        for key in self.keys():
            item = getattr(self, key, None)
            if torch.is_tensor(item): # and item.is_sparse:
                setattr(data, key, item.to(device))
        return data


class VDWDatasetHDF5(Dataset):
    """
    Dataset class for loading PyTorch Geometric Data objects from an HDF5 file.
    Expects operator tensors to be stored in the HDF5 file under the provided keys in operators_tup.
    Assumes that the Data objects have already been loaded and are stored in self.data_list.
    Assumes that the HDF5 file has been created by the parallel_process_dataset.py script.
    If index_map is provided, it maps dataset indices to original indices for HDF5 lookup.
    """
    def __init__(
        self,
        model_key: str,
        data_list: List[Data],
        h5_path: str,
        sparse_operators_tup: Tuple[str, ...] = ('P', 'P_line', 'Q'),
        index_map: Optional[Dict[int, int]] = None,
        scalar_feat_key: Optional[str] = 'z',
        vector_feat_key: Optional[str] = None,
        attach_on_access: bool = True,
        attributes_to_drop: Optional[List[str]] = None,
        num_edge_features: int = 16,
        num_bond_types: int = 5,  # 4 types + 1 no bond
        # Lazy-scatter configuration
        line_operator_key: str = 'P_line',
        vector_operator_key: str = 'Q',
        line_scatter_feature_key: Optional[str] = None,
        vector_scatter_feature_key: Optional[str] = None,
        line_scatter_kwargs: Optional[Dict] = None,
        vector_scatter_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            data_list: List of Data objects (without operators)
            h5_path: Path to HDF5 file
            sparse_operators_tup: Tuple of sparse operator keys
            index_map: Dictionary mapping dataset indices to original indices for 
                HDF5 lookup
            scalar_feat_key: Key for scalar feature
            vector_feat_key: Key for vector feature
            attach_on_access: Whether to attach attribs in __getitem__ (runs every 
                time a data object is accessed; set to False in a batch-wise attachment
                collate_fn pipeline)
        """
        self.model_key = model_key
        self.data_list = data_list  # List of Data objects (without operators)
        self.h5_path = h5_path
        self.sparse_operators_tup = sparse_operators_tup
        self.index_map = index_map
        self.scalar_feat_key = scalar_feat_key
        self.attach_on_access = attach_on_access
        self.attributes_to_drop = attributes_to_drop
        self._num_edge_features = num_edge_features
        self._num_bond_types = num_bond_types
        # Lazy-scatter settings
        self.line_operator_key = line_operator_key
        self.vector_operator_key = vector_operator_key
        self.line_scatter_feature_key = line_scatter_feature_key
        self.vector_scatter_feature_key = vector_scatter_feature_key
        self.line_scatter_kwargs = dict(line_scatter_kwargs or {})
        self.vector_scatter_kwargs = dict(vector_scatter_kwargs or {})
        # Store feature keys for flattening
        # Try to infer from first data object if not provided
        if vector_feat_key is None:
            # Try to get from first data object
            if hasattr(self.data_list[0], 'vector_feat_key'):
                vector_feat_key = self.data_list[0].vector_feat_key
            else:
                raise ValueError(
                    f"Vector feature key {vector_feat_key} not found in first data object."
                )
        self.vector_feat_key = vector_feat_key
        super().__init__()

    def __len__(self):
        return len(self.data_list)

    def _get_h5(self):
        # Use global cache to avoid pickling issues
        global _HDF5_FILE_CACHE
        
        process_id = os.getpid()
        cache_key = f"{self.h5_path}_{process_id}"
        
        if cache_key not in _HDF5_FILE_CACHE:
            _HDF5_FILE_CACHE[cache_key] = h5py.File(self.h5_path, 'r')
        
        return _HDF5_FILE_CACHE[cache_key]

    # ------------------------------------------------------------------
    # Internal helper – attach operators and graph tensors
    # ------------------------------------------------------------------
    def _attach_attribs(
        self, 
        data: Data, 
        orig_idx: int
    ) -> None:
        """
        (Idempotent) Attach P/Q and other graph tensor attributes to data in-place. 
        If attributes are already attached, do nothing.
        """
        if getattr(data, '_attribs_loaded', False):
            return  # Already attached on this worker

        h5f = self._get_h5()
        
        # To ensure PyG collation works without device-mismatch errors,
        # keep all per-sample tensors on CPU; the batch will be moved
        # to the Accelerator device after collation.
        device = torch.device('cpu')

        # ------------------------------------------------------------------
        # Sparse operators
        # ------------------------------------------------------------------
        for op_key in self.sparse_operators_tup:
            grp = h5f[op_key][str(orig_idx)]
            op_tensor = torch.sparse_coo_tensor(
                indices=torch.from_numpy(grp['indices'][...]),
                values=torch.from_numpy(grp['values'][...]),
                size=tuple(grp['size'][...].tolist()),
                dtype=torch.float32,  # float32 required for sparse operations in pytorch 2.4.1
            ).to(device)
            setattr(data, op_key, op_tensor)

        # ------------------------------------------------------------------
        # Dense graph tensors
        # ------------------------------------------------------------------
        # Preserve original chemical bonds before overwriting edge_index
        orig_edge_index = getattr(data, 'edge_index', None)
        orig_edge_attr = getattr(data, 'edge_attr', None)

        if 'edge_index' in h5f and str(orig_idx) in h5f['edge_index']:
            ei_vals = torch.from_numpy(
                h5f['edge_index'][str(orig_idx)]['values'][...]
            )
            ei_vals = ei_vals.to(dtype=torch.int64, device=device)

            # Ensure shape is (2,E).  Stored flat? -> reshape.
            if ei_vals.dim() == 1:
                assert ei_vals.numel() % 2 == 0
                ei_vals = ei_vals.view(2, -1)

            data.edge_index = ei_vals

        # Derive categorical edge_type from one-hot edge_attr if needed
        if (not hasattr(data, 'edge_type') or data.edge_type is None) \
        and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            with torch.no_grad():
                data.edge_type = data.edge_attr.argmax(dim=1).to(torch.long)

        if 'edge_weight' in h5f and str(orig_idx) in h5f['edge_weight']:
            ew_vals = torch.from_numpy(h5f['edge_weight'][str(orig_idx)]['values'][...])
            data.edge_weight = ew_vals.to(dtype=torch.float32).to(device)

        # Edge features (Bessel RBFs) — assumed to match edges in radial cutoff graph, so we load directly
        if 'edge_features' in h5f and str(orig_idx) in h5f['edge_features']:
            grp_b = h5f['edge_features'][str(orig_idx)]
            vals = torch.from_numpy(grp_b['values'][...])
            shape = tuple(grp_b['shape'][...].tolist())
            data.edge_features = vals.view(*shape).to(dtype=torch.float32).to(device)

        # ------------------------------------------------------------------
        # Edge attributes
        # ------------------------------------------------------------------
        if hasattr(data, 'edge_attr'):
            num_edges = data.edge_index.shape[1]

            # Determine class indices: prefer original 4-class bond set (+1 no-bond) if available
            if (orig_edge_attr is not None) and isinstance(orig_edge_attr, torch.Tensor) \
            and (orig_edge_attr.dim() == 2) and (orig_edge_attr.shape[1] >= 4):
                real_k = int(orig_edge_attr.shape[1])  # e.g., 4 for common bond types
                no_bond_idx = real_k  # append as new class at the end
                total_k = real_k + 1
            else:
                # Fallback to configured size: assume last index is 'no bond'
                total_k = int(self._num_bond_types)
                no_bond_idx = max(total_k - 1, 0)

            # If we have original chemical bonds, label new edges accordingly
            if (orig_edge_index is not None) and (orig_edge_attr is not None) \
            and isinstance(orig_edge_attr, torch.Tensor) and (orig_edge_attr.dim() == 2) \
            and (orig_edge_attr.shape[1] >= 4):
                with torch.no_grad():
                    orig_types = orig_edge_attr.argmax(dim=1)
                    bond_map = {}
                    for e in range(orig_edge_index.shape[1]):
                        i = int(orig_edge_index[0, e].item())
                        j = int(orig_edge_index[1, e].item())
                        t = int(orig_types[e].item())  # 0..real_k-1
                        bond_map[(i, j)] = t
                        bond_map[(j, i)] = t

                    row, col = data.edge_index[0], data.edge_index[1]
                    E = data.edge_index.shape[1]
                    et = torch.full((E,), no_bond_idx, dtype=torch.long, device=device)
                    for e in range(E):
                        i = int(row[e].item()); j = int(col[e].item())
                        et[e] = bond_map.get((i, j), no_bond_idx)
                    data.edge_type = et
                    # Provide edge_attr with 'no bond' appended as last class
                    # For non-bonded k-NN edges, set the 'no bond' feature value
                    # to the Euclidean distance between nodes (instead of 1.0).
                    one_hot = torch.zeros(E, total_k, dtype=torch.float32, device=device)
                    one_hot.scatter_(1, et.view(-1, 1), 1.0)

                    # Replace the 'no bond' column values with L2 distances for non-bond edges
                    pos = getattr(data, self.vector_feat_key)
                    row, col = data.edge_index[0], data.edge_index[1]
                    diff = pos[col] - pos[row]
                    l2_dist = torch.linalg.norm(diff, dim=1)
                    no_bond_mask = (et == no_bond_idx)
                    one_hot[no_bond_mask, no_bond_idx] = l2_dist[no_bond_mask].to(one_hot.dtype)

                    data.edge_attr = one_hot

            # Ensure edge_type exists and matches edge count
            if hasattr(data, 'edge_type') and (data.edge_type is not None):
                if data.edge_type.shape[0] != num_edges:
                    et = data.edge_type
                    new_et = torch.full((num_edges,), no_bond_idx, dtype=torch.long, device=device)
                    keep = min(et.shape[0], num_edges)
                    new_et[:keep] = et[:keep]
                    data.edge_type = new_et
            # else:
            #     data.edge_type = torch.full((num_edges,), no_bond_idx, dtype=torch.long, device=device)

        # Dirac nodes (data stored in HDF5)
        if ('dirac_nodes' in h5f) \
        and str(orig_idx) in h5f['dirac_nodes']:
            dn_vals = torch.from_numpy(
                h5f['dirac_nodes'][str(orig_idx)]['values'][...]
            )
            data.dirac_nodes = one_hot(
                dn_vals, num_classes=data.num_nodes
            ).float().T

        # ------------------------------------------------------------------
        # Optionally drop unused attributes to save memory
        # ------------------------------------------------------------------
        if self.attributes_to_drop is not None:
            for _attr in self.attributes_to_drop:
                if hasattr(data, _attr):
                    delattr(data, _attr)

        data._attribs_loaded = True


    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.attach_on_access:
            orig_idx = self.index_map[idx] \
                if self.index_map is not None else idx
            self._attach_attribs(data, orig_idx)
        return data


    def close_file_handles(self):
        """Close the HDF5 file handle for the current process."""
        global _HDF5_FILE_CACHE
        
        process_id = os.getpid()
        cache_key = f"{self.h5_path}_{process_id}"
        
        if cache_key in _HDF5_FILE_CACHE:
            _HDF5_FILE_CACHE[cache_key].close()
            del _HDF5_FILE_CACHE[cache_key]

