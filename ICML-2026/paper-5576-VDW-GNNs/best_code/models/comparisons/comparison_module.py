import torch
from torch import nn
from torch_geometric.data import Data

from models.base_module import BaseModule


class ComparisonModel(BaseModule):
    """
    Thin wrapper that adapts a PyG comparison model to the BaseModule API used by
    the training loop. It delegates forward passes to the wrapped PyG model and
    exposes predictions under the expected 'preds' key.

    The wrapper adds small compatibility shims:
    - Ensures a node index attribute named 'atoms' exists (falls back to 'z' or
      creates a zero-index tensor) for models that embed atomic numbers.

    Additionally, this wrapper supports a wind-style setup where graph connectivity
    (neighbors) is defined by one geometry (e.g., Earth coordinates in `batch.pos`)
    but equivariant message passing is performed in a different vector space
    (e.g., wind vectors). If the wrapped model exposes `pos_input_key` (a string
    attribute naming a tensor on the batch), the wrapper will temporarily replace
    `batch.pos` with `batch[pos_input_key]` for the duration of the forward pass,
    then restore the original `batch.pos`.
    """

    def __init__(
        self,
        pyg_model: nn.Module,
        *,
        base_module_kwargs: dict,
        atomic_number_key: str = 'z',
    ) -> None:
        super().__init__(**base_module_kwargs)
        self.pyg_model = pyg_model
        self.atomic_number_key = atomic_number_key

    def _ensure_atoms_attribute(self, batch: Data) -> None:
        """
        Ensure `batch.atoms` exists for compatibility with some baseline models.

        If absent, try to alias from `self.atomic_number_key` (default 'z'). If
        still absent, create a zero-index LongTensor of length num_nodes.
        """
        # If the wrapped model explicitly supports operating without `atoms`,
        # and does not have an embedding table, avoid synthesizing one so the
        # model can use its bias-initialized constant node features.
        if hasattr(self.pyg_model, 'use_bias_if_no_atoms') \
        and getattr(self.pyg_model, 'use_bias_if_no_atoms'):
            emb_in = getattr(self.pyg_model, 'emb_in', None)
            if emb_in is None:
                return
        if hasattr(batch, 'atoms') and batch.atoms is not None:
            return

        if hasattr(batch, self.atomic_number_key):
            atoms = getattr(batch, self.atomic_number_key)
            if not torch.is_tensor(atoms):
                atoms = torch.as_tensor(atoms)
            batch.atoms = atoms.to(dtype=torch.long, device=batch.pos.device if hasattr(batch, 'pos') else self.get_device())
            return

        # Fallback: synthesize a single-type atom index vector
        num_nodes = None
        if hasattr(batch, 'pos') and isinstance(batch.pos, torch.Tensor):
            num_nodes = batch.pos.size(0)
        elif hasattr(batch, 'x') and isinstance(batch.x, torch.Tensor):
            num_nodes = batch.x.size(0)
        if num_nodes is None:
            raise ValueError("Could not infer number of nodes to create dummy 'atoms' attribute.")
        device = batch.pos.device if hasattr(batch, 'pos') else self.get_device()
        batch.atoms = torch.zeros(num_nodes, dtype=torch.long, device=device)

    def forward(self, batch: Data) -> dict:
        # Small input normalization for comparison models that expect an 'atoms' field
        self._ensure_atoms_attribute(batch)

        # Optional: allow a wrapped model to request a different attribute be used
        # as the PyG-standard `pos` input (e.g., wind vectors as EGNN coordinates).
        original_pos = None
        swapped_pos = False
        pos_input_key = getattr(self.pyg_model, 'pos_input_key', None)
        if isinstance(pos_input_key, str) and len(pos_input_key) > 0:
            if not hasattr(batch, 'pos'):
                raise ValueError(
                    "ComparisonModel expected `batch.pos` to exist for pos swapping, but it was missing."
                )
            if hasattr(batch, pos_input_key):
                candidate_pos = getattr(batch, pos_input_key)
                if torch.is_tensor(candidate_pos):
                    original_pos = batch.pos
                    # Basic shape checks to avoid silent misuse
                    if (not torch.is_tensor(original_pos)) or original_pos.ndim != 2:
                        raise ValueError("ComparisonModel expected `batch.pos` to be a rank-2 tensor.")
                    if candidate_pos.ndim != 2:
                        raise ValueError(
                            f"Requested pos_input_key='{pos_input_key}' must be a rank-2 tensor, got ndim={candidate_pos.ndim}."
                        )
                    if candidate_pos.shape[0] != original_pos.shape[0]:
                        raise ValueError(
                            f"Requested pos_input_key='{pos_input_key}' has incompatible num_nodes: "
                            f"{candidate_pos.shape[0]} vs batch.pos {original_pos.shape[0]}."
                        )
                    if candidate_pos.shape[1] != original_pos.shape[1]:
                        raise ValueError(
                            f"Requested pos_input_key='{pos_input_key}' has incompatible feature dim: "
                            f"{candidate_pos.shape[1]} vs batch.pos {original_pos.shape[1]}."
                        )
                    if candidate_pos.device != original_pos.device or candidate_pos.dtype != original_pos.dtype:
                        candidate_pos = candidate_pos.to(device=original_pos.device, dtype=original_pos.dtype)
                    batch.pos = candidate_pos
                    swapped_pos = True

        ignore_edge_weight = False
        if hasattr(self.pyg_model, "ignore_edge_weight"):
            ignore_edge_weight = bool(getattr(self.pyg_model, "ignore_edge_weight"))

        edge_weight = getattr(batch, "edge_weight", None)
        try:
            if ignore_edge_weight and edge_weight is not None:
                batch.edge_weight = None
                try:
                    preds = self.pyg_model(batch)
                finally:
                    batch.edge_weight = edge_weight
            else:
                preds = self.pyg_model(batch)
        finally:
            if swapped_pos:
                batch.pos = original_pos
        # If wrapped model already returns a dict, pass through; else wrap tensor
        if isinstance(preds, dict):
            return preds
        return {'preds': preds}


