from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from .transforms_srht import SRHT, batched_hutch_scores, srht_columns as _srht_columns

# Constants for numerical stability and configuration
MAX_TRAINABLE_FRACTION = 0.78  # Maximum fraction of neurons that can be trainable (90% rule)
REGULARIZATION_COEFFICIENT = 1e-6  # Tikhonov regularization coefficient for numerical stability in eigenproblem solving
NEAR_ZERO_THRESHOLD = 1e-12  # Threshold for detecting near-zero weight matrices (degenerate case)
EPSILON_FOR_DIVISION = 1e-12  # Small epsilon to prevent division by zero in cumulative sum calculations


@torch.no_grad()
def qr_canonical(X: torch.Tensor):
    """
    Canonical QR decomposition that removes sign ambiguity.
    
    Forces diag(R) > 0 by flipping signs of Q columns and R rows accordingly.
    This makes QR deterministic (up to degeneracy) and ensures Q_in reconstruction
    is almost exact after save/load.
    
    Args:
        X: Input matrix [n, k]
    
    Returns:
        Q: Orthonormal matrix [n, k] with canonical sign
        R: Upper triangular matrix [k, k] with positive diagonal
    """
    # Reduced QR
    Q, R = torch.linalg.qr(X, mode="reduced")  # Q:[n,k], R:[k,k]
    d = torch.diagonal(R, 0)
    s = torch.sign(d)
    s[s == 0] = 1  # avoid zeros
    
    # Q' = Q @ diag(s), R' = diag(s) @ R  (keeps X = Q R)
    Q = Q * s  # columnwise
    R = (s.unsqueeze(1)) * R  # rowwise
    
    return Q, R


# Note: TF32 can be enabled for faster matmuls on Ampere+ GPUs (A100, H100, etc.)
# Users can enable it manually if desired:
#   torch.backends.cuda.matmul.allow_tf32 = True
#   torch.backends.cudnn.allow_tf32 = True
# We don't set it by default to avoid unexpected global side effects.

try:
    from transformers.pytorch_utils import Conv1D
except Exception:
    class Conv1D(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        def forward(self, x):
            y = x @ self.weight.T
            if self.bias is not None:
                y = y + self.bias
            return y


@torch.no_grad()
def select_output_neurons(
    W: torch.Tensor,
    topk: int = 8,
    d_prime: int = 128,          # sketch dimension
    anchors: int = 256,          # number of anchor rows
    seed: int = 42,
    mm_dtype: torch.dtype = torch.bfloat16,
    anchor_strategy: str = "stride",  # "stride" or "rand"
) -> torch.Tensor:
    """
    Fast neuron selector using Gaussian Random Projection + anchors:
      1) Gaussian RP
      2) Anchor mean-|cos| scoring in sketched space
      3) Top-k selection

    Args:
        W: Weight matrix [N, D] with N = #output neurons
        topk: Number of neurons to select
        d_prime: Random projection dimension
        anchors: Number of anchor neurons for scoring
        seed: Random seed 
        mm_dtype: Matrix multiplication dtype (bfloat16 for speed)
        anchor_strategy: "stride" for deterministic or "rand" for random

    Returns:
        Indices of selected neurons [topk]
    """
    device = W.device
    N, D = W.shape

    # 1) Row norms - compute in model dtype (robust operation)
    norms = torch.linalg.norm(W, dim=1, keepdim=True).clamp_min_(1e-8)  # [N,1]

    # 2) Random projection: Z0 = W @ R
    g = torch.Generator(device=device).manual_seed(seed)
    R = torch.randn(D, d_prime, generator=g, device=device, dtype=mm_dtype)
    Wmm = W.to(mm_dtype, copy=False)
    Z0 = Wmm @ R  # [N, d'] 

    # 3) Normalize rows to approximate cosine geometry
    Z = Z0 / norms.to(mm_dtype)  # [N, d']

    # 4) Choose anchors without O(N) randperm
    if anchors >= N:
        A_idx = torch.arange(N, device=device)
    elif anchor_strategy == "stride":
        # Deterministic sub-sampling: ~uniform without a full shuffle
        step = N // anchors
        A_idx = torch.arange(0, step * anchors, step, device=device)
    else:  # "rand"
        A_idx = torch.randint(0, N, (anchors,), generator=g, device=device)

    A = Z[A_idx]  # [A, d']

    # 5) Mean |cos| to anchors: sims = |Z @ A^T|
    sims = (Z @ A.t()).abs()  # [N, A]
    scores = sims.mean(dim=1)  # [N]

    _, idx = torch.topk(scores, k=min(topk, N), largest=True, sorted=False)
    return idx



@torch.no_grad()
def _polish_q_in_eigenspace(
    W: torch.Tensor,
    H: SRHT,
    C: torch.Tensor,
    k: int,
    mm_dtype: torch.dtype = torch.bfloat16,
    storage_dtype: torch.dtype = None,  # dtype for storing U_polish (None = use W.dtype)
) -> tuple:
    """
    Solve small m×m eigenproblem within candidate pool to find bottom-k eigenspace.
    
    This is the polishing step: within span(H @ E[C]), find the k dimensions
    with minimum energy by solving:
        A = W @ H_C
        M = A^T @ A  (m×m SPD matrix)
        eigh(M) -> take bottom-k eigenvectors
        Q_in = H_C @ U_bottom
    
    Args:
        W: Weight matrix [d_out, d_in]
        H: SRHT transform
        C: [m] candidate indices (m >> k)
        k: Target dimension for Q_in
        mm_dtype: Data type for matrix multiplication (bfloat16 for speed)
        
    Returns:
        U_polish: [m, k] small eigenvector matrix (for saving/reconstruction)
        Q_in: [d_in, k] polished orthonormal basis
    """
    device = W.device
    d_in = W.shape[1]
    m = C.numel()
    
    # Check for NaN in weight matrix
    if torch.isnan(W).any():
        raise ValueError(f"[PLATE] NaN detected in weight matrix W (shape {W.shape})")
    
    # 1) Get SRHT columns for candidates: H @ E[C]
    H_C = _srht_columns(H, C)  # [d_in, m]
    
    # 2) Project weight matrix: A = W @ H_C (float32 for SVD stability)
    A = (W.to(mm_dtype) @ H_C.to(mm_dtype)).float()  # [d_out, m]
    
    # 3) Build small Gram matrix: M = A^T @ A  (m×m, SPD)
    M = (A.t() @ A).to(torch.float32)
    
    # Add regularization for numerical stability (Tikhonov regularization)
    # This prevents singular/ill-conditioned matrices
    eps = REGULARIZATION_COEFFICIENT * torch.trace(M) / M.shape[0]  # Adaptive regularization
    M = M + eps * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    
    # 4) Solve eigenproblem using SVD on A (more stable than eigh on M = A^T @ A)
    # SVD of A: A = U @ S @ V^T, where A is [d_out, m]
    # Then M = A^T @ A = V @ S^2 @ V^T
    # So eigenvectors of M are columns of V, eigenvalues are S^2
    try:
        # Full SVD on A (more numerically stable than forming M = A^T @ A)
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)  # A is [d_out, m]
        # U: [d_out, min(d_out, m)]
        # S: [min(d_out, m)] (descending order)
        # Vt: [min(d_out, m), m]
        
        # V^T is Vt, so V is Vt.t()
        # Columns of V are eigenvectors of M = A^T @ A
        V = Vt.t()  # [m, min(d_out, m)]
        
        # Take bottom-k (smallest eigenvalues)
        # S is in descending order, so smallest are at the end
        evecs = V[:, -k:]  # Last k columns = smallest k singular values
        evals = (S[-k:] ** 2)  # Last k singular values squared
        
        # Reverse to get ascending order (smallest first)
        evecs = evecs.flip(dims=[1])
        evals = evals.flip(dims=[0])
        
        # Cleanup SVD intermediates immediately to free memory
        del U, S, Vt, V, evals
        
    except Exception as e:
        # Last resort: Full eigendecomposition on CPU
        import warnings
        warnings.warn(f"[PLATE] SVD failed ({e}), falling back to eigh on CPU")
        M_cpu = M.cpu()
        evals, evecs = torch.linalg.eigh(M_cpu)
        evals = evals[:k].to(device)
        evecs = evecs[:, :k].to(device)
        del M_cpu, evals
    
    # Cleanup intermediate buffers immediately
    del A, M
    
    # 5) Extract bottom-k eigenvectors (already done by SVD!)
    # Store in model's dtype for consistency with accelerator.get_state_dict()
    out_dtype = storage_dtype if storage_dtype is not None else W.dtype
    U_polish = evecs.to(out_dtype).contiguous()  # [m, k]
    
    # 6) Reconstruct Q_in = H_C @ U_polish
    Q_in = H_C @ U_polish.to(H_C.dtype)  # [d_in, k]
    
    # Cleanup H_C after use
    del H_C
    
    # 7) Re-orthonormalize (QR needs float32 for numerical stability)
    Q_in_fp32, _ = qr_canonical(Q_in.float())
    Q_in = Q_in_fp32.to(out_dtype)
    
    # Cleanup float32 copy immediately
    del Q_in_fp32
    
    return U_polish, Q_in


@torch.no_grad()
def two_stage_selection(
    W: torch.Tensor,
    H: SRHT,
    k: int,
    q: int,
    seed: int,
    side: str = 'input',
    pool_multiple: int = 4,
    max_candidates: int = 8192,
    return_candidates: bool = False,
) -> torch.Tensor:
    """
    Select candidate coordinates for polishing using SRHT-based Hutch++.
    
    Screens all coordinates with Hutch++, then selects top pool_multiple*k candidates
    for subsequent SVD polishing. The polishing step finds the optimal subspace
    regardless of candidate ordering, so no refinement is needed.
    
    Args:
        W: Weight matrix [d_out, d_in]
        H: SRHT transform (either input or output dimension)
        k: Final number of coordinates to select
        q: Number of Hutch++ probes (4 recommended, 8-16 for higher quality)
        seed: Random seed for probe generation
        side: 'input' for W^T @ W or 'output' for W @ W^T
        pool_multiple: Candidate pool size multiplier (default: 4)
        max_candidates: Maximum candidate pool size (default: 8192)
                       Hard cap to prevent expensive SVD polishing when k is large
        return_candidates: If True, return all sorted candidates; if False, return top-k
        
    Returns:
        S: [k] or [m] selected coordinate indices (smallest energies)
    """
    # Screen all coordinates with Hutch++
    energies = batched_hutch_scores(W, H, q, seed, side=side)
    
    # Create candidate pool (pool_multiple x larger than final selection)
    # Cap at max_candidates to keep polishing computationally feasible
    m = int(min(pool_multiple * k, H.d, max_candidates))
    candidates = torch.argsort(energies)[:m]  # m smallest energies
    
    if return_candidates:
        # Return all candidates (for polishing)
        return candidates
    else:
        # Return top-k only (legacy behavior)
        return candidates[:k]


@torch.no_grad()
def build_q_in_from_weight(
    W: torch.Tensor,
    col_tau: float = 0.90,
    max_rank: int = 512,
    num_probes: int = 4,
    seed: int = 42,
    pool_multiple: int = 4,
    max_candidates: int = 8192,
    svd_threshold: int = 300,  # Use exact SVD if input dim <= this threshold
) -> Tuple[SRHT, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Q_in orthogonal input basis.
    
    For small dimensions (in_ <= svd_threshold), uses exact SVD.
    For large dimensions, uses SRHT-based Hutch++ approximation.
    
    Args:
        W: Weight matrix [n_frozen, d_in] containing ONLY frozen/preserved neuron rows
        col_tau: Energy threshold for column (input) space (must be in [0, 1])
        max_rank: Maximum number of coordinates to select
        num_probes: Number of Hutch++ probes (4 optimal from experiments, 16 for extra quality)
        seed: Base seed for deterministic transform
        pool_multiple: Candidate pool size multiplier (4 optimal from experiments)
        max_candidates: Maximum candidate pool size (default: 8192)
                       Hard cap to prevent expensive SVD polishing when k is large
        svd_threshold: Use exact SVD if input dimension <= this value (default: 64)
        
    Returns:
        H_in: SRHT transform for input dimension (None if using SVD)
        C_candidates: [m] candidate indices or dummy tensor if using SVD
        U_polish: [m, k_in] eigenvector matrix or identity if using SVD
        Q_in: [d_in, k_in] orthogonal input basis
    """
    # Validate col_tau is in [0, 1]
    if not 0 <= col_tau <= 1:
        raise ValueError(
            f"`col_tau` must be in [0, 1], got {col_tau}. "
            f"col_tau represents the energy threshold for column space selection."
        )
    
    out, in_ = W.shape
    device = W.device
    
    # ========================================================================
    # SMALL DIMENSION PATH: Use exact SVD (no approximation needed)
    # ========================================================================
    if in_ <= svd_threshold:
        
        # Compute SVD of W^T @ W to find column space energy
        # W is [out, in], W^T @ W is [in, in]
        # Eigenvalues of W^T @ W = squared singular values of W
        # Note: Convert to fp32 only for SVD (numerically sensitive operation)
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        # S is in descending order
        
        # Determine k_in from tau threshold (how many dims to keep in null space)
        total_energy = (S ** 2).sum()
        if total_energy < NEAR_ZERO_THRESHOLD:
            # Degenerate case: W is near zero, keep all input dims
            k_in = min(in_, max_rank)
        else:
            cumsum_energy = torch.cumsum(S ** 2 / total_energy, dim=0)
            # Find how many singular values capture col_tau of energy
            num_high = int((cumsum_energy >= col_tau).nonzero()[0].item()) + 1 if (cumsum_energy >= col_tau).any() else len(S)
            k_in = min(in_ - num_high, max_rank)
        
        k_in = max(k_in, 1)  # At least 1 dimension
        
        # Q_in spans the orthogonal complement (bottom-k_in right singular vectors)
        # Vt is [min(out,in), in] - rows correspond to right singular vectors
        # We want last k_in rows (smallest singular values), transposed to [in, k_in]
        if Vt.shape[0] >= k_in:
            # Take last k_in rows from Vt, then transpose
            Q_in = Vt[-k_in:, :].T.contiguous()  # [in, k_in]
        else:
            # Not enough singular vectors - use all available
            Q_in = Vt.T.contiguous()  # [in, min(out,in)]
        
        # Note: Q_in is already orthonormal (from SVD), no need to re-orthogonalize
        
        # Return values for API compatibility with SRHT path
        # H_in=None signals SVD mode to caller
        # C_dummy is stored in plate_S_in for device/shape inference
        # U_dummy is unused
        H_in = None
        C_dummy = torch.arange(Q_in.shape[1], device=device)
        U_dummy = None 
        
        return H_in, C_dummy, U_dummy, Q_in
    
    # ========================================================================
    # LARGE DIMENSION PATH: Use SRHT-based Hutch++ approximation
    # ========================================================================
    # Use W's dtype for SRHT operations (supports fp32, bf16, fp16)
    # FWHT works in any dtype; only SVD polishing converts to fp32 if needed
    seed_in = seed
    H_in = SRHT(in_, seed_in, device=device, dtype=W.dtype)
    energies_in_quick = batched_hutch_scores(W, H_in, q=2, seed=seed_in + 1000, side='input')
    
    # Determine k_in from tau threshold
    sorted_energies_in = torch.sort(energies_in_quick, descending=True)[0]
    cumsum_in = torch.cumsum(sorted_energies_in / (sorted_energies_in.sum() + EPSILON_FOR_DIVISION), dim=0)
    
    num_high_in = int((cumsum_in >= col_tau).nonzero()[0].item()) + 1 if (cumsum_in >= col_tau).any() else in_ // 2
    k_in = min(in_ - num_high_in, max_rank)
    k_in = max(k_in, 1)  # At least 1 dimension
    
    # Get large candidate pool via Hutch++ screening
    # SVD polishing is order-invariant, so no refinement needed
    C_candidates = two_stage_selection(
        W, H_in, k_in,
        q=num_probes,
        seed=seed_in + 2000,
        side='input',
        pool_multiple=pool_multiple,
        max_candidates=max_candidates,
        return_candidates=True  # Return all candidates for polishing
    )
    
    # Polish: solve m×m eigenproblem for bottom-k_in eigenspace
    # Experiments show 85-99% improvement over naive diagonal selection
    # Store U_polish in model's dtype for consistency with accelerator.get_state_dict()
    U_polish, Q_in = _polish_q_in_eigenspace(
        W, H_in, C_candidates, k=k_in, mm_dtype=torch.bfloat16, storage_dtype=W.dtype
    )
    
    return H_in, C_candidates, U_polish, Q_in


# -------- PLATE adapter layer --------

class PLATELinear(nn.Module, BaseTunerLayer):
    """
    PLATE (Plasticity-Tunable Efficient Adapters) adapter:
      ΔW x = B @ A @ Q_in^T @ x
      
    Combines:
    - B: Frozen neuron selection matrix 
    - A: Trainable transformation matrix
    - Q_in: Frozen orthogonal input basis 
    
    Key Design:
    B selects r trainable neurons. Q_in is computed from the COMPLEMENT (frozen/preserved
    neurons), ensuring updates to trainable neurons use directions orthogonal to the frozen
    span. 
    """
    # Adapter layer names that PEFT should track for requires_grad management
    # ONLY plate_A is trainable - this controls which layers get requires_grad toggled
    # plate_B is NOT included because it's always frozen (selection matrix)
    adapter_layer_names = ("plate_A",)
    # plate_B and other components are in other_param_names (not trainable)
    other_param_names = ("plate_B", "r", "plate_alpha", "scaling", "plate_dropout", "col_tau",
                         "plate_Q_in", "plate_S_in")
    
    def __init__(
        self,
        base_layer: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.base_layer = base_layer
        
        # Initialize adapter dictionaries (PEFT pattern)
        self.plate_A = nn.ModuleDict({})
        self.plate_B = nn.ModuleDict({})
        self.r = {}
        self.plate_alpha = {}
        self.plate_dropout = nn.ModuleDict({})
        self.col_tau = {}
        
        # BufferDicts for Q_in
        from peft.tuners._buffer_dict import BufferDict
        self.plate_Q_in = BufferDict({}, persistent=False)      # Don't save - recompute (unless SVD mode)
        self.plate_Q_in_svd = BufferDict({}, persistent=True)   # Save Q_in directly when using SVD mode
        self.plate_S_in = BufferDict({}, persistent=True)       # Save - backward compat
        self.plate_C_candidates = BufferDict({}, persistent=True)  # Save - candidate pool indices
        self.plate_U_polish = BufferDict({}, persistent=True)      # Save - eigenvector matrix
        self.scaling = BufferDict({}, persistent=True)         # Save - alpha values
        self.use_svd_mode = {}                                 # Track whether SVD was used per adapter
        
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs
        
        # Store SRHT seed for deterministic reconstruction
        self.srht_seed_in = 42
        
        # Store base layer info
        base = self.get_base_layer()
        if isinstance(base, nn.Linear):
            self.in_features, self.out_features = base.in_features, base.out_features
        elif isinstance(base, Conv1D):
            self.out_features, self.in_features = base.weight.shape
        
        self._is_conv1d = isinstance(base, Conv1D)
    
    def _ensure_q_buffer(self, adapter_name: str) -> None:
        """
        Recompute Q_in buffer from saved buffers.
        
        Supports two modes:
        1. SVD mode: Q_in stored directly (for small dimensions)
        2. Polished SRHT: Reconstruct from C_candidates + U_polish (mandatory for SRHT mode)
        """
        # If Q already cached, skip
        if adapter_name in self.plate_Q_in:
            return
        
        # Get device early (needed for both SVD and SRHT modes)
        # Try to get from S_in buffer first, otherwise use model parameters
        if adapter_name in self.plate_S_in:
            device = self.plate_S_in[adapter_name].device
        else:
            device = next(self.parameters()).device
        
        # === SVD MODE: Q_in stored directly ===
        if adapter_name in self.plate_Q_in_svd:
            Q_in = self.plate_Q_in_svd[adapter_name]
            # Ensure Q_in is on correct device (already in model dtype)
            if Q_in.device != device:
                Q_in = Q_in.to(device)
            self.plate_Q_in[adapter_name] = Q_in
            return
            
        # Use stored dimensions (safe with DeepSpeed ZeRO-3)
        d_in = self.in_features
        
        # Get device and dtype from stored buffers
        S_in = self.plate_S_in[adapter_name]
        U_polish = self.plate_U_polish[adapter_name]
        device = S_in.device  # Update device (should be same as above, but ensure consistency)
        compute_dtype = U_polish.dtype  # Match polishing dtype

        # Deterministic SRHT
        H_in = SRHT(d_in, self.srht_seed_in, device=device, dtype=compute_dtype)

        with torch.no_grad():
            # Polishing is mandatory - adapters must have C_candidates and U_polish
            if adapter_name not in self.plate_C_candidates or adapter_name not in self.plate_U_polish:
                missing = []
                if adapter_name not in self.plate_C_candidates:
                    missing.append("plate_C_candidates")
                if adapter_name not in self.plate_U_polish:
                    missing.append("plate_U_polish")
                raise ValueError(
                    f"[PLATE] Adapter '{adapter_name}' is missing required buffers: {', '.join(missing)}. "
                    f"This typically happens when:\n"
                    f"  1. Loading a checkpoint saved without these buffers (old PLATE version)\n"
                    f"  2. Manually modifying the state_dict\n"
                    f"  3. The adapter was created with inference_mode=True but buffers weren't loaded\n"
                    f"Solution: Re-create the adapter from scratch or ensure checkpoint contains these buffers."
                )
            
            C = self.plate_C_candidates[adapter_name].to(device)  # [m]
            U = self.plate_U_polish[adapter_name].to(device)      # [m, k_in]
            
            # Ensure U is in model dtype (in case it was saved in float32 from old checkpoint)
            # compute_dtype comes from U_polish.dtype, but we want to use model's dtype
            U = U.to(compute_dtype)
            
            # Reconstruct: Q_in = H @ E[C] @ U
            # Keep in model dtype to save memory - only convert to float32 for QR
            H_C = _srht_columns(H_in, C)  # [d_in, m] - already in compute_dtype from SRHT init
            Q_in = H_C @ U  # [d_in, k_in] - both in compute_dtype
            
            # Cleanup H_C immediately after use
            del H_C
            
            # Re-orthonormalize (QR needs float32 for numerical stability, then convert back)
            Q_in_fp32, _ = qr_canonical(Q_in.to(torch.float32))
            Q_in = Q_in_fp32.to(compute_dtype)  # Convert back to model dtype to save memory
            
            # Cleanup float32 copy immediately
            del Q_in_fp32

        # Cache Q_in on GPU (needed for forward pass)
        self.plate_Q_in[adapter_name] = Q_in.to(device)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Override to handle loading BufferDict entries.
        
        When loading from checkpoint, this loads:
        - Q_in components (Q_in_svd, S_in, C_candidates, U_polish)
        - Adapter weights (plate_A, plate_B)
        - Scaling factors (self.scaling)
        """
        # Extract BufferDict entries from state_dict
        buffer_dict_names = ['plate_Q_in', 'plate_Q_in_svd', 'plate_S_in', 'plate_C_candidates', 'plate_U_polish', 'scaling']
        
        for buffer_name in buffer_dict_names:
            buffer_dict = getattr(self, buffer_name)
            buffer_prefix = prefix + buffer_name + '.'
            for key in list(state_dict.keys()):
                if key.startswith(buffer_prefix):
                    adapter_name = key[len(buffer_prefix):]
                    buffer_dict[adapter_name] = state_dict.pop(key)
        
        # Resize plate_A and plate_B modules
        for module_name in ['plate_A', 'plate_B']:
            module_dict = getattr(self, module_name)
            module_prefix = prefix + module_name + '.'
            for key in list(state_dict.keys()):
                if key.startswith(module_prefix) and key.endswith('.weight'):
                    relative_key = key[len(module_prefix):]
                    adapter_name = relative_key.split('.')[0]
                    
                    if adapter_name in module_dict:
                        incoming_weight = state_dict[key]
                        # Handle both 2D and 1D (flattened) weights
                        if incoming_weight.dim() == 2:
                            out_features, in_features = incoming_weight.shape
                        elif incoming_weight.dim() == 1:
                            # Skip 1D weights (likely ZeRO-3 placeholder or bias)
                            continue
                        else:
                            continue
                        
                        current_module = module_dict[adapter_name]
                        
                        if current_module.weight.shape != incoming_weight.shape:
                            new_module = nn.Linear(in_features, out_features, bias=False)
                            new_module.weight.data = incoming_weight.clone()
                            # plate_B should always be frozen
                            if module_name == 'plate_B':
                                new_module.weight.requires_grad = False
                            module_dict[adapter_name] = new_module
                            state_dict.pop(key)
        
        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        plate_alpha: float,
        plate_dropout: float,
        col_tau: float = 0.90,
        max_rank: int = 512,
        init_lora_weights: bool = True,
        inference_mode: bool = False,
        num_hutch_probes: int = 4,
        pool_multiple: int = 4,
        max_candidates: int = 8192,
    ):
        """Initialize PLATE adapter parameters.
        
        Args:
            adapter_name: Name of the adapter
            r: Trainable neurons
            plate_alpha: Scaling factor for adapter output
            plate_dropout: Dropout rate
            col_tau: Threshold for input basis selection (must be in [0, 1])
            max_rank: Maximum rank for input basis
            init_lora_weights: Whether to initialize A to zeros
            inference_mode: Whether loading from checkpoint
            num_hutch_probes: Number of Hutchinson probes (4 optimal from experiments)
            pool_multiple: Candidate pool size multiplier (4 optimal from experiments)
            max_candidates: Maximum candidate pool size (default: 8192)
                           Prevents expensive polishing when k is large (keeps <1s)
        """
        
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        
        # Validate col_tau is in [0, 1]
        if not 0 <= col_tau <= 1:
            raise ValueError(
                f"`col_tau` must be in [0, 1], got {col_tau}. "
                f"col_tau represents the energy threshold for column space selection."
            )
        
        # Store config
        self.r[adapter_name] = r
        self.plate_alpha[adapter_name] = plate_alpha
        self.col_tau[adapter_name] = col_tau
        
        # Dropout
        if plate_dropout > 0.0:
            plate_dropout_layer = nn.Dropout(p=plate_dropout)
        else:
            plate_dropout_layer = nn.Identity()
        self.plate_dropout[adapter_name] = plate_dropout_layer
        
        # Get base layer
        base = self.get_base_layer()
        
        # Skip expensive computation if loading from checkpoint
        if inference_mode:
            # Loading from checkpoint:
            # - Create dummy 1×1 Linear modules for A and B (required for ModuleDict resize logic)
            # - Q_in buffers are populated from state_dict
            # - scaling is set from plate_alpha (config value, not in state_dict)
            self.plate_A[adapter_name] = nn.Linear(1, 1, bias=False)
            plate_B_dummy = nn.Linear(1, 1, bias=False)
            plate_B_dummy.weight.requires_grad = False  # plate_B is always frozen
            self.plate_B[adapter_name] = plate_B_dummy
            # Set scaling from config (not saved in checkpoint, but plate_alpha IS in config)
            self.scaling[adapter_name] = torch.tensor(plate_alpha, dtype=torch.float32)
        else:
            # New adapter - compute Q_in and B
            
            with gather_params_ctx(base.weight):
                if isinstance(base, Conv1D):
                    W = base.weight.t().contiguous()
                elif isinstance(base, nn.Linear):
                    W = base.weight
                else:
                    raise ValueError(f"Unsupported base layer: {type(base)}")
                out, in_ = W.shape
                
                # Auto-adjust rank if it's too large for this layer
                # PLATE requires r < out because it needs (out - r) frozen neurons for Q_in computation
                # Use 90% rule: allow up to 90% trainable, keep 10% frozen for Q_in
                max_r = int(MAX_TRAINABLE_FRACTION * out)
                r_effective = min(r, max_r)
                
                # 1. FIRST: Select trainable output neurons using similarity
                # Select output neurons with highest average cosine similarity (these will be TRAINABLE)
                trainable_neuron_indices = select_output_neurons(W, topk=r_effective)
                
                # 2. SECOND: Extract FROZEN neuron rows (the complement) for Q_in computation
                # W_frozen contains the neurons that will NOT be updated (frozen/preserved)
                # This ensures Q_in is orthogonal to the frozen subspace only
                all_indices = torch.arange(out, device=W.device)
                frozen_mask = torch.ones(out, dtype=torch.bool, device=W.device)
                frozen_mask[trainable_neuron_indices] = False
                frozen_neuron_indices = all_indices[frozen_mask]  # [d_out - r]
                W_frozen = W[frozen_neuron_indices, :]  # [d_out-r, d_in]
                
                # 3. THIRD: Compute Q_in using ONLY frozen neurons
                H_in, C_candidates, U_polish, Q_in = build_q_in_from_weight(
                    W_frozen,                         # Use ONLY frozen neurons
                    col_tau=col_tau, 
                    max_rank=max_rank,
                    num_probes=num_hutch_probes, 
                    seed=42,
                    pool_multiple=pool_multiple,
                    max_candidates=max_candidates,
                )
                k_in = Q_in.shape[1]
                
                # Store buffers for reconstruction
                # Check if we used SVD mode (H_in is None)
                if H_in is None:
                    # SVD MODE: Store Q_in directly (no SRHT reconstruction needed)
                    self.use_svd_mode[adapter_name] = True
                    # Store Q_in in model dtype (already converted to out_dtype)
                    self.plate_Q_in_svd[adapter_name] = Q_in  # Store in model dtype
                    self.plate_S_in[adapter_name] = C_candidates  # Dummy for backward compat
                else:
                    # SRHT MODE: Store components for reconstruction
                    # Note: We don't store H_in itself since SRHT is deterministic
                    # and cheap to recreate from self.srht_seed_in when needed
                    self.use_svd_mode[adapter_name] = False
                    self.plate_C_candidates[adapter_name] = C_candidates  # Candidate pool
                    self.plate_U_polish[adapter_name] = U_polish  # Store in model dtype (already converted)
                    self.plate_S_in[adapter_name] = C_candidates[:k_in]   # Backward compat
                    # Cache Q_in on GPU (same as before DeepSpeed changes)
                    # Q_in is already computed and in model dtype, so cache it immediately
                    self.plate_Q_in[adapter_name] = Q_in
                
                # Validation: Ensure dimensions are correct
                if Q_in.shape[0] != in_:
                    raise ValueError(
                        f"[PLATE] Q_in dimension mismatch: first dim is {Q_in.shape[0]} "
                        f"but expected input dim {in_}. This indicates a bug in Q_in computation."
                    )
                if Q_in.shape[1] != k_in:
                    raise ValueError(
                        f"[PLATE] Q_in dimension mismatch: second dim is {Q_in.shape[1]} "
                        f"but expected k_in={k_in}. This indicates a bug in Q_in computation."
                    )
                if W_frozen.shape[0] != out - r_effective:
                    raise ValueError(
                        f"[PLATE] W_frozen row count mismatch: has {W_frozen.shape[0]} rows "
                        f"but expected {out - r_effective} (out={out}, r_effective={r_effective}). "
                        f"This indicates a bug in neuron selection."
                    )
                if W_frozen.shape[1] != in_:
                    raise ValueError(
                        f"[PLATE] W_frozen column count mismatch: has {W_frozen.shape[1]} cols "
                        f"but expected input dim {in_}. This indicates a bug in weight slicing."
                    )
                
                # 4. Create adapter modules
                # Match base layer dtype for consistency and memory efficiency
                base_dtype = W.dtype
                
                # plate_A: trainable [r_effective, k_in]
                plate_A_module = nn.Linear(k_in, r_effective, bias=False)
                if init_lora_weights:
                    nn.init.zeros_(plate_A_module.weight)
                # Convert to match base layer dtype (e.g., bf16, fp16)
                plate_A_module = plate_A_module.to(dtype=base_dtype, device=W.device)
                plate_A_module.weight.requires_grad = True
                self.plate_A[adapter_name] = plate_A_module
                
                # plate_B: frozen [out, r_effective] selection matrix
                # B selects which output neurons receive updates (trainable neurons)
                plate_B_module = nn.Linear(r_effective, out, bias=False)
                with torch.no_grad():
                    plate_B_module.weight.data.copy_(
                        torch.eye(out, device=W.device, dtype=base_dtype)[:, trainable_neuron_indices]
                    )
                # Convert to match base layer dtype
                plate_B_module = plate_B_module.to(dtype=base_dtype, device=W.device)
                plate_B_module.weight.requires_grad = False
                self.plate_B[adapter_name] = plate_B_module
                
                # Use plate_alpha directly as the scaling factor
                self.scaling[adapter_name] = torch.tensor(plate_alpha, dtype=torch.float32)
        
        self.set_adapter([adapter_name])

    def _get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute ΔW for merging: ΔW = B @ A @ Q_in^T
        """
        plate_A_module = self.plate_A[adapter_name]
        plate_B_module = self.plate_B[adapter_name]
        
        if not isinstance(plate_A_module, nn.Linear) or not isinstance(plate_B_module, nn.Linear):
            raise ValueError("Expected nn.Linear modules")
        
        plate_A = plate_A_module.weight  # [r, k_in]
        plate_B = plate_B_module.weight  # [d_out, r]
        
        # Ensure Q_in is available
        if adapter_name not in self.plate_Q_in:
            if adapter_name in self.plate_S_in:
                self._ensure_q_buffer(adapter_name)
            else:
                raise ValueError(f"Cannot merge adapter '{adapter_name}': Q_in not found")
        
        Q_in = self.plate_Q_in[adapter_name]  # [d_in, k_in]
        
        # Ensure all tensors are on the same device (DeepSpeed multi-GPU compatibility)
        device = plate_A.device
        dtype = plate_A.dtype
        
        # Move Q_in and plate_B to same device if needed
        if Q_in.device != device:
            Q_in = Q_in.to(device=device)
        if plate_B.device != device:
            plate_B = plate_B.to(device=device)
        
        # Cast to fp32 only on CPU with fp16/bf16 (for performance, following LoRA approach)
        # Some CPUs have slow bf16/fp16 matmuls
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        
        if cast_to_fp32:
            plate_A_fp32 = plate_A.float()
            plate_B_fp32 = plate_B.float()
            Q_in_fp32 = Q_in.float()
        else:
            plate_A_fp32 = plate_A
            plate_B_fp32 = plate_B
            Q_in_fp32 = Q_in
        
        # Compute: B @ A @ Q_in^T = [d_out, r] @ [r, k_in] @ [k_in, d_in] = [d_out, d_in]
        temp = plate_B_fp32 @ plate_A_fp32  # [d_out, k_in]
        dW = temp @ Q_in_fp32.T  # [d_out, d_in]
        
        if cast_to_fp32:
            dW = dW.to(dtype)
        
        if self._is_conv1d:
            return dW.t().contiguous()
        return dW

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        PLATE forward pass: y = base(x) + scaling * B(A(Q_in^T(x)))
        
        Note: Gradient checkpointing is not supported. The Q_in projection (a fixed buffer)
        breaks gradient flow when PyTorch recomputes activations during checkpointing.
        Use get_plate_model() which automatically disables gradient checkpointing.
        """
        previous_dtype = x.dtype
        
        # Get base output
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Module):
            result = base_layer(x, *args, **kwargs)
        else:
            raise ValueError(f"Expected nn.Module but got {type(base_layer)}")
        
        # Apply adapters if not disabled
        if not self._disable_adapters:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.plate_A.keys():
                    continue
                
                # Ensure Q_in is available
                self._ensure_q_buffer(active_adapter)
                
                plate_A_module = self.plate_A[active_adapter]
                plate_B_module = self.plate_B[active_adapter]
                
                if not isinstance(plate_A_module, nn.Linear) or not isinstance(plate_B_module, nn.Linear):
                    continue
                
                dropout_layer = self.plate_dropout[active_adapter]
                Q_in = self.plate_Q_in[active_adapter]
                
                # Forward: x @ Q_in @ A^T @ B^T
                # Extract weights (avoids DeepSpeed ZeRO-3 all_gather OOM for large PLATE matrices)
                plate_A = plate_A_module.weight
                plate_B = plate_B_module.weight
                
                x_work = dropout_layer(x).to(plate_A.dtype)  # [..., d_in]
                
                # 1. Project to constrained subspace: Z = x @ Q_in  [..., d_in] @ [d_in, k_in] = [..., k_in]
                # Only convert Q_in if dtype/device actually differs (avoids unnecessary copy)
                if Q_in.device != x_work.device or Q_in.dtype != x_work.dtype:
                    Q_in = Q_in.to(device=x_work.device, dtype=x_work.dtype)
                Z = torch.nn.functional.linear(x_work, Q_in.T)
                
                # 2. Apply trainable A: U = Z @ A^T  [..., k_in] @ [k_in, r] = [..., r]
                # Only convert if dtype/device differs (avoids unnecessary copy)
                if plate_A.device != Z.device or plate_A.dtype != Z.dtype:
                    plate_A = plate_A.to(device=Z.device, dtype=Z.dtype)
                U = torch.nn.functional.linear(Z, plate_A)
                
                # 3. Apply frozen B: y = U @ B^T  [..., r] @ [r, d_out] = [..., d_out]
                # Only convert if dtype/device differs (avoids unnecessary copy)
                if plate_B.device != U.device or plate_B.dtype != U.dtype:
                    plate_B = plate_B.to(device=U.device, dtype=U.dtype)
                adapter_out = torch.nn.functional.linear(U, plate_B)
                
                # Scale and add
                adapter_out = adapter_out * self.scaling[active_adapter]
                result = result + adapter_out.to(previous_dtype)
        
        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge adapter weights into base layer."""
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        
        for adapter_name in adapter_names:
            if adapter_name not in self.plate_A.keys():
                continue
            if adapter_name in self.merged_adapters:
                continue
            
            plate_A_module = self.plate_A[adapter_name]
            if not isinstance(plate_A_module, nn.Linear):
                continue
            
            # Compute ΔW
            dW = self._get_delta_weight(adapter_name)
            
            # Ensure scaling is on same device as dW (DeepSpeed multi-GPU compatibility)
            scaling = self.scaling[adapter_name]
            if isinstance(scaling, torch.Tensor):
                scaling = scaling.to(device=dW.device)
            dW = dW * scaling
            
            # Get base weight
            base_layer = self.get_base_layer()
            if not isinstance(base_layer, (nn.Linear, Conv1D)):
                continue
            base_weight = base_layer.weight
            
            # Cast and merge
            dW = dW.to(dtype=base_weight.dtype, device=base_weight.device)
            
            if safe_merge and not torch.isfinite(dW).all():
                raise ValueError(f"NaNs detected in adapter {adapter_name} weights. Aborting merge.")
            
            base_layer.weight.data += dW
            self.merged_adapters.append(adapter_name)

    def unmerge(self) -> None:
        """Unmerge adapter weights from base layer."""
        for adapter_name in self.merged_adapters[:]:
            if adapter_name not in self.plate_A.keys():
                continue
            
            plate_A_module = self.plate_A[adapter_name]
            if not isinstance(plate_A_module, nn.Linear):
                continue
            
            # Compute ΔW
            dW = self._get_delta_weight(adapter_name) * self.scaling[adapter_name]
            
            # Get base weight
            base_layer = self.get_base_layer()
            if not isinstance(base_layer, (nn.Linear, Conv1D)):
                continue
            base_weight = base_layer.weight
            
            # Cast and unmerge
            dW = dW.to(dtype=base_weight.dtype, device=base_weight.device)
            base_layer.weight.data -= dW
            
            self.merged_adapters.remove(adapter_name)

