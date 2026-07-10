import gzip
import os
import struct
import unittest

import numpy as np
# import sidechainnet as scn
import torch
import trimesh
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, Dataset

from models import *
from trainers import *
from utils.constraints import SimpleConstraintProjector

# Hardcoded atom order used by SidechainNet
# This is used to extract backbone fragments from protein structures

SCN_ATOM_ORDER = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CZ",
    "CZ2",
    "CZ3",
    "CH2",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "SD",
    "SG",
]


def _find_mnist_raw_dir(root):
    candidates = [
        os.path.join(root, "MNIST", "raw"),
        os.path.join(root, "raw"),
        root,
    ]
    for raw_dir in candidates:
        if os.path.isfile(os.path.join(raw_dir, "train-images-idx3-ubyte")) or os.path.isfile(
            os.path.join(raw_dir, "train-images-idx3-ubyte.gz")
        ):
            return raw_dir
    raise FileNotFoundError(
        "MNIST raw IDX files were not found. Expected files under one of: "
        + ", ".join(candidates)
    )


def _read_idx_bytes(path):
    if os.path.isfile(path):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:
            return f.read()
    gz_path = f"{path}.gz"
    if os.path.isfile(gz_path):
        with gzip.open(gz_path, "rb") as f:
            return f.read()
    raise FileNotFoundError(f"Missing MNIST IDX file: {path} or {gz_path}")


def _load_mnist_tensors(root="./data", train=True):
    """Load MNIST from local IDX files without importing torchvision."""
    raw_dir = _find_mnist_raw_dir(root)
    split = "train" if train else "t10k"
    images_blob = _read_idx_bytes(os.path.join(raw_dir, f"{split}-images-idx3-ubyte"))
    labels_blob = _read_idx_bytes(os.path.join(raw_dir, f"{split}-labels-idx1-ubyte"))

    image_magic, num_images, rows, cols = struct.unpack(">IIII", images_blob[:16])
    label_magic, num_labels = struct.unpack(">II", labels_blob[:8])
    if image_magic != 2051:
        raise ValueError(f"Invalid MNIST image magic number {image_magic} in {raw_dir}")
    if label_magic != 2049:
        raise ValueError(f"Invalid MNIST label magic number {label_magic} in {raw_dir}")
    if num_images != num_labels:
        raise ValueError(
            f"MNIST image/label count mismatch: {num_images} images, {num_labels} labels"
        )

    images = np.frombuffer(images_blob, dtype=np.uint8, offset=16).copy()
    labels = np.frombuffer(labels_blob, dtype=np.uint8, offset=8).copy()
    images = images.reshape(num_images, rows, cols)
    return torch.from_numpy(images), torch.from_numpy(labels).to(torch.long)


def is_valid_coord(coord):
    return not (np.any(np.isnan(coord)) or np.allclose(coord, 0))


def extract_backbone_fragments(
    data, fragment_length=5, max_data_length=10000, atol=1e-8
):
    """Vectorized extraction of backbone fragments.

    This avoids Python-level nested loops and per-atom list comprehensions by
    using NumPy rolling-window operations per-protein. It returns an array of
    shape (N_fragments, fragment_length, 3, 3).
    """
    out = []
    remaining = max_data_length
    f = fragment_length

    for protein in data:
        coords = np.asarray(protein.coords, dtype=np.float32, order="C")  # (L, A, 3)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            continue
        L, A, _ = coords.shape
        if L < f or A < 3:
            continue

        # Backbone slice once
        bb = coords[:, 0:3, :]  # (L, 3, 3)

        # Per-residue validity: no NaNs and not all ~0 (matching is_valid_coord)
        finite = np.isfinite(bb).all(axis=(1, 2))  # (L,)
        nonzeroish = ~(np.all(np.abs(bb) <= atol, axis=(1, 2)))  # (L,)
        good = finite & nonzeroish  # (L,)

        # Rolling AND via cumulative sum (treat True=1): window valid if sum==f
        g = good.astype(np.int32)
        cs = np.concatenate(([0], np.cumsum(g)))  # (L+1,)
        win_sum = cs[f:] - cs[:-f]  # (L-f+1,)
        starts = np.nonzero(win_sum == f)[0]
        if starts.size == 0:
            continue

        # Build windows only for valid starts
        take = min(remaining, starts.size)
        # Sliding view along residue axis (view, no copy)
        win = sliding_window_view(bb, f, axis=0)  # expected (L-f+1, f, 3, 3)
        sel = win[
            starts[:take]
        ]  # (take, f, 3, 3) OR (take, 3, 3, f) depending on input ordering

        # Robustness: some coordinate arrays may have axes permuted; ensure
        # we always return fragments shaped (take, f, 3, 3).
        if sel.ndim == 4 and sel.shape[1] != f and sel.shape[3] == f:
            # detected shape (take, 3, 3, f) -> transpose to (take, f, 3, 3)
            sel = np.transpose(sel, (0, 3, 1, 2))

        out.append(sel)

        remaining -= take
        if remaining <= 0:
            break

    if not out:
        return np.empty((0, fragment_length, 3, 3), dtype=np.float32)
    return np.concatenate(out, axis=0)

import numpy as np
import torch
from torch.utils.data import Dataset


class AnalyticNormalSpaceProjectorFast:
    """Fast, vectorized analytic normal-space projector.

    This class implements the optimized algorithm: build small per-constraint
    gradients (m x L x 3 x 3), form the Gram (m x m), solve for the
    Lagrange multipliers and compute the projected noise without ever
    forming a D x D matrix. It is intended to be a drop-in fast backend.
    """

    def __init__(self, noise_level=0.01, ridge=1e-8):
        self.noise_level = float(noise_level)
        self.ridge = float(ridge)
        # Keep physical constants handy for external inspection (non-critical)
        self.bond_lengths = {
            "N-CA": 1.46,
            "CA-C": 1.52,
            "C-N+1": 1.33,
        }
        self.bond_angles = {"N-CA-C": 110.0, "CA-C-N+1": 116.0}

    def _wrap_to_pi(self, x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def _angle(self, a, b, c):
        v1 = a - b
        v2 = c - b
        v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-12)
        v2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-12)
        cross = torch.cross(v1, v2, dim=-1).norm(dim=-1)
        dot = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.atan2(cross, dot)

    def _dihedral(self, a, b, c, d):
        b0 = b - a
        b1 = c - b
        b2 = d - c
        b1n = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-12)
        v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
        w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
        x = (v * w).sum(dim=-1)
        y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
        return torch.atan2(y, x)

    def _constraint_vector(self, frag: torch.Tensor):
        """
        Build residuals c(x) for a single fragment (L,3,3).
        Returns shape (m,), all ops differentiable wrt frag.
        """
        assert frag.dim() == 3 and frag.shape[1:] == (3, 3), "Expect (L,3,3)"
        # ensure float & grad through this function; DO NOT detach here
        frag = frag  # .to(dtype=torch.float64)

        L = frag.shape[0]
        N, CA, C = frag[:, 0, :], frag[:, 1, :], frag[:, 2, :]

        # constants that don't break graph
        one46 = torch.tensor(1.46, dtype=frag.dtype, device=frag.device)
        one52 = torch.tensor(1.52, dtype=frag.dtype, device=frag.device)
        one33 = torch.tensor(1.33, dtype=frag.dtype, device=frag.device)
        deg = torch.tensor(torch.pi / 180.0, dtype=frag.dtype, device=frag.device)
        ang_NCAC = torch.tensor(110.0, dtype=frag.dtype, device=frag.device) * deg
        ang_CACN = torch.tensor(116.0, dtype=frag.dtype, device=frag.device) * deg
        ang_CNCA = torch.tensor(121.0, dtype=frag.dtype, device=frag.device) * deg
        pi = torch.tensor(torch.pi, dtype=frag.dtype, device=frag.device)

        vec = []

        # bond lengths
        vec.append((N - CA).norm(dim=-1) - one46)
        vec.append((CA - C).norm(dim=-1) - one52)
        if L > 1:
            vec.append((C[:-1] - N[1:]).norm(dim=-1) - one33)

        # bond angles (atan2 form)
        vec.append(self._angle(N, CA, C) - ang_NCAC)
        if L > 1:
            vec.append(self._angle(CA[:-1], C[:-1], N[1:]) - ang_CACN)
            vec.append(self._angle(C[:-1], N[1:], CA[1:]) - ang_CNCA)

        # peptide planarity: omega ≈ π
        if L > 1:
            omega = self._dihedral(CA[:-1], C[:-1], N[1:], CA[1:])
            vec.append(self._wrap_to_pi(omega - pi))

        return torch.cat([t.reshape(-1) for t in vec], dim=0)  # (m,)

    def _build_constraint_grads(self, frag: torch.Tensor):
        """
        Returns grads with shape (m, L, 3, 3) via jacobian, robust to PyTorch versions.
        """
        frag_req = frag.to(dtype=torch.float64).detach().clone().requires_grad_(True)

        def f(z):
            # z: (L,3,3) with grad
            return self._constraint_vector(z)

        # jac: (m, L, 3, 3)
        jac = torch.autograd.functional.jacobian(
            f, frag_req, create_graph=False, vectorize=True
        )
        # Ensure contiguous and correct dtype
        return jac.to(dtype=frag.dtype).contiguous()

    def _project_noise(self, frag: torch.Tensor, eps: torch.Tensor | None = None):
        """
        EXACT normal-space Gaussian noise:
        eps_normal = N(x) z,  z ~ N(0, sigma^2 I_k),
        where N(x) has orthonormal columns spanning the normal space span{∇c_i(x)}.

        Notes:
        - We ignore the input `eps` (kept only for API compatibility).
        - Uses QR on the flattened constraint gradients to get an orthonormal basis.
        - Handles rank deficiency via a threshold on |diag(R)|.
        """
        grads = self._build_constraint_grads(frag)  # (m, L, 3, 3)
        m, L, A, D = grads.shape
        device, dtype = grads.device, grads.dtype

        G = grads.reshape(m, -1)  # (m, Dflat) where Dflat = L*3*3

        # Build orthonormal basis for span(G^T) using reduced QR
        # Q: (Dflat, m), R: (m, m)
        Q, R = torch.linalg.qr(G.T, mode="reduced")

        # Determine numerical rank k (dimension of normal space actually spanned)
        diag = R.diagonal()  # (m,)
        # Threshold: relative to largest diagonal entry (scale-invariant)
        # You can tighten/loosen 1e-10 depending on dtype/noise.
        tol = (diag.abs().max() * 1e-10) if diag.numel() > 0 else torch.tensor(0.0, device=device, dtype=dtype)
        k = int((diag.abs() > tol).sum().item())

        if k == 0:
            return torch.zeros_like(frag)

        Nbasis = Q[:, :k]  # (Dflat, k) orthonormal columns

        # Sample z ~ N(0, sigma^2 I_k)
        z = torch.randn((k,), device=device, dtype=dtype) * self.noise_level

        eps_flat = Nbasis @ z  # (Dflat,)
        eps_normal = eps_flat.reshape(L, 3, 3)
        return eps_normal

    def add_noise(self, frag: torch.Tensor):
        frag = frag.clone()
        eps_normal = self._project_noise(frag, eps=None)
        return frag + eps_normal

    def _build_constraint_residuals(self, frag: torch.Tensor):
        # IMPORTANT: no no_grad here—let autograd see the ops when needed.
        return self._constraint_vector(frag)

    def project(self, frag: torch.Tensor):
        """Linearized Gauss-Newton projection for fragment(s).

        Accepts frags of shape (L,3,3) or (B,L,3,3). Returns projected fragments
        with the same shape.
        """
        single = False
        if frag.dim() == 3:
            frags = frag.unsqueeze(0)
            single = True
        else:
            frags = frag

        B, L, A, D = frags.shape
        device = frags.device
        dtype = frags.dtype

        out = torch.empty_like(frags)
        # process in chunks
        chunk_size = 64
        for start in range(0, B, chunk_size):
            end = min(B, start + chunk_size)
            chunk = frags[start:end]
            b = chunk.shape[0]
            # build grads: (b, m, L, 3, 3)
            grads = []
            res_list = []
            for i in range(b):
                g = self._build_constraint_grads(chunk[i])
                grads.append(g)
                res_list.append(self._build_constraint_residuals(chunk[i]))
            grads = torch.stack(grads, dim=0)
            res = torch.stack(res_list, dim=0)  # (b, m)

            m = grads.shape[1]
            Gram = torch.einsum("bmrad,bnrad->bmn", grads, grads)
            if self.ridge > 0:
                Gram = Gram + self.ridge * torch.eye(
                    m, device=device, dtype=dtype
                ).unsqueeze(0)

            try:
                alpha = torch.linalg.solve(Gram, res.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                pinv = torch.linalg.pinv(Gram)
                alpha = torch.matmul(pinv, res.unsqueeze(-1)).squeeze(-1)

            eps_proj = torch.einsum("bmrad,bm->brad", grads, alpha)
            out[start:end] = chunk - eps_proj

        if single:
            return out.squeeze(0)
        return out

    def add_noise_batched(self, frags: torch.Tensor, generator=None):
        B = frags.shape[0]
        out = torch.empty_like(frags)
        chunk_size = 64
        for start in range(0, B, chunk_size):
            end = min(B, start + chunk_size)
            chunk = frags[start:end]
            for i in range(chunk.shape[0]):
                eps_normal = self._project_noise(chunk[i], eps=None)
                out[start + i] = chunk[i] + eps_normal
        return out


class AnalyticNormalSpaceProjector:
    """Compatibility wrapper. Keeps the original constructor/API but forwards
    to the fast implementation by default. If you need the legacy optimizer-based
    projector you can instantiate with mode='legacy' (currently falls back to
    the fast implementation but the name is preserved for API compatibility).
    """

    def __init__(self, noise_level=0.01, mode="fast", **kwargs):
        self.mode = mode
        self.noise_level = float(noise_level)
        if mode == "fast":
            self._impl = AnalyticNormalSpaceProjectorFast(
                noise_level=noise_level, **kwargs
            )
        else:
            # legacy path: fallback to fast implementation so callers don't break.
            self._impl = AnalyticNormalSpaceProjectorFast(
                noise_level=noise_level, **kwargs
            )

        # expose some useful attributes for downstream code
        self.bond_lengths = getattr(self._impl, "bond_lengths", {})
        self.bond_angles = getattr(self._impl, "bond_angles", {})

    def add_noise(self, frag):
        return self._impl.add_noise(frag)

    def add_noise_batched(self, frags, generator=None):
        return self._impl.add_noise_batched(frags, generator=generator)


def _unnormalize_backbone_fragment(frag, mean_caca: float = 3.8):
    """
    Reverse the scaling step of normalization by multiplying by mean Cα-Cα distance.
    
    This undoes only the scaling component (target_caca=1.0 normalization):
    coordinates are multiplied by ~3.8 Å to restore physical distance scales.
    Rotation and centering are NOT reversed (but angles/dihedrals are rotation-invariant).
    
    Args:
        frag: (L, 3, 3) array or tensor (normalized, unit scale)
        mean_caca: the mean Cα-Cα distance in Ångströms used during normalization (~3.8)
    
    Returns:
        Fragment scaled to Ångströms (same shape as input)
    """
    return frag * mean_caca


def _normalize_backbone_fragment(
    frag: np.ndarray,
    target_caca: float = 1.0,  # set to 1.0 for unitized scale; or 3.8 to keep Å units
    center: str = "com",  # "com" (center of mass) or "ca0" (first CA)
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize a backbone fragment with atoms ordered per residue as [N, CA, C].
    frag: (L, 3, 3) array (res, atom, xyz)

    Steps:
    1) Center (COM or first CA).
    2) Rescale by mean Cα–Cα distance to `target_caca`.
    3) Fix global rotation:
        - u1 = normalize(CA0 - N0)  -> align to +x
        - u2 = normalize( (C0 - CA0) projected orthogonal to u1 ) -> align to +y
        - u3 = u1 × u2 -> align to +z
    Result is rigid-body normalized and scale-consistent.
    """
    frag = frag  # .astype(np.float64).copy()
    L = frag.shape[0]
    assert frag.shape == (
        L,
        3,
        3,
    ), "Expected (L, 3, 3) with atoms [N, CA, C] per residue."

    # -----------------------
    # (1) Centering
    # -----------------------
    if center == "com":
        com = frag.reshape(-1, 3).mean(axis=0)
        frag -= com
    elif center == "ca0":
        frag -= frag[0, 1]  # first residue CA
    else:
        raise ValueError("center must be 'com' or 'ca0'")

    # -----------------------
    # (2) Scale by mean Cα–Cα to target
    # -----------------------
    cas = frag[:, 1, :]  # (L, 3)
    if L > 1:
        diffs = cas[1:] - cas[:-1]  # (L-1, 3)
        dists = np.linalg.norm(diffs, axis=1)
        mean_caca = np.maximum(dists.mean(), eps)
    else:
        # fallback if single residue (should not happen for 10-mer fragments)
        mean_caca = 3.8

    scale = mean_caca / max(target_caca, eps)
    frag /= scale

    # -----------------------
    # (3) Fix rotation via local frame at residue 0
    # -----------------------
    N0 = frag[0, 0]
    CA0 = frag[0, 1]
    C0 = frag[0, 2]
    u1 = CA0 - N0
    n1 = np.linalg.norm(u1)
    if n1 < eps:
        # fallback: try next residue if degenerate
        if L > 1:
            u1 = frag[1, 1] - frag[1, 0]
            n1 = np.linalg.norm(u1)
        if n1 < eps:
            # give up on rotation if degenerate; return centered/scaled
            return frag
    u1 /= n1

    # in-plane direction approximately along peptide plane, orthogonal to u1
    v = C0 - CA0
    v_perp = v - (u1 * np.dot(u1, v))
    n2 = np.linalg.norm(v_perp)
    if n2 < eps and L > 1:
        # fallback: use next residue geometry
        v = frag[1, 2] - frag[1, 1]  # C1 - CA1
        v_perp = v - (u1 * np.dot(u1, v))
        n2 = np.linalg.norm(v_perp)
    if n2 < eps:
        # final fallback: pick any orthogonal vector
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(u1, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        v_perp = tmp - u1 * np.dot(u1, tmp)
        n2 = np.linalg.norm(v_perp)

    u2 = v_perp / max(n2, eps)
    u3 = np.cross(u1, u2)
    n3 = np.linalg.norm(u3)
    if n3 < eps:
        # if degenerate, rebuild u2 using a different tmp
        tmp = (
            np.array([0.0, 1.0, 0.0]) if abs(u1[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        )
        v_perp = tmp - u1 * np.dot(u1, tmp)
        u2 = v_perp / max(np.linalg.norm(v_perp), eps)
        u3 = np.cross(u1, u2)
        n3 = np.linalg.norm(u3)
        if n3 < eps:
            return frag  # give up on rotation

    u3 /= n3

    # Orthonormal basis with columns [u1, u2, u3]; rotate so they map to [e1, e2, e3]
    B = np.stack([u1, u2, u3], axis=1)  # (3,3)
    R = B.T  # multiply coordinates by R to align

    frag = frag @ R  # (L, 3, 3)
    return frag


class BackboneFragmentDataset(Dataset):
    def __init__(
        self,
        fragments,
        normalize=True,
        unravel=True,
        noise_level=0.0,
        lifted=True,
        use_noisy=True,  # return the noisy precomputed set by default
        precompute_noise=True,  # do the noise once at init
        batch_size=4096,  # batching for precompute to avoid OOM
        seed=None,  # make noise deterministic if desired
        device="cpu",  # device for projector ops
    ):
        """
        fragments: array-like, shape (N, L, 4, 3) or similar (residues x atoms x xyz)
        normalize: apply fragment normalizer once up front
        unravel:   flatten to 1D per fragment on output
        noise_level: std for noise inside AnalyticNormalSpaceProjector
        lifted:    use the projector (normal-space noise & projection)
        use_noisy: if True and noise_level>0, __getitem__ returns the noisy version
        precompute_noise: compute noisy fragments in __init__ (no per-sample noise later)
        batch_size: chunk size for the batched projector to save memory
        seed:      RNG seed for reproducible noise (only affects precompute)
        device:    device for projector computations ("cpu" or "cuda")
        """
        self.normalize = normalize
        self.unravel = unravel
        self.noise_level = float(noise_level)
        self.lifted = lifted
        self.use_noisy = use_noisy and (self.noise_level > 0) and lifted
        self.precompute_noise = precompute_noise and (self.noise_level > 0) and lifted
        self.device = device

        # 1) Normalize (if requested), stay in NumPy for storage
        if self.normalize:
            fragments = np.stack([self._normalize(frag) for frag in fragments], axis=0)
        else:
            fragments = np.asarray(fragments)

        self.fragments = fragments.astype(np.float32)  # clean, normalized

        # 2) Initialize projector (units already consistent after normalization)
        self.projector = None
        if self.lifted:
            self.projector = AnalyticNormalSpaceProjector(noise_level=self.noise_level)

        # 3) Optionally precompute noisy fragments once
        self.fragments_noisy = None
        if self.precompute_noise:
            if seed is not None:
                # Make torch & numpy noise deterministic for this precompute pass
                rng_state_np = np.random.get_state()
                np.random.seed(seed)
                g = torch.Generator(device=self.device).manual_seed(seed)
            else:
                g = None

            self.fragments_noisy = self._precompute_noisy(self.fragments, batch_size, g)

            if seed is not None:
                # restore numpy RNG
                np.random.set_state(rng_state_np)

    def _normalize(self, frag):
        # Choose 1.0 (unitized) or 3.8 (keep Å scale). Staying with 1.0 here.
        return _normalize_backbone_fragment(frag, target_caca=1.0, center="com")

    @torch.no_grad()
    def _precompute_noisy(self, frags_np, batch_size, generator):
        """Run projector.add_noise_batched in chunks and return a NumPy array."""
        if self.projector is None:
            return None

        N = frags_np.shape[0]
        out = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.from_numpy(frags_np[start:end]).to(self.device)
            # If the projector supports a generator kwarg, pass it; otherwise it will use global RNG.
            try:
                noisy = self.projector.add_noise_batched(batch, generator=generator)
            except TypeError:
                noisy = self.projector.add_noise_batched(batch)
            out.append(noisy.cpu().float().numpy())
        return np.concatenate(out, axis=0)

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        # Select which backing array we expose
        if self.use_noisy and self.fragments_noisy is not None:
            frag_np = self.fragments_noisy[idx]
        else:
            frag_np = self.fragments[idx]

        frag_t = torch.from_numpy(frag_np)

        # Flatten if requested
        if self.unravel:
            frag_t = frag_t.reshape(-1)

        return frag_t

    def get_batch(self, indices):
        # Batch accessor mirrors __getitem__
        if self.use_noisy and self.fragments_noisy is not None:
            frags_np = self.fragments_noisy[indices]
        else:
            frags_np = self.fragments[indices]

        frags = torch.from_numpy(frags_np)

        if self.unravel:
            frags = frags.view(frags.size(0), -1)

        return frags

class BunnyDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        lifted=False,
        noise_level=1,
        bunny_path="data/bunny.obj",
        mode="heat",  # Choose 'texture', 'heat', etc.
        sigma=0.25,  # Only used for 'heat'
        mean_idx=10500,  # Only used for 'heat'
    ):
        self.num_samples = num_samples
        self.lifted = lifted
        self.noise_level = noise_level
        self.bunny_path = bunny_path
        self.mode = mode
        self.sigma = sigma
        self.mean_idx = mean_idx
        self.data = self._get_data()

    def _get_data(self):
        # Load bunny mesh
        bunny = trimesh.load_mesh(self.bunny_path)
        faces = torch.tensor(bunny.faces, dtype=torch.long)

        # Normalize vertices: center and scale to unit ball
        vertices = torch.tensor(bunny.vertices, dtype=torch.float32)
        vertices -= vertices.mean(dim=0)
        scale = vertices.norm(dim=1).max()
        vertices /= scale

        # Diagnostic
        max_radius = vertices.norm(dim=1).max().item()
        print(
            f"[BUNNY DIAGNOSTIC] Max radius from center: {max_radius:.4f}"
        )  # Should be ~1.0

        # Compute consistent vertex normals (PyTorch version)
        def compute_vertex_normals(vertices, faces):
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            face_normals = torch.cross(v1 - v0, v2 - v0)

            vertex_normals = torch.zeros_like(vertices)
            for i in range(3):
                vertex_normals.index_add_(0, faces[:, i], face_normals)

            vertex_normals = F.normalize(vertex_normals, dim=1)
            return vertex_normals

        vertex_normals = compute_vertex_normals(vertices, faces)

        # Update mesh to reflect transformed vertices
        bunny.vertices = vertices.numpy()
        bunny._cache.clear()
        _ = bunny.vertex_normals  # Force recomputation of any cached normals

        self.mesh = bunny  # Now the mesh and samples are aligned

        if self.mode == "texture":
            # Patterned texture-based distribution
            weights = self.texture_pattern(vertices)
            # Normalize and sample
            weights = weights.clamp(min=0)
            weights /= weights.sum()
            sampled_indices = torch.multinomial(
                weights, self.num_samples, replacement=True
            )
            samples_on_surface = vertices[sampled_indices]

        elif self.mode == "heat":
            nose_tip = torch.tensor([0.0, 0.1, -0.1], dtype=torch.float32)
            dists_to_nose = torch.norm(vertices - nose_tip, dim=1)
            self.mean_idx = dists_to_nose.argmin().item()
            # Geodesic heat kernel distribution
            dists = self.compute_geodesic_distances(self.mean_idx, vertices, faces)
            weights = torch.exp(-dists / (2 * self.sigma**2))

            # Set the mean vertex
            mu = vertices[self.mean_idx]

            # Compute geodesic distances using heat kernel
            dists = self.compute_geodesic_distances(self.mean_idx, vertices, faces)
            weights = torch.exp(-dists / (2 * self.sigma**2))
            weights /= weights.sum()

            # Sample indices based on geodesic heat kernel weights
            sampled_indices = torch.multinomial(
                weights, self.num_samples, replacement=True
            )
            samples_on_surface = vertices[sampled_indices]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Optional: add orthogonal noise
        if self.lifted:
            for i in range(samples_on_surface.size(0)):
                vertex_normal = vertex_normals[sampled_indices[i]]
                noise = self.noise_level * torch.randn(3)
                noise_orth = torch.dot(noise, vertex_normal) * vertex_normal
                samples_on_surface[i] += noise_orth

        return samples_on_surface

    def texture_pattern(self, vertices):
        # Center of rings
        center = torch.tensor([0.05, 0.1, 0.1], dtype=torch.float32).to(vertices.device)
        r = torch.norm(vertices - center, dim=1)

        # Ring parameters
        num_rings = 5
        r_min, r_max = r.min().item(), r.max().item()
        ring_centers = torch.linspace(r_min + 0.02, r_max - 0.02, num_rings).to(
            vertices.device
        )
        sigma = 0.003  # controls ring width (small = sharp)

        # Build the pattern as sum of Gaussians centered at ring radii
        pattern = torch.zeros_like(r)
        for rc in ring_centers:
            pattern += torch.exp(-0.5 * ((r - rc) / sigma) ** 2)

        # Add slight noise to edges
        torch.manual_seed(42)
        noise = 0.05 * torch.randn_like(pattern)
        pattern += noise

        return pattern.clamp(min=0, max=1)

    def compute_geodesic_distances(self, source_idx, vertices, faces):
        return torch.norm(vertices - vertices[source_idx], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def _plane_point_from_Ab(A, b):
    """
    Return a point p0 on the plane {x | <A,x> = b}.
    Robust to A being torch/numpy and shapes like (1,3),(3,1),(3,).
    Uses p0 = (b / ||A||^2) * A.
    """
    # to 1D numpy (3,)
    try:
        import torch

        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
    except Exception:
        pass

    A = np.asarray(A, dtype=np.float64).reshape(-1)
    if A.size != 3:
        raise ValueError(
            f"A must have 3 elements after flattening, got shape {A.shape}."
        )

    b = float(np.asarray(b).reshape(()))  # scalar

    norm2 = float(A @ A)  # ||A||^2
    if norm2 == 0.0:
        raise ValueError("A must be non-zero.")

    return (b / norm2) * A


def _plane_basis_from_normal(A):
    """
    Given a plane normal A, return unit normal n and an orthonormal basis (e1, e2)
    spanning the plane. Robust to A being torch/numpy and to shapes like (1,3),(3,1),(3,).
    """
    # Convert to 1D numpy with 3 elements
    try:
        import torch

        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
    except Exception:
        pass
    A = np.asarray(A, dtype=np.float64).reshape(-1)
    if A.size != 3:
        raise ValueError(
            f"A must have 3 elements after flattening, got shape {A.shape}."
        )

    normA = np.linalg.norm(A)
    if normA == 0:
        raise ValueError("A must be non-zero.")
    n = A / normA  # unit normal

    # Choose helper axis least aligned with n to avoid degeneracy
    idx = int(np.argmin(np.abs(n)))  # 0,1,2
    h = np.eye(3, dtype=np.float64)[idx]  # ex,ey,or ez

    e1 = np.cross(n, h)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-12:
        # Rare fallback: pick a different axis
        idx = (idx + 1) % 3
        h = np.eye(3, dtype=np.float64)[idx]
        e1 = np.cross(n, h)
        e1_norm = np.linalg.norm(e1)
        if e1_norm < 1e-12:
            idx = (idx + 1) % 3
            h = np.eye(3, dtype=np.float64)[idx]
            e1 = np.cross(n, h)
            e1_norm = np.linalg.norm(e1)
            if e1_norm < 1e-12:
                raise RuntimeError("Failed to construct a stable plane basis.")

    e1 /= e1_norm
    e2 = np.cross(n, e1)  # orthonormal if n,e1 are unit/orthogonal
    return n, e1, e2


def _rodrigues_rotate_from_z_to_n(n: np.ndarray) -> np.ndarray:
    """
    3x3 rotation matrix that maps z-hat to unit vector n (Rodrigues' formula).
    Useful if you like 'generate in xy-plane, rotate onto plane' thinking.
    """
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    c = float(np.dot(z, n))
    if s < 1e-12:
        # z and n are (anti)parallel
        if c > 0:  # already aligned
            return np.eye(3, dtype=np.float64)
        else:  # 180 deg: rotate about x-axis
            R = np.eye(3, dtype=np.float64)
            R[1, 1] = -1.0
            R[2, 2] = -1.0
            return R
    vx = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64
    )
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R


class SmileyFaceDataset(Dataset):
    def __init__(
        self,
        device,
        num_samples=1000,
        A=None,  # plane normal
        b=0.0,  # plane offset
        sphere_center=None,
        sphere_radius=None,
        noise_level=0.0,  # std of optional orthogonal/tangent noise
        lifted=False,  # if True, add orthogonal noise after embedding
        projection_type="plane",  # "plane" or "sphere"
        embed_mode="basis",  # "basis" (recommended) or "rotate"
        tangent_noise=False,  # if True, add tangent-plane noise instead of orthogonal
        isotropic=False,  # if True, add isotropic 3D Gaussian noise instead of tangent/normal
        rng=None,
        seed=None,
        preload_device=None,  # None (lazy), 'cpu' or 'cuda' to eagerly move data
    ):
        self.device = device
        self.num_samples = num_samples
        self.noise_level = float(noise_level)
        self.lifted = bool(lifted)
        self.projection_type = projection_type
        self.embed_mode = embed_mode
        self.tangent_noise = tangent_noise
        self.isotropic = bool(isotropic)
        self.rng = np.random.default_rng(seed) if rng is None else rng

        # Plane setup (only needed for plane mode)
        self.A = None
        if projection_type == "plane":
            if A is None:
                raise ValueError("For projection_type='plane', please provide A and b.")
            self.A = np.asarray(A, dtype=np.float64)
            if np.linalg.norm(self.A) == 0:
                raise ValueError("A must be non-zero.")
            self.b = float(b)
            self.n, self.e1, self.e2 = _plane_basis_from_normal(self.A)
            self.p0 = _plane_point_from_Ab(self.A, self.b)  # point on plane

            if embed_mode == "rotate":
                # rotation that maps z-hat to plane normal n
                self.R = _rodrigues_rotate_from_z_to_n(self.n)

        # Sphere setup
        self.sphere_center = (
            np.array([0.0, 0.0, 0.0], dtype=np.float64)
            if sphere_center is None
            else np.asarray(sphere_center, dtype=np.float64)
        )
        self.sphere_radius = 2.0 if sphere_radius is None else float(sphere_radius)

        # Generate points as CPU tensor by default and optionally move once
        # to a chosen device if `preload_device` is supplied. Keeping the
        # generated tensor on CPU by default avoids a large blocking host->GPU
        # transfer at process startup; training code will move batches to GPU
        # as needed which amortizes transfer latency.
        data_tensor = self._generate_smiley()
        if preload_device in ("cpu", "cuda"):
            # allow explicit preload (keeps compatibility with older callers
            # that passed device expecting an eager move)
            self.data = data_tensor.to(preload_device)
            self.preload_device = preload_device
        else:
            # lazy: keep on CPU and let DataLoader / training loop move batches
            self.data = data_tensor
            self.preload_device = None

    def _generate_2d_smiley(self):
        """Generate (N,2) smiley in its own (u,v) coords."""
        N = self.num_samples
        angles = np.linspace(0, 2 * np.pi, N // 2, endpoint=False)
        face = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # unit circle

        # eyes
        n_eye = N // 8
        left_eye = np.column_stack(
            [self.rng.normal(-0.4, 0.01, n_eye), self.rng.normal(0.4, 0.01, n_eye)]
        )
        right_eye = np.column_stack(
            [self.rng.normal(0.4, 0.01, n_eye), self.rng.normal(0.4, 0.01, n_eye)]
        )

        # mouth arc
        m = N // 4
        mouth_angles = np.linspace(-np.pi / 4, np.pi / 4, m)
        mouth = np.stack(
            [0.3 * np.cos(mouth_angles), 0.4 * np.sin(mouth_angles)], axis=1
        )
        # rotate mouth 90° clockwise in 2D: (x, y) -> (y, -x)
        mouth = np.stack([mouth[:, 1], -mouth[:, 0]], axis=1)

        pts = np.concatenate([face, left_eye, right_eye, mouth], axis=0)

        # tiny jitter (optional)
        pts += self.rng.normal(0.0, 0.05, size=pts.shape)
        return pts  # (M,2)

    def _embed_on_plane_basis(self, uv: np.ndarray) -> np.ndarray:
        """
        Embed (u,v) -> p0 + u e1 + v e2. Exactly on plane ⟨A,x⟩=b.
        """
        return (
            self.p0[None, :]
            + uv[:, [0]] * self.e1[None, :]
            + uv[:, [1]] * self.e2[None, :]
        )

    def _embed_on_plane_rotate(self, uv: np.ndarray) -> np.ndarray:
        """
        Treat (u,v,0) in the xy-plane, rotate so that z-hat maps to n, then translate onto plane.
        Translation uses p0 to satisfy ⟨A, x⟩=b.
        """
        xyz = np.column_stack([uv, np.zeros(len(uv), dtype=np.float64)])
        rotated = (self.R @ xyz.T).T
        return rotated + self.p0[None, :]

    def _embed_on_sphere(self, uv: np.ndarray) -> np.ndarray:
        # uv expected roughly in a disk of radius <= 2R (we’ll scale if needed)
        u, v = uv[:, 0], uv[:, 1]
        R = self.sphere_radius

        # Scale to fit comfortably in the valid Lambert disk (radius 2R)
        # If smiley fits in unit radius, s=1 is fine; else choose s<2R/max_r.
        s = 1.0
        x, y = s * u, s * v
        rho = np.sqrt(x**2 + y**2)
        # Clamp to avoid out-of-domain
        rho = np.minimum(rho, 2 * R - 1e-9)

        # Inverse Lambert: c = 2 asin(rho / (2R))
        c = 2.0 * np.arcsin(0.5 * rho / R)
        # Handle rho=0 safely
        sin_c_over_rho = np.where(rho > 0, np.sin(c) / rho, 1.0)

        X = R * x * sin_c_over_rho
        Y = R * y * sin_c_over_rho
        Z = R * np.cos(c)

        return np.stack([X, Y, Z], axis=1) + self.sphere_center[None, :]

    def _add_noise(self, pts3d: np.ndarray) -> np.ndarray:
        """
        If lifted:
          - orthogonal noise: add along plane normal (or radial on sphere)
          - tangent noise: add in span{e1,e2} (or any orthonormal tangent)
        """
        if self.noise_level <= 0:
            return pts3d

        if self.projection_type == "plane":
            if self.isotropic:
                # isotropic 3D Gaussian noise
                noise = self.rng.normal(0.0, self.noise_level, size=pts3d.shape)
                pts3d = pts3d + noise
            elif self.tangent_noise:
                # tangent-plane Gaussian noise
                u = self.rng.normal(0.0, self.noise_level, size=(pts3d.shape[0], 1))
                v = self.rng.normal(0.0, self.noise_level, size=(pts3d.shape[0], 1))
                pts3d = pts3d + u * self.e1[None, :] + v * self.e2[None, :]
            else:
                # orthogonal (normal) noise
                alpha = self.rng.normal(0.0, self.noise_level, size=(pts3d.shape[0], 1))
                pts3d = pts3d + alpha * self.n[None, :]
            return pts3d

        elif self.projection_type == "sphere":
            # orthogonal = radial noise
            rel = pts3d - self.sphere_center[None, :]
            rel_norm = np.linalg.norm(rel, axis=1, keepdims=True) + 1e-12
            n = rel / rel_norm
            if self.isotropic:
                # isotropic 3D Gaussian noise
                noise = self.rng.normal(0.0, self.noise_level, size=pts3d.shape)
                pts3d = pts3d + noise
            elif self.tangent_noise:
                # make two orthonormal tangent directions
                h = np.array([1.0, 0.0, 0.0])
                t1 = np.cross(n, h)
                t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
                t2 = np.cross(n, t1)
                u = self.rng.normal(0.0, self.noise_level, size=(pts3d.shape[0], 1))
                v = self.rng.normal(0.0, self.noise_level, size=(pts3d.shape[0], 1))
                pts3d = pts3d + u * t1 + v * t2
            else:
                alpha = self.rng.normal(0.0, self.noise_level, size=(pts3d.shape[0], 1))
                pts3d = pts3d + alpha * n
            return pts3d

        return pts3d

    def _generate_smiley(self) -> torch.Tensor:
        uv = self._generate_2d_smiley()  # (M,2)

        if self.projection_type == "plane":
            if self.embed_mode == "basis":
                pts3d = self._embed_on_plane_basis(uv)
            elif self.embed_mode == "rotate":
                pts3d = self._embed_on_plane_rotate(uv)
            else:
                raise ValueError("embed_mode must be 'basis' or 'rotate'.")

        elif self.projection_type == "sphere":
            pts3d = self._embed_on_sphere(uv)

        else:
            raise ValueError("projection_type must be 'plane' or 'sphere'.")

        if self.lifted and self.noise_level > 0.0:
            pts3d = self._add_noise(pts3d)

        return torch.tensor(pts3d, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

class MNISTFixedSumDataset(Dataset):
    def __init__(
        self,
        device,
        pixel_sum=100.0,
        noise_level=0.0,
        lifted=None,  # if truthy, precompute lifted (noisy) images once
        train=True,
        mnist_root="./data",
        flatten=True,
        reproject_after_noise=False,
        dtype=torch.float32,
        preload_device=None,  # None | 'cpu' | 'cuda'  (move once at init)
        cache_file=None,  # e.g. './data/mnist_fixedsum_train.pt'
        return_labels=False,  # set True if you want labels back
    ):
        """
        Tip: If you want maximum speed in training:
          - set preload_device='cuda' to keep the whole tensor on GPU (fits easily: ~188MB in fp32)
          - use DataLoader(..., pin_memory=True, num_workers>0) if staying on CPU
        """
        super().__init__()
        self.pixel_sum = float(pixel_sum)
        self.noise_level = float(noise_level)
        self.lifted = bool(lifted)
        self.flatten = bool(flatten)
        self.reproject_after_noise = bool(reproject_after_noise)
        self.dtype = dtype
        self.device = device
        self.return_labels = bool(return_labels)

        # Try cache first (avoids recomputation entirely)
        if cache_file is not None and os.path.isfile(cache_file):
            blob = torch.load(cache_file, map_location="cpu")
            X = blob["images"].to(dtype)
            y = blob["labels"].to(torch.long)
        else:
            # Load MNIST TENSORS directly (uint8, shape: (N, 28, 28))
            data, targets = _load_mnist_tensors(root=mnist_root, train=train)
            y = targets.clone().to(torch.long)
            X = data.clone().to(dtype) / 255.0  # (N, 28, 28) in [0,1]

            # Flatten if requested (do it once)
            if self.flatten:
                X = X.view(X.shape[0], -1)  # (N, 784)

            # Project all rows to sum = pixel_sum (vectorized)
            X = self._project_hyperplane_batch(X, s=self.pixel_sum)

            # Optional lifting (one scalar normal-noise per image), once
            if self.lifted and self.noise_level > 0.0:
                N, D = X.shape if self.flatten else (X.shape[0], X[0].numel())
                eps = torch.randn(N, 1, dtype=dtype) * self.noise_level
                if self.flatten:
                    X = X + eps / (D**0.5)
                else:
                    X = X + (eps / (D**0.5)).view(N, 1, 1)

                if self.reproject_after_noise:
                    if not self.flatten:
                        X = X.view(N, -1)
                        X = self._project_hyperplane_batch(X, s=self.pixel_sum)
                        X = X.view(N, 28, 28)
                    else:
                        X = self._project_hyperplane_batch(X, s=self.pixel_sum)

            # Save cache
            if cache_file is not None:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                torch.save({"images": X.cpu(), "labels": y.cpu()}, cache_file)

        # Optional one-shot preload to a target device (fastest path)
        if preload_device in ("cuda", "cpu"):
            X = X.to(preload_device, non_blocking=True)
            y = y.to(preload_device, non_blocking=True)

        # Store tensors
        self.images = X.contiguous()
        self.labels = y
        self.preload_device = preload_device

    @staticmethod
    def _project_hyperplane_batch(X, s: float):
        """
        Project each row of X onto {sum(row)=s} by adding a per-row constant.
        Works for (N, D) or (N, 28, 28) if flattened beforehand.
        """
        if X.dim() == 3:  # (N, H, W) -> treat each image as a row vector
            N = X.shape[0]
            X_flat = X.view(N, -1)
            X_proj = MNISTFixedSumDataset._project_hyperplane_batch(X_flat, s)
            return X_proj.view_as(X)
        N, D = X.shape
        correction = (s - X.sum(dim=1, keepdim=True)) / D
        return X + correction

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # No per-sample lifting or projection here—fully precomputed.
        x = self.images[idx]
        if self.preload_device is None:
            # Move once per sample only if you didn't preload
            x = x.to(self.device, non_blocking=True)
        # else it’s already on the chosen device

        if self.return_labels:
            y = self.labels[idx]
            if self.preload_device is None:
                y = y.to(self.device, non_blocking=True)
            return x, y
        return x


class TestSmileyFaceDataset(unittest.TestCase):
    def setUp(self):
        # Set up a constraint projector with a simple linear equality constraint
        self.constraint_projector = SimpleConstraintProjector()
        A = np.array([1.0, 2.0, 3.0])  # Normal vector for the constraint
        A_norm = A / np.linalg.norm(A)
        A_tensor = torch.tensor(A_norm.reshape(1, -1), dtype=torch.float32)
        b_tensor = torch.tensor([0.0], dtype=torch.float32)
        self.constraint_projector.add_linear_equality(A_tensor, b_tensor)

        # Initialize the dataset with the constraint projector
        self.dataset = SmileyFaceDataset(
            num_samples=1000,
            constraint_projector=self.constraint_projector,
            noise_level=10,
            lifted=False,
            projection_step_size=1e-3,
            projection_max_iter=100,
            projection_type="sphere",  # Change this to "plane" for plane projection
        )

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 1000)

    def test_dataset_item_shape(self):
        sample = self.dataset[0]
        self.assertEqual(sample.shape, torch.Size([3]))

    def test_data_loader(self):
        data_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        for batch in data_loader:
            self.assertEqual(batch.shape, torch.Size([32, 3]))
            break

    def test_lifted_mode(self):
        example_dataset = SmileyFaceDataset(
            num_samples=1000,
            constraint_projector=self.constraint_projector,
            noise_level=10,
            lifted=True,
            projection_step_size=1e-3,
            projection_max_iter=100,
            projection_type="sphere",
        )
        self.assertEqual(len(example_dataset), 1000)
        sample = example_dataset[0]
        self.assertEqual(sample.shape, torch.Size([3]))

    def test_seeded_generation_is_reproducible(self):
        common_kwargs = dict(
            device="cpu",
            num_samples=128,
            A=np.array([1.0, 2.0, 3.0]),
            b=1.0,
            lifted=True,
            noise_level=0.05,
            projection_type="plane",
        )

        dataset_a = SmileyFaceDataset(seed=42, **common_kwargs)
        dataset_b = SmileyFaceDataset(seed=42, **common_kwargs)
        dataset_c = SmileyFaceDataset(seed=43, **common_kwargs)

        self.assertTrue(torch.equal(dataset_a.data, dataset_b.data))
        self.assertFalse(torch.equal(dataset_a.data, dataset_c.data))

class ImageFixedSumDataset(Dataset):
    def __init__(
        self,
        device,
        dataset: str = "mnist",  # mnist (case-insensitive)
        pixel_sum: float = 100.0,
        noise_level: float = 0.0,
        lifted: bool | None = None,  # if truthy, precompute lifted (noisy) images once
        train: bool = True,
        data_root: str = "./data",
        flatten: bool = True,
        reproject_after_noise: bool = False,
        dtype=torch.float32,
        preload_device: str | None = None,  # None | 'cpu' | 'cuda'
        cache_file: str | None = None,  # e.g. './data/fixedsum_mnist_train.pt'
        return_labels: bool = False,
        num_samples: int | None = None,  # if set, randomly sample this many images
        random_seed: int = 42,  # seed for reproducible sampling
    ):
        """
        Generic fixed-sum image dataset with optional lifting noise.

        - dataset: 'mnist' (case-insensitive). Defaults to 'mnist'.
        - pixel_sum: target sum s for each image after projection.
        - lifted/noise_level: if lifted and noise_level>0, adds a single scalar
          Gaussian to each image, evenly distributed across pixels. If
          reproject_after_noise=True, reprojects back to the hyperplane.
        - flatten: return (N, D) vectors; else return (N, H, W).
        - preload_device: if set, pre-moves tensors to 'cpu' or 'cuda' once at init.
        - cache_file: optional path to save/load preprocessed tensors.
        - return_labels: when True, __getitem__ returns (x, y).
        - num_samples: if set, randomly sample this many images from the dataset.
        - random_seed: seed for reproducible sampling (default: 42).
        """
        super().__init__()
        self.pixel_sum = float(pixel_sum)
        self.noise_level = float(noise_level)
        self.lifted = bool(lifted)
        self.flatten = bool(flatten)
        self.reproject_after_noise = bool(reproject_after_noise)
        self.dtype = dtype
        self.device = device
        self.return_labels = bool(return_labels)
        self.num_samples = num_samples
        self.random_seed = random_seed

        name = (dataset or "mnist").strip().lower()
        # Always persist a cache file to disk, even if none was provided.
        # This keeps a reproducible sampled subset that can be reused elsewhere.
        if cache_file is None:
            split = "train" if train else "test"
            ns = self.num_samples if self.num_samples is not None else "all"
            cache_file = os.path.join(
                data_root,
                "cache",
                f"fixedsum_{name}_{split}_n{ns}_nl{self.noise_level}_lifted{int(self.lifted)}_seed{self.random_seed}.pt",
            )
        self.cache_file = cache_file
        if name not in ("mnist",):
            raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'mnist'.")

        # Always load and process the dataset (never load from cache)
        data, targets = _load_mnist_tensors(root=data_root, train=train)
        y = targets.clone().to(torch.long)
        X = data.clone().to(dtype) / 255.0  # (N, H, W) in [0,1]

        # Flatten if requested (do it once)
        if self.flatten:
            X = X.view(X.shape[0], -1)  # (N, D)

        # Project all rows to sum = pixel_sum (vectorized)
        X = self._project_hyperplane_batch(X, s=self.pixel_sum)

        # Optional lifting (one scalar normal-noise per image), once
        if self.lifted and self.noise_level > 0.0:
            N, D = X.shape if self.flatten else (X.shape[0], X[0].numel())
            eps = torch.randn(N, 1, dtype=dtype) * self.noise_level
            if self.flatten:
                X = X + eps / (D ** 0.5)
            else:
                X = X + (eps / (D ** 0.5)).view(N, 1, 1)

            if self.reproject_after_noise:
                if not self.flatten:
                    X = X.view(N, -1)
                    X = self._project_hyperplane_batch(X, s=self.pixel_sum)
                    X = X.view_as(data)
                else:
                    X = self._project_hyperplane_batch(X, s=self.pixel_sum)

        # Random sampling if num_samples is specified
        if self.num_samples is not None and self.num_samples < len(X):
            # Use a generator for reproducible sampling
            rng = torch.Generator()
            rng.manual_seed(self.random_seed)
            indices = torch.randperm(len(X), generator=rng)[:self.num_samples]
            X = X[indices]
            y = y[indices]

        # Always save to cache file after sampling
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        torch.save({"images": X.cpu(), "labels": y.cpu()}, self.cache_file)

        # Optional one-shot preload to a target device (fastest path)
        if preload_device in ("cuda", "cpu"):
            X = X.to(preload_device, non_blocking=True)
            y = y.to(preload_device, non_blocking=True)

        # Store tensors
        self.images = X.contiguous()
        self.labels = y
        self.preload_device = preload_device

    @staticmethod
    def _project_hyperplane_batch(X, s: float):
        """
        Project each row of X onto {sum(row)=s} by adding a per-row constant.
        Works for (N, D) or (N, H, W) if flattened beforehand.
        """
        if X.dim() == 3:  # (N, H, W) -> treat each image as a row vector
            N = X.shape[0]
            X_flat = X.view(N, -1)
            X_proj = ImageFixedSumDataset._project_hyperplane_batch(X_flat, s)
            return X_proj.view_as(X)
        N, D = X.shape
        correction = (s - X.sum(dim=1, keepdim=True)) / D
        return X + correction

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # No per-sample lifting or projection here—fully precomputed.
        x = self.images[idx]
        if self.preload_device is None:
            # Move once per sample only if you didn't preload
            x = x.to(self.device, non_blocking=True)
        # else it’s already on the chosen device

        if self.return_labels:
            y = self.labels[idx]
            if self.preload_device is None:
                y = y.to(self.device, non_blocking=True)
            return x, y
        return x


if __name__ == "__main__":
    unittest.main()
