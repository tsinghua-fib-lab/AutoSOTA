import unittest

import numpy as np
import torch
import trimesh

import numpy as np
import matplotlib.pyplot as plt

import torch

from scipy.stats import entropy

import torch.nn.functional as F

import math

from torch import nn

######################################
# Diversity: Pairwise RMSD within generated
######################################
import torch

@torch.no_grad()
def kabsch_align_batch(P, Q, eps=1e-8):
    """
    Batched Kabsch alignment: rotate/translate Q to best match P.
    P, Q: (..., N, 3) tensors
    returns Q_aligned with same shape
    """
    # center
    P_mean = P.mean(dim=-2, keepdim=True)
    Q_mean = Q.mean(dim=-2, keepdim=True)
    P0 = P - P_mean
    Q0 = Q - Q_mean

    # covariance and SVD
    # cov = Q0^T P0  (shape: ..., 3, 3)
    cov = torch.matmul(Q0.transpose(-2, -1), P0)
    U, S, Vh = torch.linalg.svd(cov)           # cov ≈ U @ diag(S) @ Vh
    V = Vh.transpose(-2, -1)

    # reflection fix: ensure a proper rotation (det = +1)
    det = torch.det(torch.matmul(V, U.transpose(-2, -1))).unsqueeze(-1).unsqueeze(-1)  # shape ...,1,1
    D = torch.diag_embed(torch.stack([torch.ones_like(det.squeeze(-1).squeeze(-1)),
                                      torch.ones_like(det.squeeze(-1).squeeze(-1)),
                                      det.squeeze(-1).squeeze(-1)], dim=-1))
    # If det is +1, D=I; if -1, flip last axis
    R = torch.matmul(torch.matmul(V, D), U.transpose(-2, -1))

    # rotate & translate
    Q_aligned = torch.matmul(Q0, R) + P_mean
    return Q_aligned

@torch.no_grad()
def rmsd_batch(P, Q):
    """
    RMSD over last two dims (N,3): sqrt(mean(||P-Q||^2))
    P,Q: (..., N, 3)
    """
    return torch.sqrt(torch.mean((P - Q) ** 2, dim=(-2, -1)))

@torch.no_grad()
def pairwise_rmsd(generated, device=None, chunk_pairs=200000):
    """
    generated: (B, N, 3) tensor (or anything reshapeable to that); stays on device.
    Returns 1D tensor of size B*(B-1)/2 with Kabsch-aligned RMSDs.
    chunk_pairs: how many (i,j) pairs to process at once
    """
    if device is None:
        device = generated.device
    B = generated.shape[0]
    # Flatten to (B, N, 3) if needed
    gen = generated.reshape(B, -1, 3).to(device)

    # upper-triangular indices
    ij = torch.triu_indices(B, B, offset=1, device=device)
    I, J = ij[0], ij[1]
    n_pairs = I.numel()
    out = torch.empty(n_pairs, device=device, dtype=gen.dtype)

    # chunk to limit memory
    start = 0
    while start < n_pairs:
        end = min(start + chunk_pairs, n_pairs)
        i_idx = I[start:end]
        j_idx = J[start:end]
        P = gen.index_select(0, i_idx)  # (M, N, 3)
        Q = gen.index_select(0, j_idx)  # (M, N, 3)
        Q_al = kabsch_align_batch(P, Q)
        out[start:end] = rmsd_batch(P, Q_al)
        start = end
    return out 
# ----------------------------------------------------------------------
# Fidelity-style RMSD between two *distributions* of structures
# ----------------------------------------------------------------------

def _sanitize_structs(X, eps_var=1e-12):
    """
    X: (B, N, 3) on any dtype/device.
    Returns:
      Xc64: centered float64 (B, N, 3)
      valid_mask: (B,) bool (finite & non-degenerate)
    """
    X = X.reshape(X.shape[0], -1, 3)
    dev = X.device
    X64 = X.to(dtype=torch.float64)

    # finite mask per structure
    finite = torch.isfinite(X64).all(dim=(1,2))

    # center
    mean = X64.mean(dim=1, keepdim=True)
    Xc = X64 - mean

    # non-degenerate: some variance
    var = (Xc**2).sum(dim=(1,2)) / (Xc.shape[1]*3)
    nondeg = var > eps_var

    valid = finite & nondeg
    return Xc, valid

@torch.no_grad()
def chamfer_rmsd(gen, true, device=None, chunk_g=512, chunk_t=512, inner_pairs=2048):
    """
    Symmetric 1-NN RMSD (Chamfer) with stable torch Kabsch.
    gen,true: (G,N,3),(T,N,3). Returns dict with 'forward','backward','sym'.
    chunk_g/chunk_t cap memory; inner_pairs caps the expanded pair batch size.
    """
    if device is None:
        # gen is expected to be a torch.Tensor; .device is an attribute, not a method.
        # Use gen.device (not callable) to obtain the torch.device.
        device = gen.device
    G, T = gen.shape[0], true.shape[0]

    # 1) sanitize once
    gen_c, g_valid = _sanitize_structs(gen.to(device))
    true_c, t_valid = _sanitize_structs(true.to(device))

    if (~g_valid).any():
        # replace invalid with a tiny jittered zero-shape to avoid NaNs dominating
        rep = torch.zeros((1, gen_c.shape[1], 3), dtype=torch.float64, device=device)
        rep += 1e-6*torch.randn_like(rep)
        gen_c[~g_valid] = rep[0]
    if (~t_valid).any():
        rep = torch.zeros((1, true_c.shape[1], 3), dtype=torch.float64, device=device)
        rep += 1e-6*torch.randn_like(rep)
        true_c[~t_valid] = rep[0]
    @torch.no_grad()
    def block_min_rmsd(Ac, Bc, chunk_b, device=None):
        """
        Ac: (a, N, 3) centered float64
        Bc: (b, N, 3) centered float64
        Returns: best per-A RMSD (a,)
        Uses: batched Kabsch via SVD on 3x3 cross-covariances computed with einsum
        """
        if device is None: device = Ac.device
        a, N, _ = Ac.shape
        b = Bc.shape[0]

        # Precompute norms once: ||A||_F^2 and ||B||_F^2 over (N,3)
        # shape: (a,), (b,)
        A_norm2 = (Ac.square()).sum(dim=(1, 2))          # (a,)
        B_norm2 = (Bc.square()).sum(dim=(1, 2))          # (b,)

        best2 = torch.full((a,), float('inf'), dtype=Ac.dtype, device=device)  # keep squared RMSD

        for bs in range(0, b, chunk_b):
            be = min(bs + chunk_b, b)
            Bblk = Bc[bs:be]                               # (b', N, 3)
            Bn2  = B_norm2[bs:be]                          # (b',)

            # Cross-covariances H_ij = A_i^T B_j, shape (a, b', 3, 3)
            # H[a, b, c, d] = sum_n A[a, n, c] * B[b, n, d]
            H = torch.einsum('anc,bnd->abcd', Ac, Bblk)    # (a, b', 3, 3)

            # Batched SVD (a, b', 3, 3)
            # Using torch.linalg.svd (full_matrices=False) is fine for 3x3
            U, S, Vh = torch.linalg.svd(H, full_matrices=False)  # S: (a, b', 3)

            # Kabsch reflection handling: if det(U)*det(V^T) < 0, subtract 2*min(S)
            detU  = torch.linalg.det(U)                     # (a, b')
            detVh = torch.linalg.det(Vh)                    # (a, b')
            reflect = (detU * detVh) < 0                    # (a, b')

            # trace term = sum(S) - 2*min(S) if reflection needed
            traceS = S.sum(-1)                              # (a, b')
            minS   = S.min(dim=-1).values                   # (a, b')
            traceS = torch.where(reflect, traceS - 2.0 * minS, traceS)

            # RMSD^2 = (||A||^2 + ||B||^2 - 2*traceS) / N, per pair (i,j)
            # Broadcast A_norm2 over b' and B_norm2 over a
            rmsd2 = (A_norm2[:, None] + Bn2[None, :] - 2.0 * traceS) / N  # (a, b')

            # Take min over B-block
            blk_min2 = rmsd2.min(dim=1).values             # (a,)
            best2 = torch.minimum(best2, blk_min2)

        # Avoid inf (shouldn't happen)
        best2 = torch.where(torch.isfinite(best2), best2, torch.tensor(1e12, dtype=best2.dtype, device=device))
        return best2.sqrt()  # return RMSD
        # if any inf survived (shouldn't), set to large but finite sentinel
        best = torch.where(torch.isfinite(best), best, torch.tensor(1e6, dtype=best.dtype, device=best.device))
        return best

    # 2) forward (gen -> true)
    fmins_all = []
    for gs in range(0, G, chunk_g):
        ge = min(gs + chunk_g, G)
        # chunk_b for block_min_rmsd should be the chunk size used to iterate Bc (true set)
        fmins_all.append(block_min_rmsd(gen_c[gs:ge], true_c, chunk_t))
    fmins = torch.cat(fmins_all)

    # 3) backward (true -> gen)
    bmins_all = []
    for ts in range(0, T, chunk_g):
        te = min(ts + chunk_g, T)
        # chunk_b for block_min_rmsd should be the chunk size used to iterate Bc (gen set)
        bmins_all.append(block_min_rmsd(true_c[ts:te], gen_c, chunk_g))
    bmins = torch.cat(bmins_all)

    f = fmins.mean().item()
    b = bmins.mean().item()
    return {"forward": f, "backward": b, "sym": 0.5*(f+b)}

######################################
# Torsion Angles
######################################
def compute_torsion(a, b, c, d):
    """Compute torsion angle between four points"""
    b0 = b - a
    b1 = c - b
    b2 = d - c

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b1, b2)

    b0xb1 /= np.linalg.norm(b0xb1, axis=-1, keepdims=True) + 1e-8
    b1xb2 /= np.linalg.norm(b1xb2, axis=-1, keepdims=True) + 1e-8

    cos_angle = np.sum(b0xb1 * b1xb2, axis=-1)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Determine sign
    sign = np.sign(np.sum(b1 * np.cross(b0xb1, b1xb2), axis=-1))
    angle = angle * sign

    return np.degrees(angle)

def extract_torsions(coords):
    """
    coords: (B, L, 3, 3)
    returns: list of torsion angles
    """
    coords = coords.cpu().numpy()
    B, L, A, D = coords.shape
    torsions = []
    for frag in coords:
        frag_torsions = []
        for i in range(L-2):
            C_prev = frag[i, 2]
            N = frag[i+1, 0]
            CA = frag[i+1, 1]
            C = frag[i+1, 2]
            N_next = frag[i+2, 0]
            phi = compute_torsion(C_prev, N, CA, C)
            psi = compute_torsion(N, CA, C, N_next)
            frag_torsions.append([phi, psi])
        torsions.append(frag_torsions)
    return np.array(torsions)  # (B, L-2, 2)

def compute_fid(emb1, emb2):
    """emb1, emb2: (B, D) arrays"""
    mu1, sigma1 = np.mean(emb1, axis=0), np.cov(emb1, rowvar=False)
    mu2, sigma2 = np.mean(emb2, axis=0), np.cov(emb2, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def evaluate(generated, true):
    print(">> Computing metrics...")

    # Diversity
    diversity_rmsds = pairwise_rmsd(generated)
    print(f"Median Pairwise RMSD (diversity): {np.median(diversity_rmsds):.3f} Å")
    fidelity_rmsds = chamfer_rmsd(generated, true)
    print(f"Chamfer RMSD (fidelity): {fidelity_rmsds:.3f} Å")

    # Torsion Distributions
    tors_gen = extract_torsions(generated)
    tors_true = extract_torsions(true)

    # Flatten torsions
    tors_gen_flat = tors_gen.reshape(-1, 2)
    tors_true_flat = tors_true.reshape(-1, 2)

    # Histograms and entropy
    plt.hist(tors_gen_flat[:,0], bins=50, alpha=0.5, label="Generated $\phi$", density=True)
    plt.hist(tors_true_flat[:,0], bins=50, alpha=0.5, label="True $\phi$", density=True)
    plt.legend()
    plt.title("Phi Torsion Angle Distribution")
    plt.savefig('phi.svg')
    plt.show()
    plt.clf()
    plt.hist(tors_gen_flat[:,1], bins=50, alpha=0.5, label="Generated $\psi$", density=True)
    plt.hist(tors_true_flat[:,1], bins=50, alpha=0.5, label="True $\psi$", density=True)
    plt.legend()
    plt.title("Psi Torsion Angle Distribution")
    plt.savefig('psi.svg')
    plt.show()

    # Entropy
    hist_gen_phi, _ = np.histogram(tors_gen_flat[:,0], bins=50, density=True)
    hist_true_phi, _ = np.histogram(tors_true_flat[:,0], bins=50, density=True)
    ent_gen_phi = entropy(hist_gen_phi + 1e-8)
    ent_true_phi = entropy(hist_true_phi + 1e-8)
    print(f"Entropy (Generated phi): {ent_gen_phi:.3f}")
    print(f"Entropy (True phi): {ent_true_phi:.3f}")

    # FID (optional, simple features)
    feat_gen = tors_gen_flat
    feat_true = tors_true_flat
    fid = compute_fid(feat_gen, feat_true)
    print(f"Fréchet Distance (torsions): {fid:.3f}")
        
# These are standard bond lengths and angles (in Å and degrees)
BOND_CONSTRAINTS = {
    "N-CA": 1.46,
    "CA-C": 1.52,
    "C-N+1": 1.33,
}
ANGLE_CONSTRAINTS_DEG = {
    "N-CA-C": 110.0,
    "CA-C-N+1": 116.0,
}

def unit_vector(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

def angle_between(v1, v2):
    dot = (v1 * v2).sum(-1)
    return torch.acos(torch.clamp(dot / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8), -1.0, 1.0))

def project_bond_length(p1, p2, desired_length):
    """
    Project two points so that their distance is exactly desired_length.
    """
    midpoint = (p1 + p2) / 2
    direction = unit_vector(p2 - p1)
    half_len = desired_length / 2
    return midpoint - half_len * direction, midpoint + half_len * direction

def find_nearest_vertex(point, vertices):
    """
    Find the closest vertex in the mesh to a given point.
    """
    dists = torch.norm(vertices - point, dim=1)
    nearest_idx = torch.argmin(dists)
    return nearest_idx, vertices[nearest_idx]


def project_to_surface(point, vertices, vertex_normals=None):
    """
    Project a point to the surface by snapping it to the nearest mesh vertex.
    """
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)

    # Find nearest vertex
    nearest_idx, nearest_vertex = find_nearest_vertex(point, vertices)

    # Return nearest vertex and the distance to it
    residual = point - nearest_vertex
    return nearest_vertex, torch.norm(residual)

@torch.no_grad()
def _closest_point_on_triangles(Q, T):
    """
    Q: (q, 3)
    T: (q, M, 3, 3) triangles (A,B,C)
    Returns:
      Cpts: (q, M, 3), d2: (q, M)
    """
    A = T[..., 0, :]  # (q, M, 3)
    B = T[..., 1, :]
    C = T[..., 2, :]

    AB = B - A
    AC = C - A
    AP = Q.unsqueeze(1) - A  # (q, M, 3)

    # region tests
    d1 = (AB * AP).sum(-1)
    d2p = (AC * AP).sum(-1)

    mask_A = (d1 <= 0) & (d2p <= 0)
    proj_A = A

    BP = Q.unsqueeze(1) - B
    d3 = (AB * BP).sum(-1)
    d4 = (AC * BP).sum(-1)
    mask_B = (d3 >= 0) & (d4 <= d3)
    proj_B = B

    vc = d1 * d4 - d3 * d2p
    mask_AB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_on_AB = d1 / (d1 - d3 + 1e-12)
    proj_AB = A + v_on_AB.unsqueeze(-1) * AB

    CP = Q.unsqueeze(1) - C
    d5 = (AB * CP).sum(-1)
    d6 = (AC * CP).sum(-1)
    mask_C = (d6 >= 0) & (d5 <= d6)
    proj_C = C

    vb = d5 * d2p - d1 * d6
    mask_AC = (vb <= 0) & (d2p >= 0) & (d6 <= 0)
    w_on_AC = d2p / (d2p - d6 + 1e-12)
    proj_AC = A + w_on_AC.unsqueeze(-1) * AC

    # inside face region
    ABAB = (AB * AB).sum(-1)
    ACAC = (AC * AC).sum(-1)
    ABAC = (AB * AC).sum(-1)
    ABAP = (AB * AP).sum(-1)
    ACAP = (AC * AP).sum(-1)

    denom2 = ABAB * ACAC - ABAC * ABAC + 1e-20
    u = (ACAC * ABAP - ABAC * ACAP) / denom2
    v = (ABAB * ACAP - ABAC * ABAP) / denom2

    mask_face = ~(mask_A | mask_B | mask_C | mask_AB | mask_AC)

    # SAFE clamping: avoid number+Tensor mix in clamp
    u_cl = torch.clamp(u, min=0.0, max=1.0)
    v_cl = torch.minimum(v, 1.0 - u_cl)   # tensor-tensor min
    v_cl = torch.clamp(v_cl, min=0.0)     # ensure nonnegative

    proj_face = A + u_cl.unsqueeze(-1) * AB + v_cl.unsqueeze(-1) * AC

    Cpts = torch.where(mask_A.unsqueeze(-1), proj_A,
            torch.where(mask_B.unsqueeze(-1), proj_B,
            torch.where(mask_C.unsqueeze(-1), proj_C,
            torch.where(mask_AB.unsqueeze(-1), proj_AB,
            torch.where(mask_AC.unsqueeze(-1), proj_AC,
                        proj_face)))))

    d2 = ((Cpts - Q.unsqueeze(1)) ** 2).sum(-1)
    return Cpts, d2


class MeshConstraintProjector:
    """
    Fast projector onto a triangle mesh surface. Ensures outputs lie on the mesh.

    Pipeline:
      1) k-NN to vertices (GEMM-based, chunked)
      2) Gather incident faces via a prebuilt padded vertex->face table (on GPU)
      3) Rank candidate faces per query by centroid distance; keep top-M
      4) Compute closest point on each candidate triangle; pick min

    Notes:
      - No Python loops in the hot path.
      - Padded adjacency is built once at init (linear in #incidences).
    """

    def __init__(
        self,
        mesh_path,
        device,
        dtype=torch.float32,
        Lcap: int = 48,              # cap for per-vertex incident faces kept
        use_half_for_knn: bool = False,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        # Load mesh (no processing; we normalize ourselves)
        mesh = trimesh.load_mesh(mesh_path, process=False)
        verts = torch.tensor(mesh.vertices, dtype=dtype)
        faces = torch.tensor(mesh.faces, dtype=torch.long)

        # Normalize vertices (center to mean, scale to max radius = 1)
        center = verts.mean(dim=0)
        verts = verts - center
        scale = verts.norm(dim=1).max().clamp_min(1e-12)
        verts = verts / scale

        self.center = center.to(self.device, dtype=dtype)
        self.scale = scale.to(self.device, dtype=dtype)

        # Cache on device
        self.vertices = verts.to(self.device).contiguous()        # (V, 3)
        self.faces = faces.to(self.device)                        # (F, 3)
        self.face_verts = self.vertices[self.faces].contiguous()  # (F, 3, 3)
        self.face_centroids = self.face_verts.mean(dim=1).contiguous()  # (F, 3)
        self.V = self.vertices.shape[0]
        self.F = self.faces.shape[0]
        self.use_half_for_knn = bool(use_half_for_knn)

        # Precompute squared norms for kNN distance formula
        self._v_sq = (self.vertices ** 2).sum(dim=1)  # (V,)
        if self.use_half_for_knn and self.dtype in (torch.float32, torch.bfloat16):
            # Keep a reduced-precision copy only for kNN
            self.vertices_hp = self.vertices.to(torch.bfloat16 if self.dtype == torch.float32 else self.dtype)
            self._v_sq_hp = self._v_sq.to(self.vertices_hp.dtype)
        else:
            self.vertices_hp = None
            self._v_sq_hp = None
            
        self.vertex_normals = self._compute_vertex_normals(self.vertices, self.faces)

        # Build padded vertex->face adjacency on device (cap to Lcap to bound memory)
        self.vertex_faces_padded, self.vertex_faces_mask = self._build_vertex_face_padded(
            V=self.V, faces=self.faces, device=self.device, Lcap=Lcap
        )
        # Sanity
        assert self.vertex_faces_padded.shape[0] == self.V

    def _compute_vertex_normals(self, vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        # specify dim explicitly (or use torch.linalg.cross)
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # or: torch.linalg.cross(...)
        vertex_normals = torch.zeros_like(vertices)
        vertex_normals.index_add_(0, faces[:, 0], face_normals)
        vertex_normals.index_add_(0, faces[:, 1], face_normals)
        vertex_normals.index_add_(0, faces[:, 2], face_normals)

        return F.normalize(vertex_normals, dim=1, eps=1e-12)
        
    @torch.no_grad()
    def _build_vertex_face_padded(self, V, faces, device, Lcap=None):
        """
        Build a padded table:
          vf:   (V, Lmax) long with face indices or -1
          mask: (V, Lmax) bool

        Lmax is min(max incidence, Lcap) if Lcap is provided.
        """
        F = faces.shape[0]
        v_ids = faces.reshape(-1)                                       # (3F,)
        f_ids = torch.arange(F, device=device, dtype=torch.long).repeat_interleave(3)

        # Sort by vertex id to group incidences
        order = torch.argsort(v_ids)
        v_sorted = v_ids[order]
        f_sorted = f_ids[order]

        # Count incidences per vertex
        counts = torch.bincount(v_sorted, minlength=V)
        Lmax = int(counts.max().item())
        if Lcap is not None:
            Lmax = min(Lmax, int(Lcap))

        vf = torch.full((V, Lmax), -1, dtype=torch.long, device=device)
        # Fill by walking the sorted list once (linear in 3F). This loop is at init only.
        fill_ptr = torch.zeros(V, dtype=torch.long, device=device)

        for idx in range(v_sorted.numel()):
            v = v_sorted[idx]
            c = fill_ptr[v].item()
            if c < Lmax:
                vf[v, c] = f_sorted[idx]
                fill_ptr[v] = c + 1

        mask = vf >= 0
        return vf, mask

    @torch.no_grad()
    def _knn_vertices(self, Q, k=8, batch=1_000_000):
        """
        Brute-force kNN to vertices via |q|^2 + |v|^2 - 2 q·v, chunked.
        Q: (N, 3) on device
        Returns:
          nn_idx: (N, k) long
          nn_d2 : (N, k) float (squared distances)
        """
        N = Q.shape[0]
        V = self.V
        k = min(k, V)

        use_hp = (self.vertices_hp is not None)
        Verts = self.vertices_hp if use_hp else self.vertices
        v_sq = self._v_sq_hp if use_hp else self._v_sq

        idx_out = []
        d2_out = []
        for s in range(0, N, batch):
            e = min(s + batch, N)
            qpts = Q[s:e].contiguous()
            if use_hp:
                qhp = qpts.to(Verts.dtype)
                q2 = (qhp * qhp).sum(dim=1, keepdim=True)  # (b,1)
                d2 = q2 - 2.0 * (qhp @ Verts.T) + v_sq[None, :]
                d2 = d2.to(qpts.dtype)
            else:
                q2 = (qpts * qpts).sum(dim=1, keepdim=True)
                d2 = q2 - 2.0 * (qpts @ self.vertices.T) + self._v_sq[None, :]
            d2 = d2.clamp_min_(0)
            d2k, idxk = torch.topk(d2, k=k, largest=False, dim=1)
            idx_out.append(idxk)
            d2_out.append(d2k)
        return torch.cat(idx_out, 0), torch.cat(d2_out, 0)

    def constraint_residual(self, X):
        """
        Differentiable PIDM residual for a mesh surface.

        The nearest surface point is selected with the existing projector and
        treated as fixed for the backward pass. The residual itself is still a
        tensor depending on X, so ||residual||^2 trains the denoiser to move
        x0 estimates toward the selected surface point.
        """
        X = X.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            snapped, _, _ = self.project(X.detach())
        return X - snapped.detach()

    @torch.no_grad()
    def project(
        self,
        X,
        k_vertices: int = 2,
        max_faces_per_point: int = 16,
        chunk: int = 128_000,
        return_details: bool = False,
    ):
        """
        Project points X onto the mesh surface.

        Args:
          X:  (..., 3) tensor (any shape ending with 3)
          k_vertices: how many nearest vertices to use to gather faces
          max_faces_per_point: cap of candidate faces per query (after ranking)
          chunk: batch size for the final closest-point computation
        """
        X = X.to(self.device, dtype=self.dtype)
        orig_shape = X.shape
        Xf = X.reshape(-1, 3).contiguous()
        N = Xf.shape[0]

        if N == 0:
            if return_details:
                return X, torch.zeros(X.shape[:-1], device=self.device, dtype=self.dtype), {"method": "nearest-point-on-triangles"}
            return X, torch.zeros((), device=self.device, dtype=self.dtype), None

        # 1) kNN to vertices
        nn_idx, _ = self._knn_vertices(Xf, k=k_vertices)

        # 2) Gather candidate faces per query from prebuilt adjacency (vectorized)
        cand_faces = self.vertex_faces_padded[nn_idx]         # (N, k, Lmax)
        cand_mask  = self.vertex_faces_mask[nn_idx]           # (N, k, Lmax)
        N_, k_, Lmax = cand_faces.shape
        assert N_ == N
        Ktot = k_ * Lmax
        cand_faces = cand_faces.view(N, Ktot)                 # (N, Ktot)
        cand_mask  = cand_mask.view(N, Ktot)                  # (N, Ktot)

        # 3) Rank by centroid distance and trim to M
        M = min(max_faces_per_point, Ktot)
        # Compute distances only for valid slots
        valid_faces = cand_faces.clamp_min(0)
        centroids = self.face_centroids[valid_faces]          # (N, Ktot, 3)
        d2 = ((centroids - Xf.unsqueeze(1)) ** 2).sum(-1)     # (N, Ktot)
        d2 = torch.where(cand_mask, d2, torch.full_like(d2, float('inf')))
        _, idx_sel = torch.topk(d2, k=M, largest=False, dim=1)  # (N, M)
        face_sel = torch.gather(cand_faces, 1, idx_sel)          # (N, M)
        valid_mask = face_sel >= 0                               # (N, M)

        # 4) Closest point on selected triangles (chunked)
        snapped = torch.empty_like(Xf)
        best_d2 = torch.full((N,), float('inf'), device=self.device, dtype=self.dtype)

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            Q = Xf[s:e]                           # (q, 3)
            Fidx = face_sel[s:e]                  # (q, M)
            mask = valid_mask[s:e]                # (q, M)
            if M == 0 or mask.sum() == 0:
                snapped[s:e] = Q
                best_d2[s:e] = 0
                continue

            T = self.face_verts[Fidx.clamp_min(0)]  # (q, M, 3, 3)
            Cpts, d2_local = _closest_point_on_triangles(Q, T)  # (q, M, 3), (q, M)

            # Mask invalid slots
            d2_local = torch.where(mask, d2_local, torch.full_like(d2_local, float('inf')))
            best_vals, best_idx = d2_local.min(dim=1)  # (q,)
            row = torch.arange(Q.shape[0], device=self.device)
            chosen = Cpts[row, best_idx, :]            # (q, 3)

            snapped[s:e] = chosen
            best_d2[s:e] = best_vals

        snapped = snapped.view(orig_shape)
        dist = best_d2.view(*orig_shape[:-1]).sqrt()

        if return_details:
            return snapped, dist, {
                "method": "nearest-point-on-triangles",
                "k_vertices": k_vertices,
                "faces_per_point": M,
            }
        return snapped, dist.mean(), None
class SimpleConstraintProjector:
    def __init__(self, device):
        self.linear_equalities = []
        self.nonlinear_equalities = []
        self.sphere_constraints = []  # New list for sphere constraints
        self.device = device

    def add_sphere_constraint(self, center, radius):
        self.sphere_constraints.append((center, radius))

    def add_linear_equality(self, A_eq, b_eq):
        self.linear_equalities.append((A_eq.to(self.device), b_eq.to(self.device)))

    def add_nonlinear_equality(self, equality_func):
        self.nonlinear_equalities.append(equality_func)

    def add_constraints_from_dict(self, constraints_dict):
        for constraint_type, constraints in constraints_dict.items():
            if constraint_type == "linear_equality":
                # Expect an ordered pair (A_eq, b_eq). Reject unordered containers
                # (e.g., set) or malformed inputs to avoid silent/unpredictable bugs
                # where tensors may be unhashable or unpacked in the wrong order.
                if not isinstance(constraints, (list, tuple)):
                    raise TypeError(
                        f"linear_equality constraints must be a tuple/list (A_eq, b_eq); got {type(constraints)}"
                    )
                if len(constraints) != 2:
                    raise ValueError("linear_equality constraints must contain exactly two elements: (A_eq, b_eq)")
                A_eq, b_eq = constraints
                self.add_linear_equality(A_eq.to(self.device), b_eq.to(self.device))
            elif constraint_type == "nonlinear_equality":
                equality_func = constraints
                self.add_nonlinear_equality(equality_func)
            elif constraint_type == "sphere_equality":
                sphere_center, sphere_radius = constraints
                self.add_sphere_constraint(sphere_center, sphere_radius)
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")

    def constraint_residual(self, x):
        """
        Return differentiable equality residuals evaluated at x, without
        projecting x first. Shape is (B, M), where M is the total number of
        scalar constraints.
        """
        was_1d = x.dim() == 1
        x_eval = x.unsqueeze(0) if was_1d else x
        x_eval = x_eval.to(self.device)
        residuals = []

        for center, radius in self.sphere_constraints:
            center = torch.as_tensor(
                center, device=x_eval.device, dtype=x_eval.dtype
            ).view(1, -1)
            radius = torch.as_tensor(radius, device=x_eval.device, dtype=x_eval.dtype)
            residuals.append(torch.linalg.vector_norm(x_eval - center, dim=1, keepdim=True) - radius)

        for A_eq, b_eq in self.linear_equalities:
            A_eq = A_eq.to(device=x_eval.device, dtype=x_eval.dtype)
            b_eq = b_eq.to(device=x_eval.device, dtype=x_eval.dtype)
            residuals.append(x_eval @ A_eq.T - b_eq.view(1, -1))

        for func in self.nonlinear_equalities:
            residual = func(x_eval)
            residuals.append(residual.reshape(x_eval.shape[0], -1))

        if not residuals:
            return torch.zeros(
                (x_eval.shape[0], 1), device=x_eval.device, dtype=x_eval.dtype
            )
        return torch.cat(residuals, dim=1)

    def project_nonlinear_equality(
        self, x, equality_func, step_size=1e-3, max_iter=10000, tol=1e-3
    ):
        x_proj = x.clone().requires_grad_(True)
        prev_residual = float("inf")

        for _ in range(max_iter):
            equality_value = equality_func(x_proj)
            residual = torch.abs(equality_value)

            if torch.all(residual <= tol):
                return x_proj.detach(), residual.mean().item(), None

            if torch.any(residual >= prev_residual):
                step_size *= 0.5
                if step_size < 1e-10:
                    break

            prev_residual = residual

            equality_grad = torch.autograd.grad(
                equality_value.sum(), x_proj, create_graph=True
            )[0]
            x_proj = x_proj - step_size * equality_grad

        residual = torch.abs(equality_func(x_proj))
        normal = torch.autograd.grad(equality_func(x_proj).sum(), x_proj)[0]
        return x_proj.squeeze().detach(), residual.mean().item(), normal

    def project_linear_equality(self, x, A_eq, b_eq):
        """
        Projects a batch of points x ∈ ℝ^{B×D} onto the linear constraint A_eq x = b_eq.
        A_eq ∈ ℝ^{m×D}, b_eq ∈ ℝ^{m}
        Returns: x_proj ∈ ℝ^{B×D}, residual scalar, normal ∈ ℝ^{D}
        """
        x = x.to(self.device)                         # (B, D)
        A_eq = A_eq.to(self.device)                   # (m, D)
        b_eq = b_eq.to(self.device)                   # (m,)

        B, D = x.shape
        m = A_eq.shape[0]

        # Compute pseudoinverse of A Aᵀ once (m x m)
        AAt_inv = torch.linalg.pinv(A_eq @ A_eq.T).to(self.device)  # (m, m)

        # Compute projection matrix P = Aᵀ (A Aᵀ)^(-1) A ∈ ℝ^{D×D}
        P = (A_eq.T @ AAt_inv @ A_eq).to(self.device)               # (D, D)

        # Compute offset correction: Aᵀ (A Aᵀ)^(-1) b
        offset = (A_eq.T @ AAt_inv @ b_eq).to(self.device)          # (D,)

        # Project: x_proj = (I - P)x + offset
        I = torch.eye(D, device=self.device)                        # (D, D)
        x_proj = x @ (I - P).T + offset                             # (B, D)

        # Compute constraint residual
        residual = torch.norm((A_eq @ x_proj.T - b_eq[:, None]), dim=0).mean()

        # Extract normal vector
        if A_eq.shape[0] == 1:
            normal = A_eq[0] / torch.norm(A_eq[0])
        else:
            normal = torch.linalg.svd(A_eq)[0][:, 0]  # first left-singular vector

        return x_proj, residual.item(), normal

    def project(self, x, step_size=1e-3, max_iter=100, return_residual=True):
        """
        Projects x ∈ ℝ^{B×D} onto the intersection of constraints.
        Supports batch projection.
        Returns:
            x_proj ∈ ℝ^{B×D},
            norm_residual: float,
            mean_normal ∈ ℝ^{D} (if applicable)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return_single = True
        else:
            return_single = False

        norm_residual = 0.0
        normals = []

        # Sphere constraints (optional)
        for center, radius in self.sphere_constraints:
            x = x.to(self.device)
            center = torch.tensor(center, device=self.device).view(1, -1)
            delta = x - center
            delta_norm = torch.norm(delta, dim=1, keepdim=True) + 1e-8
            x = center + delta * (radius / delta_norm)
            residual = torch.abs(torch.norm(x - center, dim=1) - radius)
            norm_residual += residual.mean().item()
            normals.append((x - center).squeeze(0) / radius)

        # Linear equality constraints
        for A_eq, b_eq in self.linear_equalities:
            x, residual, normal = self.project_linear_equality(x, A_eq, b_eq)
            norm_residual += residual
            if normal is not None:
                normals.append(normal)

        # Nonlinear equality constraints
        for func in self.nonlinear_equalities:
            x_proj = x.clone().requires_grad_(True)
            prev_residual = float("inf")

            for _ in range(max_iter):
                val = func(x_proj)
                residual = torch.abs(val)
                if torch.all(residual <= 1e-3):
                    break
                grad = torch.autograd.grad(val.sum(), x_proj, create_graph=True)[0]
                x_proj = x_proj - step_size * grad

            residual = torch.abs(func(x_proj))
            norm_residual += residual.mean().item()
            normal = torch.autograd.grad(func(x_proj).sum(), x_proj)[0].mean(dim=0)
            normals.append(normal.detach())
            x = x_proj.detach()

        mean_normal = torch.stack(normals).mean(dim=0) if normals else None

        if return_single:
            x = x.squeeze(0)
        if return_residual:
            return x, norm_residual, mean_normal
        return x


class TestSimpleConstraintProjector(unittest.TestCase):
    def setUp(self):
        self.projector = SimpleConstraintProjector()

    def test_add_sphere_constraint(self):
        center = torch.tensor([1.0, 1.0])
        radius = 3.0
        self.projector.add_sphere_constraint(center, radius)
        self.assertEqual(len(self.projector.sphere_constraints), 1)
        self.assertTrue(torch.equal(self.projector.sphere_constraints[0][0], center))
        self.assertEqual(self.projector.sphere_constraints[0][1], radius)

    def test_project_sphere_constraint(self):
        center = torch.tensor([0.0, 0.0])
        radius = 2.0
        self.projector.add_sphere_constraint(center, radius)
        x = torch.tensor([[3.0, 4.0], [1.0, 1.0]])  # Norm 5 and sqrt(2)
        x_proj = self.projector.project(x, return_residual=False)
        expected = torch.tensor(
            [[3.0 / 5 * 2, 4.0 / 5 * 2], [1.0 / np.sqrt(2) * 2, 1.0 / np.sqrt(2) * 2]]
        )
        self.assertTrue(torch.allclose(x_proj, expected, atol=1e-6))

    def test_project_sphere_constraint_at_center(self):
        center = torch.tensor([1.0, 1.0])
        radius = 3.0
        self.projector.add_sphere_constraint(center, radius)
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        x_proj = self.projector.project(x, return_residual=False)
        expected_norm = radius
        computed_norm = torch.norm(x_proj - center.unsqueeze(0), dim=1)
        self.assertTrue(
            torch.allclose(
                computed_norm, torch.tensor([expected_norm, expected_norm]), delta=1e-6
            )
        )


class TestMeshProjection(unittest.TestCase):
    """
    Unit tests for mesh projection functions.
    """

    def setUp(self):
        self.vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        self.normals = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        )

    def test_find_nearest_vertex(self):
        point = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
        idx, vertex = find_nearest_vertex(point, self.vertices)
        self.assertIn(idx, range(len(self.vertices)))

    def test_project_to_surface(self):
        point = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        projected_point = project_to_surface(point, self.vertices, self.normals)
        self.assertEqual(projected_point.shape, point.shape)

def project_bond_angle(p1, vertex, p2, target_angle_deg):
    """
    Given three points (p1 - vertex - p2), adjust p1 and p2 such that the angle at the vertex
    equals the target angle in degrees. The vertex is kept fixed.
    """
    v1 = p1 - vertex
    v2 = p2 - vertex
    current_angle = angle_between(v1, v2)
    if current_angle == target_angle_deg:
        return p1, p2  # Already satisfied

    # Rotate v2 around the normal vector to change the angle
    axis = np.cross(v1, v2)
    if np.linalg.norm(axis) < 1e-6:
        return p1, p2  # Can't rotate if v1 and v2 are colinear

    axis = unit_vector(axis)
    angle_diff_rad = np.radians(target_angle_deg - current_angle)

    # Rodrigues' rotation formula
    def rotate(vec, axis, theta):
        return (vec * np.cos(theta) +
                np.cross(axis, vec) * np.sin(theta) +
                axis * np.dot(axis, vec) * (1 - np.cos(theta)))

    # Apply rotation to both vectors
    v1_rot = rotate(v1, axis, -angle_diff_rad / 2)
    v2_rot = rotate(v2, axis, +angle_diff_rad / 2)

    # Return adjusted points
    return vertex + v1_rot, vertex + v2_rot

def correct_structure_with_angles(coords, atom_names):
    """
    Applies corrections to enforce bond lengths and bond angles across a protein structure.
    """
    corrected_coords = [res.copy() for res in coords]
    L = len(corrected_coords)
    for i in range(L):
        name_to_coord = dict(zip(atom_names[i], corrected_coords[i]))

        # Enforce bond lengths
        for bond in ["N-CA", "CA-C", "C-O"]:
            a1, a2 = bond.split("-")
            if a1 in name_to_coord and a2 in name_to_coord:
                p1, p2 = name_to_coord[a1], name_to_coord[a2]
                p1_proj, p2_proj = project_bond_length(p1, p2, BOND_CONSTRAINTS[bond])
                name_to_coord[a1], name_to_coord[a2] = p1_proj, p2_proj

        # Enforce bond angles
        if all(a in name_to_coord for a in ["N", "CA", "C"]):
            p1, vertex, p2 = name_to_coord["N"], name_to_coord["CA"], name_to_coord["C"]
            p1_proj, p2_proj = project_bond_angle(p1, vertex, p2, ANGLE_CONSTRAINTS["N-CA-C"])
            name_to_coord["N"], name_to_coord["C"] = p1_proj, p2_proj

        if all(a in name_to_coord for a in ["CA", "C", "O"]):
            p1, vertex, p2 = name_to_coord["CA"], name_to_coord["C"], name_to_coord["O"]
            p1_proj, p2_proj = project_bond_angle(p1, vertex, p2, ANGLE_CONSTRAINTS["CA-C-O"])
            name_to_coord["CA"], name_to_coord["O"] = p1_proj, p2_proj

        corrected_coords[i] = np.array([name_to_coord[atom] for atom in atom_names[i]])

        # Inter-residue constraints
        if i < L - 1:
            current = dict(zip(atom_names[i], corrected_coords[i]))
            next_res = dict(zip(atom_names[i + 1], corrected_coords[i + 1]))
            if "C" in current and "N" in next_res:
                p1, p2 = current["C"], next_res["N"]
                p1_proj, p2_proj = project_bond_length(p1, p2, BOND_CONSTRAINTS["C-N+1"])
                current["C"], next_res["N"] = p1_proj, p2_proj
                corrected_coords[i] = np.array([current[atom] for atom in atom_names[i]])
                corrected_coords[i + 1] = np.array([next_res[atom] for atom in atom_names[i + 1]])

            if "CA" in current and "C" in current and "N" in next_res:
                p1, vertex, p2 = current["CA"], current["C"], next_res["N"]
                p1_proj, p2_proj = project_bond_angle(p1, vertex, p2, ANGLE_CONSTRAINTS["CA-C-N+1"])
                current["CA"], next_res["N"] = p1_proj, p2_proj
                corrected_coords[i] = np.array([current[atom] for atom in atom_names[i]])
                corrected_coords[i + 1] = np.array([next_res[atom] for atom in atom_names[i + 1]])

    return corrected_coords

from torch.optim import LBFGS

import torch
from torch.optim import LBFGS

# --- canonical backbone targets (Å, degrees) ---
BOND_CONSTRAINTS = {
    "N-CA": 1.46,
    "CA-C": 1.52,
    "C-N+1": 1.33,
}
ANGLE_CONSTRAINTS = {
    "N-CA-C": 110.0,
    "CA-C-N+1": 116.0,
    "C-N-CA": 121.0,   # added: hinge at N
}
# peptide planarity (ω ~ 180°, trans)
DIHEDRAL_TARGETS = {
    "omega": 180.0,    # CA_i—C_i—N_{i+1}—CA_{i+1}
    # optional priors (set weights > 0 below if you want them)
    "phi": None,       # C_{i-1}—N_i—CA_i—C_i
    "psi": None,       # N_i—CA_i—C_i—N_{i+1}
}

# ----- stable primitives -----
def _angle_stable(a, b, c):
    v1 = a - b; v2 = c - b
    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-12)
    v2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-12)
    cross = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    dot   = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.atan2(cross, dot)

def _dihedral(a, b, c, d):
    b0 = b - a; b1 = c - b; b2 = d - c
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-12)
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)

def _wrap_to_pi(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

# ----- residual vector c(x) with the full constraint set -----
def _constraint_vector_full(frag, BL, BA, DT):
    L = frag.shape[0]
    dt, dv = frag.dtype, frag.device
    N, CA, C = frag[:,0,:], frag[:,1,:], frag[:,2,:]

    one46 = torch.tensor(BL["N-CA"],  dtype=dt, device=dv)
    one52 = torch.tensor(BL["CA-C"],  dtype=dt, device=dv)
    one33 = torch.tensor(BL["C-N+1"], dtype=dt, device=dv)
    deg   = torch.tensor(torch.pi/180.0, dtype=dt, device=dv)

    ang_NCAC = torch.tensor(BA["N-CA-C"],   dtype=dt, device=dv) * deg
    ang_CACN = torch.tensor(BA["CA-C-N+1"], dtype=dt, device=dv) * deg
    ang_CNCA = torch.tensor(BA["C-N-CA"],   dtype=dt, device=dv) * deg

    vec = []
    vec.append((N - CA).norm(dim=-1) - one46)
    vec.append((CA - C).norm(dim=-1) - one52)
    if L > 1:
        vec.append((C[:-1] - N[1:]).norm(dim=-1) - one33)

    vec.append(_angle_stable(N, CA, C) - ang_NCAC)
    if L > 1:
        vec.append(_angle_stable(CA[:-1], C[:-1], N[1:]) - ang_CACN)
        vec.append(_angle_stable(C[:-1],  N[1:],  CA[1:]) - ang_CNCA)

    if L > 1:
        omega = _dihedral(CA[:-1], C[:-1], N[1:], CA[1:])
        vec.append(_wrap_to_pi(omega - torch.tensor(torch.pi, dtype=dt, device=dv)))

    return torch.cat([t.reshape(-1) for t in vec], dim=0)  # (m,)

class FastConstraintProjector(nn.Module):
    """
    One-step linearized constraint projection:
        δx = - J^T (J J^T + λ I)^-1 c
    Batched over chunks; early-exit if already close. No GN loops.
    """
    def __init__(self, bond_lengths, bond_angles, dihedral_targets,
                 damping=1e-3, chunk=256, tol_inf=1e-3):
        super().__init__()
        self.BL = bond_lengths
        self.BA = bond_angles
        self.DT = dihedral_targets
        self.damping = damping
        self.chunk = chunk
        self.tol_inf = tol_inf  # skip if max residual below this

    @torch.no_grad()
    def optimize(self, x0):  # x0: (B,L,3,3)
        B, L, _, _ = x0.shape
        x = x0.clone()
        D = L * 9

        for s in range(0, B, self.chunk):
            e = min(B, s + self.chunk)
            xb = x[s:e]  # (b,L,3,3)
            b = xb.shape[0]

            # Per-sample (small) solves; batch-chunk keeps GPU busy.
            for i in range(b):
                xi = xb[i].clone().requires_grad_(True)

                # c(x)
                def c_fun(z):
                    return _constraint_vector_full(z, self.BL, self.BA, self.DT)

                ci = c_fun(xi)                     # (m,)
                if ci.abs().max().item() < self.tol_inf:
                    xb[i] = xi.detach()
                    continue

                # Jacobian J via jacobian wrt flattened x
                zi = xi.reshape(-1)
                def c_fun_flat(z_flat):
                    z = z_flat.view(L,3,3).requires_grad_(True)
                    return c_fun(z)

                Ji = torch.autograd.functional.jacobian(
                    c_fun_flat, zi, create_graph=False, vectorize=True
                )                                    # (m, D)
                # Solve in constraint space: (G + λI) α = c, then δx = - J^T α
                G = Ji @ Ji.T                         # (m, m)
                G = G + self.damping * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
                try:
                    alpha = torch.linalg.solve(G, ci)  # (m,)
                except RuntimeError:
                    alpha = torch.linalg.pinv(G) @ ci

                delta = -(Ji.T @ alpha)              # (D,)
                xi_new = zi + delta
                xb[i] = xi_new.view(L,3,3).detach()

            x[s:e] = xb
        return x

class BatchedProteinGeometryOptimizer:
    def __init__(
        self,
        bond_lengths,
        bond_angles,
        dihedral_targets,
        w_len=10.0, w_ang=3.0,
        w_omega=5.0, w_phi=0.0, w_psi=0.0,
        max_iter=100, tol=1e-6
    ):
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.dihedral_targets = dihedral_targets
        self.w_len = w_len
        self.w_ang = w_ang
        self.w_omega = w_omega
        self.w_phi = w_phi
        self.w_psi = w_psi
        self.max_iter = max_iter
        self.tol = tol

    def optimize(self, x0):
        """
        x0: (B, L, 3, 3)
        returns projected x: (B, L, 3, 3)
        """
        x = x0.clone().detach().requires_grad_(True)
        optimizer = LBFGS([x], max_iter=self.max_iter, tolerance_grad=self.tol)

        def closure():
            optimizer.zero_grad()
            loss = self._constraint_loss(x)
            loss.backward()
            return loss

        optimizer.step(closure)
        return x.detach()

    def _constraint_loss(self, coords):
        """
        coords: (B, L, 3, 3) with atom order [N, CA, C]
        """
        B, L, A, D = coords.shape
        assert A == 3 and D == 3, "Expect atom order [N, CA, C] with xyz"

        N  = coords[:, :, 0, :]  # (B,L,3)
        CA = coords[:, :, 1, :]
        C  = coords[:, :, 2, :]

        loss_terms = []
        counts = []

        # ----- bond lengths -----
        def bond_len(a, b, target):
            return ((a - b).norm(dim=-1) - target) ** 2  # (B,L) or (B,L-1)

        # per-residue
        L_N_CA  = bond_len(N, CA, self.bond_lengths["N-CA"]).sum()
        L_CA_C  = bond_len(CA, C, self.bond_lengths["CA-C"]).sum()
        counts += [N.numel()//3, CA.numel()//3]  # approximate count accounting
        loss_terms += [self.w_len * L_N_CA, self.w_len * L_CA_C]

        # across peptide
        if L > 1:
            L_C_Np1 = bond_len(C[:, :-1, :], N[:, 1:, :], self.bond_lengths["C-N+1"]).sum()
            loss_terms += [self.w_len * L_C_Np1]
            counts += [C[:, :-1, :].numel()//3]

        # ----- bond angles (stable) -----
        def angle_sq(a, b, c, target_deg):
            theta = _angle_stable(a, b, c)
            t = target_deg * torch.pi / 180.0
            return (theta - t) ** 2

        # N-CA-C (per residue)
        A_N_CA_C = angle_sq(N, CA, C, self.bond_angles["N-CA-C"]).sum()
        loss_terms += [self.w_ang * A_N_CA_C]
        counts += [L * B]

        if L > 1:
            # CA-C-N+1 at C_i
            A_CA_C_Np1 = angle_sq(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :], self.bond_angles["CA-C-N+1"]).sum()
            loss_terms += [self.w_ang * A_CA_C_Np1]
            counts += [B * (L - 1)]
            # C-N-CA at N_i (with C_{i-1}, N_i, CA_i)
            A_Cm1_N_CA = angle_sq(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :], self.bond_angles["C-N-CA"]).sum()
            loss_terms += [self.w_ang * A_Cm1_N_CA]
            counts += [B * (L - 1)]

        # ----- dihedrals -----
        if L > 1:
            # ω: CA_i—C_i—N_{i+1}—CA_{i+1} ~ 180°
            omega = _dihedral(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])  # (B, L-1)
            t_om = self.dihedral_targets["omega"] * torch.pi / 180.0
            # wrap error into [-pi, pi] before squaring (handles ±π equivalence)
            d_om = torch.atan2(torch.sin(omega - t_om), torch.cos(omega - t_om))
            L_omega = (d_om ** 2).sum()
            loss_terms += [self.w_omega * L_omega]
            counts += [B * (L - 1)]

            # optional φ prior: C_{i-1}—N_i—CA_i—C_i
            if self.w_phi > 0.0:
                phi = _dihedral(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :], C[:, 1:, :])
                if self.dihedral_targets["phi"] is not None:
                    t_phi = self.dihedral_targets["phi"] * torch.pi / 180.0
                    d_phi = torch.atan2(torch.sin(phi - t_phi), torch.cos(phi - t_phi))
                    L_phi = (d_phi ** 2).sum()
                else:
                    # zero-mean soft prior (very weak) if no target provided
                    L_phi = (torch.atan2(torch.sin(phi), torch.cos(phi)) ** 2).sum()
                loss_terms += [self.w_phi * L_phi]
                counts += [B * (L - 1)]

            # optional ψ prior: N_i—CA_i—C_i—N_{i+1}
            if self.w_psi > 0.0:
                psi = _dihedral(N[:, :-1, :], CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])
                if self.dihedral_targets["psi"] is not None:
                    t_psi = self.dihedral_targets["psi"] * torch.pi / 180.0
                    d_psi = torch.atan2(torch.sin(psi - t_psi), torch.cos(psi - t_psi))
                    L_psi = (d_psi ** 2).sum()
                else:
                    L_psi = (torch.atan2(torch.sin(psi), torch.cos(psi)) ** 2).sum()
                loss_terms += [self.w_psi * L_psi]
                counts += [B * (L - 1)]

        # ----- aggregate with sensible normalization -----
        total = 0.0
        tot_count = 0
        for term, cnt in zip(loss_terms, counts):
            total = total + term / max(cnt, 1)
            tot_count += 1
        # average over term groups and batch
        return total / max(tot_count, 1) / B
class FixedSumProjector:
    """
    L2 projection onto the hyperplane { x : 1^T x = target_sum }.

    For each row x in R^n, the projection is:
        x_proj = x + ((target_sum - sum(x)) / n) * 1

    Notes:
      - No box constraints (values may go <0 or >1).
      - Batched and 1D inputs supported.
    """
    def __init__(self, target_sum=1.0):
        self.target_sum = float(target_sum)

    @staticmethod
    def _project_batch(X, s):
        """
        X: (B, n) tensor
        s: float (same target sum for all rows)
        returns X_proj: (B, n)
        """
        B, n = X.shape
        # Per-row correction: ((s - row_sum) / n)
        row_sums = X.sum(dim=1, keepdim=True)                # (B,1)
        corr = (s - row_sums) / float(n)                     # (B,1)
        return X + corr                                      # broadcast add

    def constraint_residual(self, x):
        """
        Differentiable residual for the fixed-sum hyperplane.
        """
        was_1d = x.dim() == 1
        X = x.view(1, -1) if was_1d else x
        target = torch.as_tensor(self.target_sum, device=X.device, dtype=X.dtype)
        return X.sum(dim=1, keepdim=True) - target

    def project(self, x):
        """
        x: shape (n,) or (B, n)
        Returns:
          x_proj: same shape as x, with each row sum == target_sum (up to FP eps)
          residual: max |sum(row) - target_sum|
          normal: unit all-ones vector of length n
        """
        # Ensure tensor, keep dtype/device
        X_in = x
        was_1d = (X_in.dim() == 1)
        X = X_in.view(1, -1) if was_1d else X_in
        B, n = X.shape

        # Closed-form projection
        Xproj = self._project_batch(X, self.target_sum)

        # Diagnostics
        sums = Xproj.sum(dim=1)
        residual = (sums - self.target_sum).abs().max().item()

        normal = torch.ones(n, device=Xproj.device, dtype=Xproj.dtype)
        normal = normal / torch.linalg.vector_norm(normal)

        if was_1d:
            Xproj = Xproj.view(-1)

        return Xproj, residual, normal

import torch
from torch.optim import LBFGS

def _unit(v, eps=1e-8):
    n = v.norm(dim=-1, keepdim=True).clamp_min(eps)  # NOT in-place
    return v / n

def _deg_to_rad(x, device, dtype):
    return torch.as_tensor(x, device=device, dtype=dtype) * torch.pi / 180.0

import torch

def _safe_norm(x, dim=-1, eps=1e-12):
    return torch.sqrt((x * x).sum(dim=dim).clamp_min(eps))

def _cos_angle(a, b, c, eps=1e-12):
    """
    Cosine of angle abc, where b is the vertex.
    a,b,c: (...,3)
    returns: (...,)
    """
    u = a - b
    v = c - b
    u = u / _safe_norm(u, dim=-1, eps=eps).unsqueeze(-1)
    v = v / _safe_norm(v, dim=-1, eps=eps).unsqueeze(-1)
    return (u * v).sum(dim=-1).clamp(-1.0, 1.0)

def _deg_to_rad_tensor(deg, device, dtype):
    return torch.as_tensor(deg, device=device, dtype=dtype) * torch.pi / 180.0

class BatchedProteinGeometryProjectorGN:
    """
    Linearized orthogonal projection onto the constraint manifold defined by:
      - bond lengths: N-CA, CA-C, C-N+1
      - bond angles : N-CA-C, CA-C-N+1
    No dihedrals, no clashes, no other priors.
    """

    def __init__(
        self,
        bond_lengths: dict,
        bond_angles_deg: dict,
        max_iter: int = 5,
        damping: float = 1e-4,
        step_size: float = 1.0,
        tol_max_res: float = 1e-4,
        eps: float = 1e-9,
    ):
        self.bond_lengths = bond_lengths
        self.bond_angles_deg = bond_angles_deg
        self.max_iter = int(max_iter)
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.tol_max_res = float(tol_max_res)
        self.eps = float(eps)

    def _constraints(self, x):
        """
        x: (B,L,3,3) requires_grad=True
        returns:
          c: (B,M) constraint residual vector (target = 0)
          names: list[str] length M (optional; useful for debugging)
        """
        B, L, A, D = x.shape
        assert A == 3 and D == 3

        dev, dt = x.device, x.dtype
        N  = x[:, :, 0, :]
        CA = x[:, :, 1, :]
        C  = x[:, :, 2, :]

        c_list = []
        names = []

        # ---- bond lengths residuals: ||a-b|| - target ----
        def len_res(a, b, target, name):
            r = _safe_norm(a - b, dim=-1, eps=self.eps) - torch.as_tensor(target, device=dev, dtype=dt)
            c_list.append(r)  # (B,L) or (B,L-1)
            names.append(name)

        len_res(N,  CA, self.bond_lengths["N-CA"],   "len(N-CA)")
        len_res(CA, C,  self.bond_lengths["CA-C"],   "len(CA-C)")
        if L > 1:
            len_res(C[:, :-1, :], N[:, 1:, :], self.bond_lengths["C-N+1"], "len(C-N+1)")

        # ---- bond angles residuals in cosine-space: cos(theta) - cos(target) ----
        def ang_cos_res(a, b, c, target_deg, name):
            cos_th = _cos_angle(a, b, c, eps=self.eps)
            cos_t = torch.cos(_deg_to_rad_tensor(target_deg, dev, dt))
            r = cos_th - cos_t
            c_list.append(r)
            names.append(name)

        ang_cos_res(N,  CA, C, self.bond_angles_deg["N-CA-C"], "ang(N-CA-C)")
        if L > 1:
            ang_cos_res(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :],
                        self.bond_angles_deg["CA-C-N+1"], "ang(CA-C-N+1)")

        # Flatten into (B,M)
        # Each entry in c_list is (B, K) where K is L or L-1.
        c = torch.cat([r.reshape(B, -1) for r in c_list], dim=1)
        return c, names

    def _build_J(self, x_var, c):
        """
        Build full Jacobian J = dc/dx (batched).

        x_var: (B,L,3,3) leaf with requires_grad=True  (this is what c was built from)
        c:     (B,M)

        returns: J: (B,M,n) where n = L*3*3
        """
        B = x_var.shape[0]
        n = x_var.numel() // B
        M = c.shape[1]

        J = torch.zeros((B, M, n), device=x_var.device, dtype=x_var.dtype)

        for i in range(M):
            gi = torch.autograd.grad(
                c[:, i].sum(),
                x_var,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,   # important for robustness
            )[0]

            if gi is None:
                # If autograd says unused, treat as zero row
                continue

            J[:, i, :] = gi.reshape(B, -1)

        return J

    def optimize(self, x0):
        """
        x0: (B,L,3,3)
        returns: x_proj same shape
        """
        assert x0.dim() == 4 and x0.shape[2:] == (3, 3)
        B = x0.shape[0]
        x = x0.clone().detach()

        for _ in range(self.max_iter):
            x_var = x.detach().clone().requires_grad_(True)

            c, _ = self._constraints(x_var)   # (B,M)
            max_res = c.abs().amax().item()
            if max_res <= self.tol_max_res:
                x = x_var.detach()
                break

            J = self._build_J(x_var, c)       # (B,M,n)

            A = J @ J.transpose(1, 2)         # (B,M,M)
            A = A + self.damping * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0)

            y = torch.linalg.solve(A, c.unsqueeze(-1)).squeeze(-1)  # (B,M)
            delta = -(J.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1)  # (B,n)

            x_flat = x_var.reshape(B, -1)
            x = (x_flat + self.step_size * delta).reshape_as(x_var).detach()

        return x

# These are standard bond lengths and angles (in Å and degrees)
BOND_CONSTRAINTS = {
    "N-CA": 1.46,
    "CA-C": 1.52,
    "C-N+1": 1.33,
}
ANGLE_CONSTRAINTS_DEG = {
    "N-CA-C": 110.0,
    "CA-C-N+1": 116.0,
}

# ---------- projector wrapper ----------
class ProteinConstraintProjector:
    def __init__(self, device, L=10, compile_opt=False, print_err=False):
        self.bond_lengths = BOND_CONSTRAINTS
        self.bond_angles_deg = ANGLE_CONSTRAINTS_DEG
        self.device = device
        self.L = L
        self.print_err = print_err

        opt = BatchedProteinGeometryProjectorGN(
            bond_lengths=self.bond_lengths,
            bond_angles_deg=self.bond_angles_deg,
            max_iter=5,          # GN steps
            damping=1e-4,        # stabilize JJ^T inversion
            step_size=1.0,       # can drop to 0.5 if you see overshoot
            tol_max_res=1e-4,
        )
        self.optimizer = torch.compile(opt) if compile_opt else opt

    def _as_coords4d(self, coords):
        A, D = 3, 3
        was_flat = False

        if coords.dim() == 4:
            B, L, A_in, D_in = coords.shape
            assert A_in == A and D_in == D, f"Expected (B, L, 3, 3), got {coords.shape}"
            assert L == self.L, f"L mismatch: got {L}, projector expects L={self.L}"
            return coords, was_flat

        if coords.dim() == 2:
            B, F = coords.shape
            expected = self.L * A * D
            assert F == expected, f"Got F={F}, expected {expected} (L={self.L}, A=3, D=3)"
            return coords.reshape(B, self.L, A, D), True

        if coords.dim() == 1:
            F = coords.shape[0]
            expected = self.L * A * D
            assert F == expected, f"Got F={F}, expected {expected} (L={self.L}, A=3, D=3)"
            return coords.reshape(1, self.L, A, D), True

        raise ValueError(f"Unsupported coords shape: {coords.shape}")

    def constraint_residual(self, coords):
        """
        Differentiable bond-length and bond-angle residuals evaluated at coords.
        """
        coords = coords.to(self.device, non_blocking=True).contiguous()
        coords4d, _ = self._as_coords4d(coords)
        residuals, _ = self.optimizer._constraints(coords4d)
        return residuals

    @torch.no_grad()
    def project(self, coords):
        """
        coords: (B, L, 3, 3) or (B, L*3*3) or (L*3*3,)
        Returns:
          coords4d_proj: (B, L, 3, 3)
          mean_err: float
          coords_flat_proj: (B, L*3*3) if input was flat; else None
        """
        coords = coords.to(self.device, non_blocking=True).contiguous()
        coords4d, was_flat = self._as_coords4d(coords)
        orig = coords4d.clone()

        # GN projection (needs grads internally)
        with torch.enable_grad():
            coords4d = self.optimizer.optimize(coords4d)

        diff = (coords4d - orig)
        norms = diff.norm(dim=-1).mean(dim=(-1, -2))
        mean_err = norms.mean().item()
        if self.print_err:
            print(f"[ProteinConstraintProjector] mean Δ = {mean_err:.6f}")

        coords4d_proj = coords4d.detach()
        coords_flat_proj = coords4d_proj.reshape(coords4d_proj.shape[0], -1) if was_flat else None
        return coords4d_proj, mean_err, coords_flat_proj

if __name__ == "__main__":
    unittest.main()
