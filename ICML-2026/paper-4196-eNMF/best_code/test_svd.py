import numpy as np
from scipy.sparse.linalg import svds
from nmf_algos.utils.utils import load_data_matrix

data_path = 'Dataset/face_id_4.npy'
X = load_data_matrix(data_path)
print(f'X shape: {X.shape}')

# Full numpy SVD
U_full, s_full, Vt_full = np.linalg.svd(X, full_matrices=False)
print(f's_full (first 10 largest): {s_full[:10]}')
print(f's_full (last 10): {s_full[-10:]}')

# svds gives smallest first
u_svds, s_svds, vt_svds = svds(X, 10, random_state=42)
print(f's_svds (ascending): {s_svds}')
print(f's_svds (descending): {s_svds[::-1]}')

# Compare with full SVD
print(f'\nFull SVD top-10: {s_full[:10]}')
# svds returns sorted ascending, full SVD is descending
s_svds_desc = s_svds[::-1]
print(f'svds top-10:       {s_svds_desc}')
print(f'Difference: {np.abs(s_full[:10] - s_svds_desc)}')
print(f'Relative diff: {np.abs(s_full[:10] - s_svds_desc) / s_full[:10]}')

# Check reconstruction error with svds
X_svds = u_svds @ np.diag(s_svds) @ vt_svds
print(f'\nReconstruction error (svds): {np.linalg.norm(X - X_svds):.6f}')

# Check reconstruction error with full SVD top-10
X_full_top10 = U_full[:, :10] @ np.diag(s_full[:10]) @ Vt_full[:10, :]
print(f'Reconstruction error (full SVD top-10): {np.linalg.norm(X - X_full_top10):.6f}')
