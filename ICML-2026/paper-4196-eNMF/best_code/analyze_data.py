import numpy as np
from scipy.sparse.linalg import svds
from nmf_algos.utils.utils import load_data_matrix

X_orig = load_data_matrix('Dataset/face_id_4.npy')
X_shifted = X_orig + 255.0

print('Original (no shift):')
print('  shape=%s, min=%s, max=%s' % (X_orig.shape, X_orig.min(), X_orig.max()))
print('  Frobenius norm=%.1f' % np.linalg.norm(X_orig))

print()
print('Shifted (+255):')
print('  shape=%s, min=%s, max=%s' % (X_shifted.shape, X_shifted.min(), X_shifted.max()))
print('  Frobenius norm=%.1f' % np.linalg.norm(X_shifted))

u, s, vt = svds(X_orig, 10, random_state=42)
svd_err_orig = np.linalg.norm(X_orig - u @ np.diag(s) @ vt)
print()
print('SVD rank-10 error (no shift): %.4f' % svd_err_orig)
print('Relative error (no shift): %.6f' % (svd_err_orig / np.linalg.norm(X_orig)))

u2, s2, vt2 = svds(X_shifted, 10, random_state=42)
svd_err_shifted = np.linalg.norm(X_shifted - u2 @ np.diag(s2) @ vt2)
print('SVD rank-10 error (+255): %.4f' % svd_err_shifted)
print('Relative error (+255): %.6f' % (svd_err_shifted / np.linalg.norm(X_shifted)))

print()
print('12400.48 / norm(no shift) = %.4f%%' % (12400.48 / np.linalg.norm(X_orig) * 100))
print('12400.48 / norm(+255) = %.4f%%' % (12400.48 / np.linalg.norm(X_shifted) * 100))
