import time, sys, os, json
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
from nmf_algos import NMF_ENMF
from nmf_algos.utils.utils import load_data_matrix
from nmf_algos.utils.ENMF_utils import gen_svd_sol, admm_rotation, move_to_positive_orthant, HALS_pos
from nmf_algos.utils.algo_utils import calculate_obj_NMF

data_path = 'Dataset/face_id_4.npy'
X = load_data_matrix(data_path)
rank = 10
trace_XTX = np.trace(X.T @ X)

# Try multiple rotation initializations
best_error = np.inf
best_result = None

for trial in range(10):
    # Get SVD
    U_svd, V_svd = gen_svd_sol(X, rank)
    W = np.vstack((U_svd, V_svd))
    
    # Try ADMM with different random initial R
    rng = np.random.default_rng(trial)
    A = rng.normal(size=(rank, rank))
    R_init, _ = np.linalg.qr(A)
    
    # Manually run ADMM with custom init
    nm_dim, r_dim = W.shape
    Y = np.ones((nm_dim, r_dim))
    R = R_init.copy()
    
    rho = 5
    tolerance = 1e-4
    max_iter = 4000
    tau_inc = 1.1
    tau_dec = 1.1
    mu = 2
    rho_mode = 0
    
    tmp_res_d = np.inf
    tmp_res_p = np.inf
    opt_val = np.inf
    cnt = 0
    
    while (tmp_res_p > tolerance) and (tmp_res_d > tolerance) and (cnt < max_iter):
        cnt += 1
        # Z update
        B = np.matmul(W, R) - 1.0 / rho * Y
        Z = B.copy()
        Z[B < -1.0 / rho] = (B + 1.0 / rho)[B < -1.0 / rho]
        Z[(B > -1.0 / rho) & (B < 0)] = 0
        tmp_res_d = np.linalg.norm(rho * (np.matmul(W, R) - Z))
        
        # R update (Procrustes)
        B2 = Z + 1.0 / rho * Y
        tmp_mat = np.matmul(np.transpose(W), B2)
        U_tmp, s_tmp, V_tmp = np.linalg.svd(tmp_mat)
        R = np.matmul(U_tmp, V_tmp)
        tmp_res_p = np.linalg.norm(Z - np.matmul(W, R))
        
        Y = Y + rho * (Z - np.matmul(W, R))
        
        tmp_res = np.matmul(W, R)
        obj_f = np.sum(tmp_res[tmp_res < 0])
        
        if obj_f < opt_val:
            opt_val = obj_f
            best_R = R.copy()
        
        if rho_mode != 1:
            bool_1 = tmp_res_p > mu * tmp_res_d
            bool_2 = tmp_res_d > mu * tmp_res_p
            if bool_1:
                rho = tau_inc * rho
            elif bool_2:
                rho = rho / tau_dec
    
    # Use best rotation
    U_rot = U_svd @ best_R
    V_rot = V_svd @ best_R
    dist_po = opt_val
    
    # Move to PO
    tol_asc = 0.2
    inner_iter_asc = 2
    num_steps = 100
    
    U_mp, V_mp = move_to_positive_orthant(X, U_rot, V_rot, tol_asc, inner_iter_asc, num_steps, dist_po)
    hitmp_error = calculate_obj_NMF(X, U_mp, V_mp, trace_XTX)
    
    # HALS descent (limited)
    U_nmf, V_nmf = HALS_pos(X, trace_XTX, U_mp, V_mp, rank, 100, 5, 0, verbose=False)
    enmf_error = calculate_obj_NMF(X, U_nmf, V_nmf, trace_XTX)
    
    if enmf_error < best_error:
        best_error = enmf_error
        best_result = {'trial': trial, 'error': enmf_error, 'dist_po': dist_po, 'hitmp_error': hitmp_error}
    
    print(f'Trial {trial}: error={enmf_error:.2f}, dist_po={dist_po:.2f}')

print(f'\nBest: trial={best_result["trial"]}, error={best_result["error"]:.2f}')
