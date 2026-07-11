
import os
from copy import deepcopy
import numpy as np
import numpy.linalg as LA

from nmf_algos.utils.utils import generate_result_dir, generate_data_name
from nmf_algos.utils.ENMF_utils import cal_po_dist

def update_W3(W_2, R_2, P, t):
    tmp_mat = W_2@R_2 - P
    W_3 = deepcopy(tmp_mat)
    W_3[tmp_mat < -1.0/t] = (tmp_mat + 1.0/t)[tmp_mat < -1.0/t]
    W_3[(tmp_mat > -1.0/t) & (tmp_mat < 0)] = 0
    return W_3


def update_R1(W_left, W_right, dual_v):
    tmp_mat =  W_left.T@W_right + W_left.T@dual_v
    U,s,V = LA.svd(tmp_mat)
    R = np.matmul(U,V)
    return R


def update_W2(W_1, W_3, R_2, Delta, P, N, gamma, t, n_dim, m_dim, r_dim):
    """
    W_1: (n+m, r) the first n rows is W11, the following m rows is W12
    """
    if m_dim > 0:
        W_11 = W_1[:n_dim,:]
        W_12 = W_1[n_dim:(n_dim + m_dim),:]

        W_31 = W_3[:n_dim,:]
        W_32 = W_3[n_dim:(n_dim + m_dim),:]

        N_1 = N[:n_dim,:]
        N_2 = N[n_dim:(n_dim + m_dim),:]

        P_1 = P[:n_dim,:]
        P_2 = P[n_dim:(n_dim + m_dim),:]

        W_21 = gamma*(W_11@Delta - N_1) + t*(W_31@R_2.T + P_1@R_2.T)
        W_21 = 1.0/(gamma + t) * W_21


        W_22 = gamma*(W_12@LA.inv(Delta) - N_2) + t*(W_32@R_2.T + P_2@R_2.T)
        W_22 = 1.0/(gamma + t) * W_22

        W_2 = np.vstack((W_21,W_22))
    else:
        W_11 = W_1[:n_dim,:]
        W_31 = W_3[:n_dim,:]
        N_1 = N[:n_dim,:]
        P_1 = P[:n_dim,:]

        W_21 = gamma*(W_11@Delta - N_1) + t*(W_31@R_2.T + P_1@R_2.T)
        W_21 = 1.0/(gamma + t) * W_21

        W_2 = W_21
        
    return W_2
    
def update_W1(W_svd, R_1, W_2, M, N, gamma, rho, Delta, n_dim, m_dim,r_dim):
    """
    W_svd: stacked U*, V*
    """
    if m_dim > 0 :
        Wsvd_R = W_svd @ R_1
        UsvdR = Wsvd_R[:n_dim,:]
        VsvdR = Wsvd_R[n_dim:(n_dim + m_dim),:]
        W_21 = W_2[:n_dim,:]
        W_22 = W_2[n_dim:(n_dim + m_dim),:]
        N_1 =  N[:n_dim,:]
        N_2 =  N[n_dim:(n_dim + m_dim),:]
        M_1 =  M[:n_dim,:]
        M_2 =  M[n_dim:(n_dim + m_dim),:]
        tmp_res = rho* (UsvdR - M_1) + gamma * (W_21 +  N_1) @ Delta
        delta_arr = np.diag(Delta)
        delta_div1 = gamma * np.multiply(delta_arr,  delta_arr) + rho
        W_11 = tmp_res @ (np.diag(1/delta_div1))
        
        delta_div2 = gamma * np.multiply(1/delta_arr, 1/delta_arr) + rho
        tmp_res2 = rho* (VsvdR - M_2) + gamma * (W_22 +  N_2) @ np.diag(1/delta_arr)
        W_12 = tmp_res2 @ (np.diag(1/delta_div2))

        W_1 = np.vstack((W_11,W_12))
    else:
        Wsvd_R = W_svd @ R_1
        UsvdR = Wsvd_R[:n_dim,:]
        W_21 = W_2[:n_dim,:]
        N_1 =  N[:n_dim,:]
        M_1 =  M[:n_dim,:]
        M_2 =  M[n_dim:(n_dim + m_dim),:]
        tmp_res = rho* (UsvdR - M_1) + gamma * (W_21 +  N_1) @ Delta
        delta_arr = np.diag(Delta)
        delta_div1 = gamma * np.multiply(delta_arr,  delta_arr) + rho
        W_11 = tmp_res @ (np.diag(1/delta_div1))
        W_1 = W_11

    return W_1
    

def update_delta(W1, W2, N, n_dim, m_dim, r_dim):
    """
    Fill delta solving 4th order equation
    ### Create equation array, solve them sequentially
    """
    if m_dim > 0:
        W_11 = W1[:n_dim,:]
        W_12 = W1[n_dim:(n_dim + m_dim),:]
        W_21 = W2[:n_dim,:]
        W_22 = W2[n_dim:(n_dim + m_dim),:]
        N_1 =  N[:n_dim,:]
        N_2 =  N[n_dim:(n_dim + m_dim),:]

        A_arr = np.sum(np.multiply(W_11, W_11), axis=0)
        B_arr = -np.sum(np.multiply(W_21 + N_1, W_11), axis=0)
        D_arr = np.sum(np.multiply(W_22 + N_2, W_12), axis=0)
        E_arr = -np.sum(np.multiply(W_12, W_12), axis=0)

        tmp_res = []
        for i in range(r_dim):
            candi_sol = np.roots([A_arr[i],B_arr[i], 0, D_arr[i], E_arr[i]])
            real_sol = candi_sol[np.isreal(candi_sol)]
            #print("real solution and all the solutions", real_sol, candi_sol)
            ### get the real part from complex number format
            tmp_res.append(np.max(real_sol).real)

        Delta = np.diag(tmp_res)
        
    else:
        A_arr = np.sum(np.multiply(W1, W1), axis=0)
        AB_arr = np.sum(np.multiply(W1, W2), axis=0)
        AC_arr = np.sum(np.multiply(W1, N), axis=0)
        tmp_res = (AC_arr + AB_arr)/(A_arr) 
        Delta = np.diag(tmp_res)
        
    return Delta


def admm_gen_rot(U_star,V_star,tolerance,max_iter,rho,gamma,t, mu, tau_inc, tau_dec, mode=0):
    """
    mode:0 means adpative version
    f + rho/2(W1, ) + gamma/2(W2) +  t/2(w3)
    """
    
    #Initializations and shape extraction
    
    n_dim, r_dim = U_star.shape
    m_dim,_ = V_star.shape
    
    residual = np.inf
    residual_vec = []
    cnt = 0
    
    W_2 = np.random.rand((n_dim + m_dim), r_dim)
    R_2 = np.eye(r_dim)
    R_1 = np.eye(r_dim)
    P = np.random.rand((n_dim + m_dim), r_dim)
    N = np.random.rand((n_dim + m_dim), r_dim)
    M = np.random.rand((n_dim + m_dim), r_dim)
    Delta = np.eye(r_dim)
    W_svd = np.vstack((U_star,V_star))

    while (residual > tolerance) and (cnt < max_iter):
        cnt += 1
        ## Block 1
        W_1 = update_W1(W_svd, R_1, W_2, M, N,gamma, rho, Delta, n_dim,m_dim,r_dim)
        W_3 = update_W3(W_2,R_2,P,t)
        
        ## Block 2
        W_2 = update_W2(W_1,W_3,R_2, Delta,P,N,gamma,t,n_dim,m_dim,r_dim)
        R_1 = update_R1(W_svd,W_1,M)
        
        ## Block 3
        #R_2 = update_R1(W_2,W_3,P)
        R_2 = np.eye(r_dim)
        Delta = update_delta(W_1, W_2, N, n_dim, m_dim, r_dim)
        
        ## Updating the dual variables
        
        M = M + (W_1 - W_svd@R_1)
        P = P + (W_3 - W_2@R_2)
        tmp_res_p = LA.norm(W_3 - W_2@R_2)
        tmp_res_d = LA.norm(t*(-W_3 + W_2@R_2))
        
        N_1 = N[:n_dim, :]
        N_2 = N[n_dim:(n_dim + m_dim), :]
        N_1 += W_2[:n_dim, :] - W_1[:n_dim, :] @ Delta
        N_2 += W_2[n_dim:(n_dim + m_dim), :] - W_1[n_dim:(n_dim + m_dim), :] @ LA.inv(Delta)
        
        N = np.vstack((N_1,N_2))
        residual = LA.norm(W_3 - W_2@R_2)
        residual_vec.append(residual)
        
        bool_1 = ((tmp_res_p) > (mu * tmp_res_d))
        bool_2 = ((tmp_res_d) > (mu * tmp_res_p))
        # Updating rho: adaptive rho or fixed
        if mode == 1:
            t = 1 * t
        else:
            if bool_1:
                t = tau_inc * t
            elif bool_2:
                t /=  tau_dec
            else:
                t = 1 * t
    return W_3, residual_vec, R_1, R_2, Delta

def result_analysis(U_star, V_star, R_1, R_2, Delta, X_data, W_3):
    m_dim, r_dim = U_star.shape
    n_dim, _  = V_star.shape
    
    U_new = U_star @ R_1 @ Delta @ R_2
    V_new = (V_star @ R_1 @ LA.inv(Delta) @ R_2).T
    print("Distance to PO using Rotation, scaling and rotation", cal_po_dist(U_new, V_new ))
    print("Reconstruction error using Rotation, scaling and rotation", LA.norm(X_data - U_new@V_new))
    tmp_W = W_3[:m_dim, :] @ (W_3[m_dim : (m_dim +  n_dim), :].T)
    print(tmp_W.shape)
    print("Distance to PO using W3", cal_po_dist(W_3[:m_dim, :],W_3[m_dim : (m_dim +  n_dim), :] ))
    print("Reconstruction error Using W3", LA.norm(X_data - tmp_W))
    
def test():
    # Use exact eNMF result. 
    max_iter = 80000
    rho = 0.98
    gamma = 0.99
    t = 0.99
    tolerance = 0.0001
    tau_inc = 2
    tau_dec = 0.8
    mu = 2
    
    project_dir = os.path.join(os.getcwd())
    #latent_dim = 10
    #dataset_name = "exacts_RSR_50_40_10_0.1"
    latent_dim = 20
    dataset_name = "exacts_RSR_100_80_20_0.1"
    method_name = "ENMF"
    f_data_dir = generate_result_dir(dataset_name, method_name=method_name, latent_dim=latent_dim, iter=1)
    f_name = generate_data_name(dataset_name, method_name=method_name, latent_dim=latent_dim)
    data_path = os.path.join(project_dir, "Results", f_data_dir,f_name)
    
    exacts_data = np.load(data_path, allow_pickle=True).all()
   
    X_data = exacts_data['X']
    U_star = exacts_data['U_eig']
    V_star = exacts_data['V_eig']
    U_rotate = exacts_data['U_rotate']
    V_rotate = exacts_data['V_rotate']
    print("One Rotation, PO distance: ", cal_po_dist(U_rotate, V_rotate))
    # print(X_data.shape, U_star.shape, V_star.shape)
    W_3, res_vect, R_1, R_2, Delta = admm_gen_rot(U_star, V_star, tolerance, max_iter, rho, gamma, t, mu, tau_inc, tau_dec, mode=1)
    result_analysis(U_star, V_star, R_1, R_2, Delta, X_data, W_3)


if __name__ == "__main__":
    test()