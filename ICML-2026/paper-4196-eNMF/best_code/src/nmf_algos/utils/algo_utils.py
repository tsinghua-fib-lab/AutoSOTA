import numpy as np
import copy
import numpy.linalg as LA


def local_min(X, U, V):
    """ " Check KKT condition."""
    t1 = np.matmul(U, np.transpose(V)) - X
    gradU = np.matmul(t1, V)
    gradV = np.matmul(np.transpose(t1), U)
    gradW = np.vstack((gradU, gradV))
    W = np.vstack((U, V))
    out_mat = np.multiply(W, gradW)
    out_mat_abs = np.absolute(out_mat)
    return np.max(out_mat_abs), np.sum(out_mat_abs), np.min(gradW)


def NLS(V, W, Hinit, tol, maxiter):
    H = copy.deepcopy(Hinit)
    WtV = np.matmul(W.T, V)
    WtW = np.matmul(W.T, W)
    alpha = 1
    beta = 0.1
    total_inner_iter = 20
    for out_iter in range(maxiter):
        grad = np.matmul(WtW, H) - WtV
        projgrad = LA.norm(grad[(grad < 0) | (H > 0)])
        if projgrad < tol:
            break
        for inner_iter in range(total_inner_iter):
            Hn = np.maximum(H - alpha * grad, np.zeros(H.shape))
            d = Hn - H
            gradd = sum(sum(grad * d))
            dQd = sum(sum(np.matmul(WtW, d) * d))
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0
            if inner_iter == 0:
                decr_alpha = not suff_decr
                Hp = H
            if decr_alpha:
                if suff_decr:
                    H = Hn
                    break
                else:
                    alpha = alpha * beta
            else:
                if (not suff_decr) or np.array_equal(Hp, Hn):
                    H = Hp
                    break
                else:
                    alpha = alpha / beta
                    Hp = Hn
    # if out_iter == maxiter-1:
    # print("max iter in NLS")
    # else:
    # print("exit with iter", out_iter)
    return H, grad, out_iter


def compute_grad(X, U, V):
    gradU = np.matmul(U, np.matmul(V, V.T)) - np.matmul(X, V.T)
    gradV = np.matmul(np.matmul(U.T, U), V) - np.matmul(U.T, X)
    initgrad = LA.norm(np.vstack((gradU, gradV.T)))
    return gradU, gradV, initgrad


def calculate_obj_NMF(X, U, V, trXTX):
    # X[m,n], U[m,r], V[n,r]
    UTU = U.T @ U
    VTV = V.T @ V
    UTX = U.T @ X
    obj = np.sqrt(trXTX + np.trace(VTV @ UTU) - 2 * np.trace(UTX @ V))
    return obj


def HALS_iter_solver(X, U, V, r, eps, use_nesterov=True):
    # HALS with lightweight Nesterov momentum (blind extrapolation, beta=0.3)
    # Cost: O(mr+nr) extra for vector ops, negligible vs O(mnr) products
    beta = 0.5 if use_nesterov else 0.0

    # V-update (U fixed)
    XTU = X.T @ U
    UTU = U.T @ U
    V_prev = V.copy() if beta > 0 else None
    for kk in range(0, r):
        temp_vec = V[:, kk] + XTU[:, kk] - V @ UTU[:, kk]
        V[:, kk] = np.maximum(temp_vec, eps)

    if beta > 0 and V_prev is not None:
        V = np.maximum(V + beta * (V - V_prev), eps)

    # U-update (V fixed)
    XV = X @ V
    VTV = V.T @ V
    U_prev = U.copy() if beta > 0 else None
    for kk in range(0, r):
        temp_vec = U[:, kk] * VTV[kk, kk] + XV[:, kk] - U @ VTV[:, kk]
        U[:, kk] = np.maximum(temp_vec, eps)
        ss = LA.norm(U[:, kk])
        if ss > 0:
            U[:, kk] = U[:, kk] / ss

    if beta > 0 and U_prev is not None:
        U_extrap = np.maximum(U + beta * (U - U_prev), eps)
        for kk in range(r):
            ss = LA.norm(U_extrap[:, kk])
            if ss > 0:
                U_extrap[:, kk] = U_extrap[:, kk] / ss
        U = U_extrap

    return U, V
