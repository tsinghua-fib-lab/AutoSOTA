"""This file contains the helper functions for the enmf algorithm."""

import time
import copy
import numpy as np
import scipy.sparse as sps
from numpy import linalg as LA
from scipy.sparse.linalg import svds

from .algo_utils import HALS_iter_solver, calculate_obj_NMF


## Helper functions for Step 1
def flip_svd(U, V, ignore_V=True):
    """
    Given svd solution, decide whether flip the U, V column sign to minimize distance to PO
    """
    U_pos = (U + np.abs(U)) / 2
    U_neg = (np.abs(U) - U) / 2
    if ignore_V:
        Pos_sum = np.sum(U_pos, axis=0)
        Neg_sum = np.sum(U_neg, axis=0)
    else:
        Pos_sum = np.sum(U_pos, axis=0) + np.sum((V + np.abs(V)) / 2, axis=0)
        Neg_sum = np.sum(U_neg, axis=0) + np.sum((np.abs(V) - V) / 2, axis=0)
    s_sign = np.ones(U_pos.shape[1])
    s_sign[Pos_sum < Neg_sum] = -1
    # print(Pos_sum, Neg_sum, s_sign)
    return s_sign


def check_po_distance_from_SVD_factors(U, V):
    u_dist, v_dist = 0, 0
    if not np.all(U >= 0):
        u_dist = np.sum(U[U < 0])
    if not np.all(V >= 0):
        v_dist = np.sum(V[V < 0])
    return u_dist + v_dist


def flip_factor_based_on_norm(U, V, s):
    """
    Given svd solution, decide whether flip the U, V column sign to minimize norm
    """

    def divide_with_mask(a, b, mask):
        return np.divide(a, b, out=np.zeros_like(a), where=mask)

    U_pos = (U + np.abs(U)) / 2
    U_neg = (np.abs(U) - U) / 2
    V_pos = (V + np.abs(V)) / 2
    V_neg = (np.abs(V) - V) / 2
    U_pos_norm = LA.norm(U_pos, axis=0)
    V_pos_norm = LA.norm(V_pos, axis=0)
    U_neg_norm = LA.norm(U_neg, axis=0)
    V_neg_norm = LA.norm(V_neg, axis=0)
    pos_norm = U_pos_norm * V_pos_norm
    neg_norm = U_neg_norm * V_neg_norm

    selection_mask = pos_norm >= neg_norm
    print("selection_mask", selection_mask)
    updated_U = np.where(
        selection_mask,
        U * np.sqrt(divide_with_mask(V_pos_norm, U_pos_norm, selection_mask)),
        -U * np.sqrt(divide_with_mask(V_neg_norm, U_neg_norm, ~selection_mask)),
    )
    updated_V = np.where(
        selection_mask,
        V * np.sqrt(divide_with_mask(U_pos_norm, V_pos_norm, selection_mask)),
        -V * np.sqrt(divide_with_mask(U_neg_norm, V_neg_norm, ~selection_mask)),
    )
    return updated_U * np.sqrt(s), updated_V * np.sqrt(s)


def gen_svd_sol(X, r):
    """Generate SVD solution with minimum sum of negative partsgiven X."""
    m, n = X.shape
    # [m,r], [r], [r, n]
    ueig, s, vteig = svds(X, r)
    updated_U, updated_V = flip_factor_based_on_norm(ueig, vteig.T, s)
    U_init = np.dot(ueig, np.diag(np.sqrt(s)))
    V_init = np.dot(vteig.T, np.diag(np.sqrt(s)))
    print("diag s", s, np.sum(np.abs(ueig)), np.sum(np.abs(vteig)))
    print("SVD init:", check_po_distance_from_SVD_factors(U_init, V_init))
    U_star = updated_U
    V_star = updated_V
    print("ueig shape", ueig.shape, np.diag(s).shape, vteig.shape)

    print(
        "previous, current svd error same?",
        LA.norm(X - ueig @ np.diag(s) @ vteig),
        LA.norm(X - updated_U @ updated_V.T),
    )
    print("SVD after flip:", check_po_distance_from_SVD_factors(U_star, V_star))
    return U_star, V_star


## Helper functions for Step 2: ADMM rotation
## Parameters for this step: Step size (rho), if adaptive rho is used then three additional parameters mu, tau_inc and tau_dec


def update_Z(W, R, Y, rho):
    # W:(n+m)r, R: r(r), Y:(n+m)r
    B = np.matmul(W, R) - 1.0 / rho * Y
    Z = copy.deepcopy(B)
    Z[B < -1.0 / rho] = (B + 1.0 / rho)[B < -1.0 / rho]
    Z[(B > -1.0 / rho) & (B < 0)] = 0
    return Z


def update_R(W, Z, Y, rho):
    B = Z + 1.0 / rho * Y
    tmp_mat = np.matmul(np.transpose(W), B)
    U, s, V = np.linalg.svd(tmp_mat)
    R = np.matmul(U, V)
    return R


def admm_rotation(W, rho, tolerance, max_iter, tau_inc, tau_dec, mu, rho_mode, alpha=1.0):
    """
    Tracking the best result in the history
    W:(n+m)r
    rho: rho step size in ADMM
    tolerance: maximal primal/dual residual
    max_iter: max number of steps for ADMM
    tau_inc:, tau_dec, mu: parameters for adpative rho
    rho_mode:   1: constant rho
                other: adaptive rho
    """
    # initialize Y, R
    nm_dim, r_dim = W.shape
    Y = np.ones((nm_dim, r_dim))
    R = np.eye(r_dim)

    tmp_res_d = np.inf
    tmp_res_p = np.inf
    opt_val = np.inf
    cnt = 0
    prev_res_p = np.inf
    div_count = 0
    current_alpha = alpha

    while (tmp_res_p > tolerance) and (tmp_res_d > tolerance) and (cnt < max_iter):
        cnt += 1
        Z = update_Z(W, R, Y, rho)
        tmp_res_d = LA.norm(rho * (np.matmul(W, R) - Z))  # Dual residual

        R = update_R(W, Z, Y, rho)
        tmp_res_p = LA.norm(Z - np.matmul(W, R))  # Primal residual

        # Divergence detection: if primal residual increases 3 consecutive times, reduce alpha
        if cnt > 10 and tmp_res_p > prev_res_p:
            div_count += 1
            if div_count >= 3:
                current_alpha = max(1.0, current_alpha - 0.25)
                div_count = 0
        else:
            div_count = max(0, div_count - 1)
        prev_res_p = tmp_res_p

        pre_y = Y
        Y = Y + current_alpha * rho * (Z - np.matmul(W, R))
        delta_y = LA.norm(Y - pre_y)
        tmp_res = np.matmul(W, R)
        obj_f = sum(tmp_res[tmp_res < 0])

        if obj_f < opt_val:
            best_val = obj_f
            best_R = copy.deepcopy(R)

        # Updating rho: adaptive rho or fixed
        if rho_mode == 1:
            rho = 1 * rho
        else:
            bool_1 = (tmp_res_p) > (mu * tmp_res_d)
            bool_2 = (tmp_res_d) > (mu * tmp_res_p)
            if bool_1:
                rho = tau_inc * rho
            elif bool_2:
                rho = rho / tau_dec
            else:
                rho = 1 * rho
    return best_R, best_val


## Helper functions for Step 3: Feasibility using PBCD
## Parameters for this step: Tolerance (epsilon) and maximum number of iterations (inner_iter)
def row_Uupdate(X, U, V, known_mask=None):
    """
    PBCD: allow input with known mask.
    """
    # U = np.asarray(U)
    # V = np.asarray(V)
    U_nonneg = U >= 0
    if known_mask is not None:
        t1 = np.multiply(known_mask, np.matmul(U, np.transpose(V)) - X)
    else:
        t1 = np.matmul(U, np.transpose(V)) - X
    gradU = np.matmul(t1, V)
    gradU_pos = copy.deepcopy(gradU)

    gradU_pos[U <= 0] = 0

    G_U = np.matmul(gradU_pos, np.transpose(V))
    p_U = np.sum(np.multiply(t1, G_U), axis=1)
    q_U = np.sum(np.multiply(G_U, G_U), axis=1)
    d_U = np.divide(p_U, q_U, out=np.zeros_like(p_U), where=q_U != 0)
    d_U = d_U.reshape((-1, 1))
    U = np.asarray(U - np.multiply(d_U, gradU_pos))
    # negative part of updated U_pos to 0
    U[(U < 0) & (U_nonneg)] = 0
    # calculate the gradient after updating U
    if known_mask is not None:
        t1 = np.multiply(known_mask, np.matmul(U, V.T) - X)
    else:
        t1 = np.matmul(U, np.transpose(V)) - X
    gradU = np.matmul(t1, V)
    projgrad = LA.norm(gradU[((U == 0) & (gradU < 0)) | (U > 0)])
    # print("one row update", np.min(U),projgrad )
    return U, projgrad


def row_Uupdate_speedup(X, U, V, VTV, XV, known_mask=None):
    """
    PBCD: allow input with known mask.
    """
    # U = np.asarray(U)
    # V = np.asarray(V)
    U_nonneg = U >= 0
    gradU = np.matmul(U, VTV) - XV
    if known_mask is not None:
        t1 = np.multiply(known_mask, np.matmul(U, np.transpose(V)) - X)
        gradU = np.matmul(t1, V)

    gradU_pos = copy.deepcopy(gradU)
    gradU_pos[U <= 0] = 0

    G_U = np.matmul(gradU_pos, np.transpose(V))
    p_U = np.sum(np.multiply(gradU_pos, gradU_pos), axis=1)

    q_U = np.sum(np.multiply(G_U, G_U), axis=1)
    d_U = np.divide(p_U, q_U, out=np.zeros_like(p_U), where=q_U != 0)
    d_U = d_U.reshape((-1, 1))
    U = np.asarray(U - np.multiply(d_U, gradU_pos))
    # negative part of updated U_pos to 0
    U[(U < 0) & (U_nonneg)] = 0
    # calculate the gradient after updating U
    gradU = np.matmul(U, VTV) - XV
    projgrad = LA.norm(gradU[((U == 0) & (gradU < 0)) | (U > 0)])
    return U, projgrad


def get_error1(X, U, V):
    # compute reconstruction error: X- UV^T
    return LA.norm(X - matmul(U, V.T))


def move_to_positive_orthant_one_step(
    X, u1, v1, tol, inner_iter, step, known_mask=None
):
    u = copy.deepcopy(u1)
    v = copy.deepcopy(v1)
    if not np.all(u >= 0):
        u[u < 0] = 0
    if not np.all(v >= 0):
        v[v < 0] = 0
    return u, v


def move_to_positive_orthant(
    X, u1, v1, tol, inner_iter, step, dist_po, known_mask=None
):
    """
    This is the vanilla ascent algorithm:
    X = uvT
    tol: exit gradient ratio compared to initial grad
    inner_iter: maximal number of iters for updating one factor
    step: number of steps move to PO.
    known_mask: Mask for known entries. If None, classic ascent algorithm.
    """
    re_error = []
    # if np.sum(u1 < 0) == 0 and np.sum(v1 < 0) == 0:
    if np.all(u1 >= 0) and np.all(v1 >= 0):
        return u1, v1
        # t1 = np.matmul(u1, np.transpose(v1)) - X
        # re_error.append(get_error1(X, u1, v1))
        # return (u1, v1, 0, 0, re_error)
    u = copy.deepcopy(u1)
    v = copy.deepcopy(v1)

    if np.all(u >= 0):
        deltaminU = 0
    else:
        deltaminU = np.abs(np.min(u))
    if np.all(v >= 0):
        deltaminV = 0
    else:
        deltaminV = np.abs(np.min(v))
    # print("delta min val of U, V when entering ascent", deltaminU, deltaminV)

    deltamin_U = deltaminU / step
    deltamin_V = deltaminV / step

    # deltamax = deltamin * 10
    if known_mask is not None:
        t1 = np.multiply(np.matmul(u, v.T) - X, known_mask)
        known_maskT = known_mask.T
    else:
        t1 = np.matmul(u, v.T) - X
        known_maskT = None

    gradU = np.matmul(t1, v)
    gradV = np.matmul(np.transpose(t1), u)
    init_gradU = LA.norm(gradU[((u == 0) & (gradU < 0)) | (u > 0)])
    init_gradV = LA.norm(gradV[((v == 0) & (gradV < 0)) | (v > 0)])
    # very close to orthant and initial gradient is very flat: directly jump there.
    if dist_po > -(10 ** (-4)):
        if init_gradU < 10 ** (-6) and init_gradV < 10 ** (-6):
            return move_to_positive_orthant_one_step(
                X, u1, v1, tol, inner_iter, step, known_mask=None
            )

    j = 0
    # while np.sum(u < 0) > 0 or np.sum(v < 0) > 0:
    while np.any(u < 0) or np.any(v < 0):

        start_t = time.time()

        boolean_v = v < 0
        v[boolean_v] = v[boolean_v] + deltamin_V
        i = 0
        # print("minV", np.min(v))
        projgrad = 10 ** (6)
        utu = u.T @ u
        xtu = X.T @ u

        while (projgrad > tol * init_gradV) and (i < inner_iter):
            # v, projgrad = row_Uupdate(X.T, v, u, known_maskT)
            v, projgrad = row_Uupdate_speedup(X.T, v, u, utu, xtu, known_maskT)
            i = i + 1
        # print("minV Updated", np.min(v))

        boolean_u = u < 0
        u[boolean_u] = u[boolean_u] + deltamin_U
        # print("minU", np.min(u))
        i = 0
        projgrad = 10 ** (6)
        vtv = v.T @ v
        xv = X @ v
        while (projgrad > tol * init_gradU) and (i < inner_iter):
            # u, projgrad = row_Uupdate(X, u, v, known_mask)
            u, projgrad = row_Uupdate_speedup(X, u, v, vtv, xv, known_mask)
            i = i + 1
        j = j + 1
        # print("minU Updated", np.min(u))

        # print("iter j", j, "t:", time.time() - start_t)

    return u, v


def HALS_pos(
    X, trace_XTX, U_mp, V_mp, r, hals_rounds, stop_time, target_error, verbose=True
):
    """
    HALS applied after hitting the boundary
    """
    U = copy.deepcopy(U_mp)
    V = copy.deepcopy(V_mp)
    U, V, _ = normalize_column_pair(U, V)
    # print("before HALS, hit bound norm of U, V:", LA.norm(U), LA.norm(V))
    start_t = time.time()
    cnt = 0
    t = 0
    sys_eps = 10 ** (-16)
    obj = np.inf
    while (cnt < hals_rounds) and (t < stop_time) and (obj > target_error):
        U, V = HALS_iter_solver(X, U, V, r, sys_eps)
        obj = calculate_obj_NMF(X, U, V, trace_XTX)
        if cnt % 200 == 0 and verbose:
            print("obj:", obj)
        t = time.time() - start_t
        cnt += 1
    return (U, V)


## Helper functions for Step 4


def column_norm(X, by_norm="2"):
    """Compute the norms of each column of a given matrix
    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix
    Optional Parameters
    -------------------
    by_norm : '2' for l2-norm, '1' for l1-norm.
              Default is '2'.
    Returns
    -------
    numpy.array
    """
    if sps.issparse(X):
        if by_norm == "2":
            norm_vec = np.sqrt(X.multiply(X).sum(axis=0))
        elif by_norm == "1":
            norm_vec = X.sum(axis=0)
        return np.asarray(norm_vec)[0]
    else:
        if by_norm == "2":
            norm_vec = np.sqrt(np.sum(X * X, axis=0))
        elif by_norm == "1":
            norm_vec = np.sum(X, axis=0)
        return norm_vec


def normalize_column_pair(W, H, by_norm="2"):
    """Column normalization for a matrix pair
    Scale the columns of W and H so that the columns of W have unit norms and
    the product W.dot(H.T) remains the same.  The normalizing coefficients are
    also returned.
    Side Effect
    -----------
    W and H given as input are changed and returned.
    Parameters
    ----------
    W : numpy.array, shape (m,k)
    H : numpy.array, shape (n,k)
    Optional Parameters
    -------------------
    by_norm : '1' for normalizing by l1-norm, '2' for normalizing by l2-norm.
              Default is '2'.
    Returns
    -------
    ( W, H, weights )
    W, H : normalized matrix pair
    weights : numpy.array, shape k
    """
    norms = column_norm(W, by_norm=by_norm)

    toNormalize = norms > 0
    W[:, toNormalize] = W[:, toNormalize] / norms[toNormalize]
    H[:, toNormalize] = H[:, toNormalize] * norms[toNormalize]

    weights = np.ones(norms.shape)
    weights[toNormalize] = norms[toNormalize]
    return (W, H, weights)


def cal_po_dist(UR_star, VR_star):
    return -(np.sum(UR_star[UR_star < 0]) + np.sum(VR_star[VR_star < 0]))
