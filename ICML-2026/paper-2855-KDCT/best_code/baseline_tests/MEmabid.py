import numpy as np
import torch
import scipy.stats as stats
from past.utils import old_div
from torch.autograd import Function
import scipy.linalg
import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath('..'))
import time
from dataloader import load_data
class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

sqrtm = MatrixSquareRoot.apply

def MatConvert(S, device, dtype):
    """convert the numpy to a torch tensor."""
    S = torch.from_numpy(np.array(S)).to(device, dtype)
    return S

def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * np.dot(X, Y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D

def init_locs_randn(S, N1, n_Anchors, seed=1):
    """Fit a Gaussian to the merged data of the two samples and draw
    n_test_locs points from the Gaussian"""
    # set the seed
    rand_state = np.random.get_state()
    np.random.seed(seed)

    X = S[:N1]
    Y = S[N1:]
    d = X.shape[1]
    # fit a Gaussian in the middle of X, Y and draw sample to initialize T
    xy = np.vstack((X, Y))
    mean_xy = np.mean(xy, 0)
    cov_xy = np.cov(xy.T)
    [Dxy, Vxy] = np.linalg.eig(cov_xy + 1e-3 * np.eye(d))
    Dxy = np.real(Dxy)
    Vxy = np.real(Vxy)
    Dxy[Dxy <= 0] = 1e-3
    eig_pow = 0.9  # 1.0 = not shrink
    reduced_cov_xy = Vxy.dot(np.diag(Dxy ** eig_pow)).dot(Vxy.T) + 1e-3 * np.eye(d)

    T0 = np.random.multivariate_normal(mean_xy, reduced_cov_xy, n_Anchors)
    # reset the seed back to the original
    np.random.set_state(rand_state)
    return T0

def init_locs_2randn(S, N1, n_Anchors, seed=1):
    """Fit a Gaussian to each dataset and draw half of n_test_locs from
    each. This way of initialization can be expensive if the input
    dimension is large.
    """
    # with util.NumpySeedContext(seed=seed):

    rand_state = np.random.get_state()
    np.random.seed(seed)
    if n_Anchors == 1:
        return init_locs_randn(S, N1, n_Anchors, seed)

    X = S[:N1]
    Y = S[N1:]
    d = X.shape[1]

    # fit a Gaussian to each of X, Y
    mean_x = np.mean(X, 0)
    mean_y = np.mean(Y, 0)
    cov_x = np.cov(X.T)
    [Dx, Vx] = np.linalg.eig(cov_x + 1e-3 * np.eye(d))
    Dx = np.real(Dx)
    Vx = np.real(Vx)
    # a hack in case the data are high-dimensional and the covariance matrix
    # is low rank.
    Dx[Dx <= 0] = 1e-3

    # shrink the covariance so that the drawn samples will not be so
    # far away from the data
    eig_pow = 0.9  # 1.0 = not shrink
    reduced_cov_x = Vx.dot(np.diag(Dx ** eig_pow)).dot(Vx.T) + 1e-3 * np.eye(d)
    cov_y = np.cov(Y.T)
    [Dy, Vy] = np.linalg.eig(cov_y + 1e-3 * np.eye(d))
    Vy = np.real(Vy)
    Dy = np.real(Dy)
    Dy[Dy <= 0] = 1e-3
    reduced_cov_y = Vy.dot(np.diag(Dy ** eig_pow).dot(Vy.T)) + 1e-3 * np.eye(d)
    # integer division
    Jx = old_div(n_Anchors, 2)
    Jy = n_Anchors - Jx

    assert Jx + Jy == n_Anchors, 'total test locations is not n_Anchors'
    Tx = np.random.multivariate_normal(mean_x, reduced_cov_x, Jx)
    Ty = np.random.multivariate_normal(mean_y, reduced_cov_y, Jy)
    T0 = np.vstack((Tx, Ty))
    np.random.set_state(rand_state)
    return T0

def torch_cov(input_vec):
    u = torch.mean(input_vec, 0)
    x = input_vec - u
    cov_matrix = torch.matmul(x.t(), x)
    return cov_matrix, u

def HT_Statistics_kernel(S, N1, Anchors, gwidth, device, dtype):
    diag = 1e-5 * torch.eye(Anchors.shape[0]).to(device, dtype)
    Cst = torch.div((len(S) - N1) * N1, len(S)).to(device, dtype)
    D1 = torch.sum(S[:N1] ** 2, 1).reshape((-1, 1)) - 2 * torch.mm(S[:N1], Anchors.t()) + torch.sum(Anchors ** 2, 1).reshape((1, -1))
    D2 = torch.sum(S[N1:] ** 2, 1).reshape((-1, 1)) - 2 * torch.mm(S[N1:], Anchors.t()) + torch.sum(Anchors ** 2, 1).reshape((1, -1))
    D1 = torch.exp(torch.div(-D1, 2.0 * gwidth))
    D2 = torch.exp(torch.div(-D2, 2.0 * gwidth))
    Sig1, u1 = torch_cov(D1)
    Sig2, u2 = torch_cov(D2)
    Sig = torch.div(Sig1 + Sig2, max(len(S) - 2, 1))
    T = Cst * torch.mv(torch.inverse(Sig + diag).t(), u1 - u2).dot(u1 - u2)
    return T  ### return the statistic

def grid_search_gwidth(S, N1, Anchors, list_gwidth, alpha, device, dtype):
    """
    Linear search for the best Gaussian width in the list that maximizes
    the test power, fixing the test locations to T.
    The test power is given by the CDF of a non-central Chi-squared
    distribution.
    return: (best width index, list of test powers)
    """
    # number of test locations
    powers = np.zeros(len(list_gwidth))
    lambs = np.zeros(len(list_gwidth))
    thresh = stats.chi2.isf(alpha, df=Anchors.shape[0])
    # print('thresh: %.3g'% thresh)
    for wi, gwidth in enumerate(list_gwidth):
        # non-centrality parameter
        try:
            # from IPython.core.debugger import Tracer
            # Tracer()()
            lamb = HT_Statistics_kernel(S, N1, Anchors, gwidth, device, dtype).cpu()
            if lamb <= 0:
                # This can happen when Z, Sig are ill-conditioned.
                # print('negative lamb: %.3g'%lamb)
                raise np.linalg.LinAlgError
            if np.iscomplex(lamb):
                # complext value can happen if the covariance is ill-conditioned?
                print('Lambda is complex. Truncate the imag part. lamb: %s' % (str(lamb)))
                lamb = np.real(lamb)

            # print('thresh: %.3g, df: %.3g, nc: %.3g'%(thresh, df, lamb))
            power = stats.ncx2.sf(thresh, df=Anchors.shape[0], nc=lamb)
            powers[wi] = power
            lambs[wi] = lamb
            # print('i: %2d, lamb: %5.3g, gwidth: %5.3g, power: %.4f'
            #       % (wi, lamb, gwidth, power))
        except np.linalg.LinAlgError:
            # probably matrix inverse failed.
            print('LinAlgError. skip width (%d, %.3g)' % (wi, gwidth))
            powers[wi] = np.NINF
            lambs[wi] = np.NINF
    # to prevent the gain of test power from numerical instability,
    # consider upto 3 decimal places. Widths that come early in the list
    # are preferred if test powers are equal.
    besti = np.argmax(np.around(powers, 3))
    return besti, powers

def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and
        there are more slightly more 0 than 1. In this case, the m

    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)

"""calculate the statistic and the vector for inference direction based on original samples"""
def HT_Statistics_Mkernels(S, N1, Anchors, gwidths, Matrixs, device, dtype):
    gwidths = torch.max(gwidths, torch.tensor(10 ** -5).to(device, dtype))
    Mats_A = torch.sum(torch.mul(Matrixs, Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats = torch.sum(torch.mul(Matrixs.transpose(1, 2), Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats_A = torch.sum(torch.mul(Mats_A, Anchors), 1)
    S1_Mats = torch.matmul(S[:N1], Matrixs)
    S1_Mats_S1 = torch.sum(torch.mul(S1_Mats, S[:N1]), 2).t()
    S1_Mats_A = torch.sum(torch.mul(S1_Mats.transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), Anchors), 2)
    A_Mats_S1 = torch.sum(torch.mul(A_Mats.reshape(Anchors.shape[0], -1, Anchors.shape[1]), S[:N1]).transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), 2)
    D1 = S1_Mats_S1 - A_Mats_S1 - S1_Mats_A + A_Mats_A
    S2_Mats = torch.matmul(S[N1:], Matrixs)
    S2_Mats_S2 = torch.sum(torch.mul(S2_Mats, S[N1:]), 2).t()
    S2_Mats_A = torch.sum(torch.mul(S2_Mats.transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), Anchors), 2)
    A_Mats_S2 = torch.sum(torch.mul(A_Mats.reshape(Anchors.shape[0], -1, Anchors.shape[1]), S[N1:]).transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), 2)
    D2 = S2_Mats_S2 - A_Mats_S2 - S2_Mats_A + A_Mats_A

    D1 = torch.max(D1, torch.tensor(0.0).to(device, dtype))
    D2 = torch.max(D2, torch.tensor(0.0).to(device, dtype))
    D1 = torch.exp(torch.div(-D1, 2.0 * gwidths))
    D2 = torch.exp(torch.div(-D2, 2.0 * gwidths))
    Sig1, u1 = torch_cov(D1)
    Sig2, u2 = torch_cov(D2)
    Sig = torch.div(Sig1 + Sig2, max(len(S) - 2, 1))
    diag = 1e-5 * torch.eye(Anchors.shape[0]).to(device, dtype)
    L_inv = torch.inverse(sqrtm(Sig + diag))

    D1 = torch.matmul(L_inv, D1.transpose(0, 1)).transpose(0, 1)
    D2 = torch.matmul(L_inv, D2.transpose(0, 1)).transpose(0, 1)
    diff = torch.mean(D1, dim=0) - torch.mean(D2, dim=0)

    Cst = torch.div((len(S) - N1) * N1, len(S)).to(device, dtype)
    T = Cst * torch.mv(torch.inverse(Sig + diag).t(), u1 - u2).dot(u1 - u2)
    return -T, diff.detach()   ### return the statistic and the vector for inference direction

def MEmabid_OPT(S, N1, n_Anchors, N_epoch, learning_rate, seed, device, dtype, batch_size=None):
    reg = torch.tensor(1e-5).to(device, dtype)

    """initialization for test locations"""
    Anchors = init_locs_2randn(S, N1, n_Anchors, seed + 5)
    Anchors = MatConvert(Anchors, device, dtype)
    med = meddistance(S, 1000)
    list_gwidth = np.hstack(((med ** 2) * (2.0 ** np.linspace(-3, 4, 30))))
    list_gwidth.sort()
    list_gwidth = MatConvert(list_gwidth, device, dtype)

    S = MatConvert(S, device, dtype)

    """initialization for parameter gamma of Mahalanobis kernels"""
    besti, powers = grid_search_gwidth(S, N1, Anchors, list_gwidth, 0.05, device, dtype)
    gwidth = list_gwidth[besti].cpu()
    gwidths = np.repeat(gwidth, n_Anchors)
    gwidths = MatConvert(gwidths, device, dtype)

    """initialization for Mahalanobis matrices of Mahalanobis kernels"""
    M_matrix = np.identity(S.shape[1])
    M_matrixs = np.tile(M_matrix, (n_Anchors, 1)).reshape((-1, S.shape[1], S.shape[1]))
    M_matrixs = MatConvert(M_matrixs, device, dtype)

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    Anchors.requires_grad = True
    gwidths.requires_grad = True
    M_matrixs.requires_grad = True

    optimizer_u = torch.optim.Adam([Anchors] + [gwidths] + [M_matrixs], lr=learning_rate)

    for t in range(N_epoch):
        S1 = S[:N1, :]
        S2 = S[N1:, :]
        epoch = max(min(int(len(S1) / batch_size) * 2, int(len(S2) / batch_size) * 2), 1)
        for i in range(epoch):
            if int(len(S1) / batch_size) * 2 <= 1 or int(len(S2) / batch_size) * 2 <= 1:
                ind1 = np.random.choice(np.arange(len(S1)), min(len(S1), len(S2)), replace=False)
                ind2 = np.random.choice(np.arange(len(S2)), min(len(S1), len(S2)), replace=False)
            else:
                ind1 = np.random.choice(np.arange(len(S1)), int(batch_size / 2), replace=False)
                ind2 = np.random.choice(np.arange(len(S2)), int(batch_size / 2), replace=False)

            X = torch.cat([S1[ind1], S2[ind2]], 0)
            if device == torch.device("cpu"):
                S1 = np.delete(S1, ind1, 0)
                S2 = np.delete(S2, ind2, 0)
            else:
                S1 = torch.index_select(S1, 0, torch.tensor(np.delete(np.arange(len(S1)), ind1, 0), dtype=torch.long).cuda())
                S2 = torch.index_select(S2, 0, torch.tensor(np.delete(np.arange(len(S2)), ind2, 0), dtype=torch.long).cuda())

            loss, _ = HT_Statistics_Mkernels(X, int(len(X) / 2), Anchors, gwidths, M_matrixs, device, dtype)

            optimizer_u.zero_grad()
            loss.backward(retain_graph=True)
            # Update weights using gradient
            optimizer_u.step()

        # map Mahalanobis matrices to the positive-definite cone
        if (t + 1) % 5 == 0:
            with torch.no_grad():
                for j in range(len(M_matrixs)):
                    eigvalues, eigvectors = torch.linalg.eig(M_matrixs[j])
                    eigvalues = torch.max(eigvalues.real, reg)
                    eigvectors = eigvectors.real
                    eigvectors = eigvectors.t().reshape(eigvectors.shape[0], -1, eigvectors.shape[1])
                    M_matrixs[j] = eigvalues[0] * eigvectors[0].t() * eigvectors[0]
                    for i in range(1, len(eigvalues)):
                        M_matrixs[j] += eigvalues[i] * eigvectors[i] * eigvectors[i].t()
        if (t + 1) % 500 == 0 or t == 0:
            print("STAT_value: ", loss.item())

    gwidths = torch.max(gwidths, torch.tensor(10 ** -8).to(device,dtype))
    Mats_A = torch.sum(torch.mul(M_matrixs, Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats = torch.sum(torch.mul(M_matrixs.transpose(1, 2), Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats_A = torch.sum(torch.mul(Mats_A, Anchors), 1)
    S_Mats = torch.matmul(S, M_matrixs)
    S_Mats_S = torch.sum(torch.mul(S_Mats, S), 2).t()
    S_Mats_A = torch.sum(torch.mul(S_Mats.transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), Anchors), 2)
    A_Mats_S = torch.sum(torch.mul(A_Mats.reshape(Anchors.shape[0], -1, Anchors.shape[1]), S).transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), 2)
    D = S_Mats_S - A_Mats_S - S_Mats_A + A_Mats_A
    D = torch.max(D, torch.tensor(0.0).to(device,dtype))
    D = torch.exp(torch.div(-D, 2.0 * gwidths))

    diag = 1e-5 * torch.eye(n_Anchors).to(device, dtype)
    Sig1, u1 = torch_cov(D[:N1])
    Sig2, u2 = torch_cov(D[N1:])
    Sig = torch.div(Sig1+Sig2,max(len(S)-2,1))
    try:
        L_inv = torch.inverse(sqrtm(Sig))
    except:
        L_inv = torch.inverse(sqrtm(Sig+diag))
    D = torch.matmul(L_inv, D.transpose(0, 1)).transpose(0, 1)
    diff = torch.mean(D[:N1],dim=0) - torch.mean(D[N1:],dim=0)
    infer_dire = torch.sign(diff)

    return Anchors.detach(), gwidths.detach(), M_matrixs.detach(), infer_dire.detach()

def MEmabid_TEST(S, N1, Anchors, gwidths, M_matrixs, infer_dire, alpha, beta, device, dtype):
    stat, diff = HT_Statistics_Mkernels(S, N1, Anchors, gwidths, M_matrixs, device, dtype)
    test_flags = diff
    stat = stat.cpu().detach().numpy()
    J, d = Anchors.shape

    pvalue = stats.chi2.sf(-stat, J)
    # pvalue = sf_chi2(J, -stat) ###  calculate the p-value by simulation
    if sum(test_flags * infer_dire) >= 0:
        h = int(pvalue <= beta * alpha)
    else:
        pvalue = stats.chi2.sf(-stat, J)
        h = int(pvalue <= (2 - beta) * alpha)
    return h ## return the test results h


def TST_MEmabid(name, N1, rs, check, n_test, alpha, device, dtype, n_Anchors, beta, N_epoch, batch_size, learning_rate):
    np.random.seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)

    S_train = np.concatenate((X_train, Y_train), axis=0)

    start_time = time.time()
    Anchors, gwidths, M_matrixs, infer_dire = MEmabid_OPT(S_train, N1, n_Anchors, N_epoch, learning_rate, rs, device, dtype, batch_size)
    train_time = time.time() - start_time
    
    H_MEmabid= np.zeros(n_test)
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
    test_time = 0
    # test 
    for k in range(n_test):
        ind_test = np.random.choice(N_test_all, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]

        S_test = np.concatenate((X_test, Y_test), axis=0)
        S_test = MatConvert(S_test, device, dtype)
        
        start_time = time.time()
        h_MEmabid = MEmabid_TEST(S_test, N1, Anchors, gwidths, M_matrixs, infer_dire, alpha, beta, device, dtype)
        test_time += time.time() - start_time
        H_MEmabid[k] = h_MEmabid
    
    return H_MEmabid, train_time, test_time