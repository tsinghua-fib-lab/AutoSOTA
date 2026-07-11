import numpy as np
import scipy
import scipy.sparse as sp
import numpy.linalg as LA
import math


def norm_fro(X):
    """Compute the Frobenius norm of a matrix
    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix
    Returns
    -------
    float
    """
    if sp.issparse(X):  # scipy.sparse array
        return math.sqrt(X.multiply(X).sum())
    else:  # numpy array
        return LA.norm(X)


def get_l2_error(X, U, V):
    return LA.norm(X - np.matmul(U,V.T))

def project_error(X, U, V, known_mask):
    re_error = LA.norm(np.multiply(known_mask, X - np.matmul(U, V.T)))
    re_error_relative = re_error/LA.norm(np.multiply(known_mask, X))
    return re_error, re_error_relative

def norm_fro_err(X, W, H, norm_X):
    """Compute the approximation error in Frobeinus norm
    norm(X - W.dot(H.T)) is efficiently computed based on trace() expansion
    when W and H are thin.
    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix, shape (m,n)
    W : numpy.array, shape (m,k)
    H : numpy.array, shape (n,k)
    norm_X : precomputed norm of X
    Returns
    -------
    float
    """
    sum_squared = (
        norm_X * norm_X
        - 2 * np.trace(H.T.dot(X.T.dot(W)))
        + np.trace((W.T.dot(W)).dot(H.T.dot(H)))
    )
    return math.sqrt(np.maximum(sum_squared, 0))


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
    if sp.issparse(X):
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


def normalize_column(X, by_norm="2"):
    """Column normalization
    Scale the columns of X so that they have unit l2-norms.
    The normalizing coefficients are also returned.
    Side Effect
    -----------
    X given as input are changed and returned
    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix
    Returns
    -------
    ( X, weights )
    X : normalized matrix
    weights : numpy.array, shape k
    """
    if sp.issparse(X):
        weights = column_norm(X, by_norm)
        # construct a diagonal matrix
        dia = [1.0 / w if w > 0 else 1.0 for w in weights]
        N = X.shape[1]
        r = np.arange(N)
        c = np.arange(N)
        mat = sp.coo_matrix((dia, (r, c)), shape=(N, N))
        Y = X.dot(mat)
        return (Y, weights)
    else:
        norms = column_norm(X, by_norm)
        toNormalize = norms > 0
        X[:, toNormalize] = X[:, toNormalize] / norms[toNormalize]
        weights = np.ones(norms.shape)
        weights[toNormalize] = norms[toNormalize]
        return (X, weights)


def sparse_remove_row(X, to_remove):
    """Delete rows from a sparse matrix
    Parameters
    ----------
    X : scipy.sparse matrix
    to_remove : a list of row indices to be removed.
    Returns
    -------
    Y : scipy.sparse matrix
    """
    if not sp.isspmatrix_lil(X):
        X = X.tolil()

    to_keep = [i for i in iter(range(0, X.shape[0])) if i not in to_remove]
    Y = sp.vstack([X.getrowview(i) for i in to_keep])
    return Y


def sparse_remove_column(X, to_remove):
    """Delete columns from a sparse matrix
    Parameters
    ----------
    X : scipy.sparse matrix
    to_remove : a list of column indices to be removed.
    Returns
    -------
    Y : scipy.sparse matrix
    """
    B = sparse_remove_row(X.transpose().tolil(), to_remove).tocoo().transpose()
    return B


def multiply(X, Y):
    """
    Compute element-wise multiplication of matrices :param:`X` and :param:`Y`.

    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    Asume X and Y are same type: either sparse or np.mat
    """
    if sp.isspmatrix(X) and sp.isspmatrix(Y):
        return X.multiply(Y)
    else:
        return np.multiply(np.array(X), np.array(Y))
