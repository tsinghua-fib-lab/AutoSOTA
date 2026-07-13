"""Defines a bunch of metrics for mixing / separability."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import silhouette_score
import numpy as np
import torch
from torch import optim
from MLP_variants import MLP
from tqdm import trange

class MINE(MLP):
    # TODO: Add flexibility to opt if needed
    def __init__(
        self,
        in_dim,
        hidden_dims,
        lr=1e-3,
        bias=True,
        extras=None,
        verbose=False
    ):
        MLP.__init__(self, in_dim, hidden_dims, 1, bias=bias, extras=extras)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.verbose = verbose

    def forward(self, x, y):
        return MLP.forward(self, torch.cat((x, y), dim=1))

    def mutual_information(self, x, y, steps=150):

        itr = range(steps)
        
        if self.verbose:
            itr  = trange(steps)
        
        for _ in itr:
            joint = self(x, y)
            y_shuffle = y[torch.randperm(y.size(0))]
            marg = self(x, y_shuffle)
    
            mi_loss = - (torch.mean(joint) - torch.log(torch.mean(torch.exp(marg))))
            
            self.optimizer.zero_grad()
            mi_loss.backward()
            self.optimizer.step()
        
        return -mi_loss.item()  # Approximation of I(x; y)


# KNN Classifier Error with euclidean distance
def knn_error(X: torch.Tensor, y: torch.Tensor, n_neighbors: int = 30, fast: bool = True):
    """Calculates KNN classifier accuracy on embeddings.

    Trains a KNN classifier on `y` labels using the inputs `X`.

    Parameters
    ----------
    X : :class:`~torch.Tensor`
        Input features / embeddings.

    y : :class:`~torch.Tensor`
        Input labels.

    n_neighbors : :class:`int`, default: 30
        K for the KNN classifier.

    fast : `bool`, default: False
        Calls scoring on the training set using :class:`~sklearn.neighbors.KNeighborsClassifier`, which in fact does not equal :func:`~sklearn.model_selection.cross_val_score` with :class:`~sklearn.model_selection.LeaveOneOut`.

    Returns
    -------
    :class:`float`
        Accuracy of the KNN classifier.
    """
    if y.shape[-1] == 1:
        y = y.squeeze(-1)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)

    if fast:
        return knn.score(X,y)
    
    return cross_val_score(knn, X, y, cv=LeaveOneOut()).mean()


def _run_kmeans(X: torch.Tensor, y: torch.Tensor):

    if y.shape[-1] == 1:
        y = y.squeeze(-1)

    y_pred = KMeans(
        n_clusters=len(np.unique(y)),
        init="k-means++",
        n_init="auto",
        max_iter=300,
        random_state=42,
    ).fit_predict(X) 

    return y, y_pred

def kmeans_nmi(X: torch.Tensor, y: torch.Tensor):
    """Calculates K-Means NMI for embeddings `X`.

    Parameters
    ----------
    X : :class:`~torch.Tensor`
        Input features / embeddings.

    y : :class:`~torch.Tensor`
        Input labels.

    Returns
    -------
    float
        NMI for original labels and K-Means clusters.
    """

    return normalized_mutual_info_score(*_run_kmeans(X,y))


def kmeans_ari(X: torch.Tensor, y: torch.Tensor):
    """Calculates K-Means ARI for latent representations.

    Parameters
    ----------
    X : :class:`~torch.Tensor`
        Input features / embeddings.

    y : :class:`~torch.Tensor`
        Input labels.

    Returns
    -------
    float
        ARI for original labels and K-Means clusters.
    """

    return adjusted_rand_score(*_run_kmeans(X,y))


def calc_asw(X: torch.Tensor, y: torch.Tensor):
    """Calculates the mean silhouette coefficient of all samples.

    Parameters
    ----------
    X : :class:`~torch.Tensor`
        Input features / embeddings.

    y : :class:`~torch.Tensor`
        Input labels.

    Returns
    -------
    float
        Mean silhouette score.
    """

    if y.shape[-1] == 1:
        y = y.squeeze(-1)

    return silhouette_score(X=X, labels=y)