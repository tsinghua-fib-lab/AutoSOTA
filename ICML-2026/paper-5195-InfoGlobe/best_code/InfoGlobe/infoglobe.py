import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Union, Iterable, Optional, Tuple
from collections.abc import Iterable as Iterable
# from .metrics import beta_div
from .utils import initialize, normalize
from .metrics import fisher_rao_dis, fisher_rao_dis_matrix, fisher_rao_dis_matrix_block, cosine_loss, angle_mse_loss, angle_mae_loss, elastic_net_loss, orthogonal_loss
from tqdm import tqdm
from .constants import eps


class GlobeEmbedding(torch.nn.Module):
    r"""Base class for MMF modules.

    Args:
        rank (int): size of hidden dimension
        A (Tensor or size): size or initial weights of template tensor A
        Q (Tensor or size): size or initial weights of activation tensor Q
        trainable_W (bool):  controls whether template tensor W is trainable when initial weights is given. Default: ``True``
        trainable_H (bool):  controls whether activation tensor H is trainable when initial weights is given. Default: ``True``

    Attributes:
        A (Tensor): the template tensor of the module if corresponding argument is given.
            If size is given, values are initialized non-negatively and follow multinomial distributions
        Q (Tensor): the activation tensor of the module if corresponding argument is given.
            If size is given, values are initialized non-negatively and follow multinomial distributions

       """
    __constants__ = ['rank']
    __annotations__ = {'A': Optional[Tensor],
                       'Q': Optional[Tensor],}

    rank: int
    A: Optional[Tensor]
    Q: Optional[Tensor]

    def __init__(self,
                 rank: int = None,
                 A: Union[Iterable[int], Tensor] = None,
                 Q: Union[Iterable[int], Tensor] = None,
                 c: Union[Iterable[int], Tensor] = None,
                 trainable_A: bool = True,
                 trainable_Q: bool = True,
                 device: str = None,
                 loss1 = None,
                 loss2 = None,
                 loss3 = None):
        super().__init__()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        infer_rank = None
        if isinstance(A, Tensor):
            assert torch.all(A >= 0.), "Tensor A should be non-negative."
            
            col_sum = A.sum(dim=0)
            assert torch.allclose(
                col_sum,
                torch.ones_like(col_sum),
                atol=1e-6
            ), "Each column of A must sum to 1."

            self.register_parameter('A', Parameter(
                torch.empty(*A.size()).to(self.device), requires_grad=trainable_A))
            self.A.data.copy_(A)
            infer_rank = self.A.shape[1]
        elif isinstance(A, Iterable):
            self.register_parameter('A', Parameter(torch.randn(*A).abs().to(self.device), requires_grad=trainable_A))
            infer_rank = A[1]
        else:
            self.register_parameter('A', None)

        if isinstance(Q, Tensor):
            assert torch.all(Q >= 0.), "Tensor Q should be non-negative."
            col_sum = Q.sum(dim=0)
            assert torch.allclose(
                col_sum,
                torch.ones_like(col_sum),
                atol=1e-6
            ), "Each column of Q must sum to 1."
            Q_shape = Q.shape
            self.register_parameter('Q', Parameter(
                torch.empty(*Q_shape).to(self.device), requires_grad=trainable_Q))
            self.Q.data.copy_(Q)
            infer_rank = self.Q.shape[0]
        elif isinstance(Q, Iterable):
            self.register_parameter('Q', Parameter(torch.randn(*Q).abs().to(self.device), requires_grad=trainable_Q))
            infer_rank = Q[0]
        else:
            self.register_parameter('Q', None)

        self.register_parameter(
            'c_raw',
            Parameter(torch.tensor(float(c), device=self.device))
        )
        if infer_rank is None:
            assert rank, "A rank should be given when A and Q are not available!"
        else:
            if getattr(self, "Q") is not None:
                assert self.Q.shape[0] == infer_rank, "Latent size of Q does not match with others!"
            if getattr(self, "A") is not None:
                assert self.A.shape[1] == infer_rank, "Latent size of A does not match with others!"
            rank = infer_rank

        self.rank = rank
        

    def forward(self, A: Tensor = None, Q: Tensor = None) -> Tensor:
        r"""An outer wrapper of :meth:`self.reconstruct(A,Q) <torchnmf.nmf.BaseComponent.reconstruct>`.

        .. note::
                Should call the :class:`BaseComponent` instance afterwards
                instead of this since the former takes care of running the
                registered hooks while the latter silently ignores them.

        Args:
            A(Tensor, optional): input activation tensor A. If no tensor was given will use :attr:`A` from this module
                                instead
            Q(Tensor, optional): input template tensor Q. If no tensor was given will use :attr:`Q` from this module
                                instead

        Returns:
            Tensor: tensor
        """
        if A is None:
            A = self.H
        if Q is None:
            Q = self.Q
        assert A is not None
        assert Q is not None
        return self.reconstruct(A, Q)

    @staticmethod
    def reconstruct(A: Tensor, Q: Tensor) -> Tensor:
        return A@Q

    # def _sp_recon_beta_pos_neg(self, P, A, Q, beta):
    #     raise NotImplementedError

    @torch.jit.ignore
    def fit(self,
            P: Tensor,
            max_iter: int = 3000,
            verbose: bool = False,
            l1_ratio: float = 0.5,
            l2_ratio: float = 0.5,
            alpha: float = 0.05,
            num_pairs = 50000 
            ) -> int:
        r"""Learn a MMF model for the data P by minimizing fisher-rao distance.

        To invoke this function, attributes :meth:`A <torchnmf.nmf.BaseComponent.A>` and
        :meth:`Q <torchnmf.nmf.BaseComponent.Q>` should be presented in this module.

        Args:
            P (Tensor): data tensor to be decomposed. Can be a sparse tensor returned by :func:`torch.sparse_coo_tensor` 
            beta (float): beta divergence to be minimized, measuring the distance between V and the NMF model.
                        Default: ``1.``
            tol (float): tolerance of the stopping condition. Default: ``1e-4``
            max_iter (int): maximum number of iterations before timing out. Default: ``200``
            verbose (bool): whether to be verbose. Default: ``False``
            alpha (float): constant that multiplies the regularization terms. Set it to zero to have no regularization
                            Default: ``0``
            l1_ratio (float):  the regularization mixing parameter, with 0 <= l1_ratio <= 1.
                                For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm).
                                For l1_ratio = 1 it is an elementwise L1 penalty.
                                For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2. Default: ``0``

        Returns:
            int: total number of iterations
        """

        assert torch.all((P._values() if P.is_sparse else P) >=
                         0.), "Target should be non-negative."
        col_sum = P.sum(dim=0)
        assert torch.allclose(
            col_sum,
            torch.ones_like(col_sum),
            atol=1e-6
            ), "Each column of Q must sum to 1."

        A = self.A
        Q = self.Q
        c_raw = self.c_raw

        P = P.to(self.device)
        
        loss1_list = []
        loss2_list = []
        
        optimizer = torch.optim.Adam([A, Q, c_raw], lr=1e-3)

        # dis_P = fisher_rao_dis_matrix(P)
        # dis_P = fisher_rao_dis_matrix_block(P, block=1024)
        with tqdm(total=max_iter, disable=verbose) as pbar:
            scaler = torch.cuda.amp.GradScaler()
            for n_iter in range(max_iter):
                optimizer.zero_grad()

                # new code for reduce memory
                with torch.cuda.amp.autocast():
                    c = F.softplus(c_raw)
                    A_recon = normalize(A, axis=0)
                    Q_recon = normalize(Q, axis=0)

                    recon = self.reconstruct(A_recon,Q_recon)
                    loss1 = fisher_rao_dis(recon, P)

                    # dis_P = fisher_rao_dis_matrix(P)
                    # dis_Q = fisher_rao_dis_matrix(Q_recon)
                    # dis_Q = fisher_rao_dis_matrix_block(Q_recon, block=1024)

                    # num_pairs = 50000   # 可调，1e4~1e5
                    N = Q_recon.shape[1]
                    max_pairs = N * N

                    if num_pairs >= max_pairs:
                        idx_i, idx_j = torch.meshgrid(
                            torch.arange(N, device=Q_recon.device),
                            torch.arange(N, device=Q_recon.device),
                            indexing="ij"
                        )
                        idx_i = idx_i.reshape(-1)
                        idx_j = idx_j.reshape(-1)
                    else:

                        idx_i = torch.randint(0, N, (num_pairs,), device=Q_recon.device)
                        idx_j = torch.randint(0, N, (num_pairs,), device=Q_recon.device)

                    Qi = Q_recon[:, idx_i]
                    Qj = Q_recon[:, idx_j]
                    Pi = P[:, idx_i]
                    Pj = P[:, idx_j]

                    dis_Q = fisher_rao_dis(Qi, Qj)
                    dis_P = fisher_rao_dis(Pi, Pj)



                    # loss2 = cosine_loss(c*dis_Q, dis_P)
                    # loss2 = angle_mse_loss(c*dis_Q, dis_P)
                    loss2 = angle_mse_loss(c*dis_Q, dis_P)
                    
                    # loss2 = elastic_net_loss(dis_Q, dis_P, alpha=0.1)       

                    loss = l1_ratio*loss1 + l2_ratio*loss2

                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                store_every = 10
                if n_iter % store_every == 0:
                    loss1_list.append(loss1.item())
                    loss2_list.append(loss2.item())
                pbar.update(1)
               
        self.loss1 = loss1_list
        self.loss2 = loss2_list
        with torch.no_grad():
            self.A.copy_(normalize(self.A, axis=0))
            self.Q.copy_(normalize(self.Q, axis=0)) 
        return n_iter + 1

    @torch.jit.ignore
    def sparse_fit(self,
        P: Tensor,
        max_iter=10000,
        verbose=False,
        l1_ratio: float = 0.5,
        l2_ratio: float = 0.5,
        l3_ratio: float = 0.5,
        ) -> int:
        r"""Learn a MMF model for the data P by minimizing fisher-rao distance with orthognal constraints(P=A@Q each column of A is orthognal)

        Args:
            P (Tensor): data tensor to be decomposed. Can be a sparse tensor returned by :func:`torch.sparse_coo_tensor` 
            max_iter (int): maximum number of iterations before timing out. Default: ``10000``
            verbose (bool): whether to be verbose. Default: ``False``
            alpha: float = 1,
            l1_ratio: float = 0.5
            l2_ratio: float = 0.5

        Returns:
            int: total number of iterations
        """

        assert torch.all((P._values() if P.is_sparse else P) >=
                         0.), "Target should be non-negative."
        col_sum = P.sum(dim=0)
        assert torch.allclose(
            col_sum,
            torch.ones_like(col_sum),
            atol=1e-6
            ), "Each column of Q must sum to 1."

        A = self.A
        Q = self.Q

        P = P.to(self.device)
        
        loss1_list = []
        loss2_list = []
        loss3_list = []
        
        optimizer = torch.optim.Adam([A, Q], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-5)

        with tqdm(total=max_iter, disable=verbose) as pbar:
            for n_iter in range(max_iter):
                optimizer.zero_grad()

                A_recon = normalize(A, axis=0)
                Q_recon = normalize(Q, axis=0)

                recon = self.reconstruct(A_recon,Q_recon)
                loss1 = fisher_rao_dis(recon, P)

                dis_P = fisher_rao_dis_matrix(P)
                dis_Q = fisher_rao_dis_matrix(Q_recon)
                # loss2 = cosine_loss(dis_Q, dis_P)
                loss2 = angle_mae_loss(dis_Q, dis_P)

                loss3 = orthogonal_loss(A_recon)

                loss = l1_ratio*loss1 + l2_ratio*loss2 + l3_ratio*loss3
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                store_every = 10
                if n_iter % store_every == 0:
                    loss1_list.append(loss1.item())
                    loss2_list.append(loss2.item())
                    loss3_list.append(loss3.item())
                pbar.update(1)
               
        self.loss1 = loss1_list
        self.loss2 = loss2_list
        self.loss3 = loss3_list
        with torch.no_grad():
            self.A.copy_(normalize(self.A, axis=0))
            self.Q.copy_(normalize(self.Q, axis=0)) 
        return n_iter + 1
