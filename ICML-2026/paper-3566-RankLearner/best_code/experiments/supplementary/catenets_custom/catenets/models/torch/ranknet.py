from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_R,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.torch.base import (
    DEVICE,
    BaseCATEEstimator,
    BasicNet,
    PropensityNet,
    RepresentationNet,
)
from catenets.models.torch.utils.model_utils import make_val_split


class RankLearner(BaseCATEEstimator):
    """Two-stage rank learner with orthogonal pairwise training.

    Stage 1 fits nuisance functions (propensity and potential outcomes) on a nuisance fold.
    Stage 2 freezes nuisances, computes orthogonal pairwise labels on a separate ranking fold,
    and trains a ranking network using pairwise BCE on score differences.
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_layers_out_nuis: int = DEFAULT_LAYERS_OUT,
        n_layers_out_rank: int = DEFAULT_LAYERS_OUT,
        n_units_out_nuis: int = DEFAULT_UNITS_OUT,
        n_units_out_rank: int = DEFAULT_UNITS_OUT,
        n_layers_r_nuis: int = DEFAULT_LAYERS_R,
        n_layers_r_rank: int = DEFAULT_LAYERS_R,
        n_units_r_nuis: int = DEFAULT_UNITS_R,
        n_units_r_rank: int = DEFAULT_UNITS_R,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        nuisance_split_prop: float = 0.5,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        early_stopping: bool = True,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        patience: int = DEFAULT_PATIENCE,
        batch_norm: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
        pair_batch_size: int = 2048,
        kappa: float = 1.0,
        orthogonal_weight: float = 1.0,
        clip_labels: bool = True,
        nuisance_epochs: Optional[int] = None,
        ranker_epochs: Optional[int] = None,
    ) -> None:
        super(RankLearner, self).__init__()
        self.binary_y = binary_y
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_iter = n_iter
        self.val_split_prop = val_split_prop
        self.nuisance_split_prop = nuisance_split_prop
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.early_stopping = early_stopping
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.pair_batch_size = pair_batch_size
        self.kappa = kappa
        self.orthogonal_weight = orthogonal_weight
        self.clip_labels = clip_labels
        self.nuisance_epochs = nuisance_epochs if nuisance_epochs is not None else n_iter
        self.ranker_epochs = ranker_epochs if ranker_epochs is not None else n_iter

        # nuisance components (stage 1)
        self._nuisance_repr = RepresentationNet(
            n_unit_in,
            n_units=n_units_r_nuis,
            n_layers=n_layers_r_nuis,
            nonlin=nonlin,
            batch_norm=batch_norm,
        )
        self._po_estimators = nn.ModuleList([
            BasicNet(
                "ranknet_po_0",
                n_units_r_nuis,
                binary_y=binary_y,
                n_layers_out=n_layers_out_nuis,
                n_units_out=n_units_out_nuis,
                nonlin=nonlin,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_prob=dropout_prob,
            ),
            BasicNet(
                "ranknet_po_1",
                n_units_r_nuis,
                binary_y=binary_y,
                n_layers_out=n_layers_out_nuis,
                n_units_out=n_units_out_nuis,
                nonlin=nonlin,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_prob=dropout_prob,
            ),
        ])
        
        self._propensity_estimator = PropensityNet(
            "ranknet_propensity",
            n_units_r_nuis,
            n_unit_out=2,
            weighting_strategy="prop",
            n_units_out_prop=n_units_out_nuis,
            n_layers_out_prop=n_layers_out_nuis,
            nonlin=nonlin,
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_prob=dropout_prob,
        )

        # ranking components (stage 2)
        self._rank_repr = RepresentationNet(
            n_unit_in,
            n_units=n_units_r_rank,
            n_layers=n_layers_r_rank,
            nonlin=nonlin,
            batch_norm=batch_norm,
        )
        self._rank_estimator = BasicNet(
            "ranknet_score",
            n_units_r_rank,
            binary_y=False,
            n_layers_out=n_layers_out_rank,
            n_units_out=n_units_out_rank,
            nonlin=nonlin,
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_prob=dropout_prob,
        )

        self.nuisance_fitted_ = False

    def _set_seeds(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _sample_pairs(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = torch.randint(0, n, (self.pair_batch_size,), device=DEVICE)
        j = torch.randint(0, n - 1, (self.pair_batch_size,), device=DEVICE)
        j = j + (j >= i).long()
        return i, j

    def _compute_dr(
        self, y: torch.Tensor, w: torch.Tensor, m0: torch.Tensor, m1: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        p = torch.clamp(p, 1e-3, 1 - 1e-3)
        treated = w * (y - m1) / p
        control = (1 - w) * (y - m0) / (1 - p)
        return treated - control + (m1 - m0)

    def _orthogonal_pair_labels(
        self, tau: torch.Tensor, dr: torch.Tensor, i: torch.Tensor, j: torch.Tensor
    ) -> torch.Tensor:
        plug_in = torch.sigmoid((tau[i] - tau[j]) / self.kappa)
        delta = (dr[i] - tau[i]) - (dr[j] - tau[j])
        correction = (1.0 / self.kappa) * plug_in * (1 - plug_in)
        labels = plug_in + self.orthogonal_weight * correction * delta
        if self.clip_labels:
            labels = torch.clamp(labels, 0.0, 1.0)
        return labels.detach()

    def _compute_autoc(self, scores: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
        scores_np = scores.detach().cpu().numpy().astype(np.float64)
        dr_np = dr.detach().cpu().numpy().astype(np.float64)
        n = len(dr_np)
        if n < 2:
            return torch.tensor(0.0, device=DEVICE)

        q_grid = np.linspace(0.05, 1.0, 21)
        centered_dr = dr_np - dr_np.mean()
        toc_vals = []
        eps = 1e-12

        for q in q_grid:
            target = int(np.round(q * n))
            thr = np.quantile(scores_np, 1.0 - q)
            above = scores_np > thr
            equal = scores_np == thr
            n_above = int(above.sum())
            n_equal = int(equal.sum())
            need = max(target - n_above, 0)
            frac = 0.0 if n_equal == 0 else min(need / n_equal, 1.0)
            iq = above.astype(np.float64) + frac * equal.astype(np.float64)
            pi = max(iq.mean(), eps)
            centered_iq = (iq / pi) - (iq / pi).mean()
            toc_vals.append(np.mean(centered_dr * centered_iq))

        autoc = np.sum(np.asarray(toc_vals[:-1]) * np.diff(q_grid))
        return torch.tensor(float(autoc), device=DEVICE)

    def _nuisance_forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rep = self._nuisance_repr(X)
        m0 = self._po_estimators[0](rep).squeeze()
        m1 = self._po_estimators[1](rep).squeeze()
        p = self._propensity_estimator(rep)[:, 1].squeeze()
        return m0, m1, p

    def fit_nuisance(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> Tuple["RankLearner", dict]:
        self._set_seeds()
        self.train()

        X = self._check_tensor(X)
        y = self._check_tensor(y).squeeze()
        w = self._check_tensor(w).squeeze()

        X, y, w, X_val, y_val, w_val, val_string = make_val_split(
            X, y, w=w, val_split_prop=self.val_split_prop, seed=self.seed
        )

        n = X.shape[0]
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.ceil(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        params = (
            list(self._nuisance_repr.parameters())
            + list(self._po_estimators[0].parameters())
            + list(self._po_estimators[1].parameters())
            + list(self._propensity_estimator.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        outcome_loss = nn.BCELoss() if self.binary_y else nn.MSELoss()
        prop_loss = nn.NLLLoss()

        best_val = LARGE_VAL
        best_state = None
        patience = 0

        for i in range(self.nuisance_epochs):
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                optimizer.zero_grad()
                idx_next = train_indices[(b * batch_size) : min((b + 1) * batch_size, n)]

                X_next = X[idx_next]
                y_next = y[idx_next]
                w_next = w[idx_next]

                m0, m1, p = self._nuisance_forward(X_next)
                y_hat = (1 - w_next) * m0 + w_next * m1

                loss = outcome_loss(y_hat, y_next) + prop_loss(
                    torch.log(torch.stack([1 - p, p], dim=1) + 1e-8), w_next.long()
                )
                loss.backward()
                optimizer.step()
                train_loss.append(loss.detach())

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    m0_val, m1_val, p_val = self._nuisance_forward(X_val)
                    y_val_hat = (1 - w_val) * m0_val + w_val * m1_val
                    val_loss = outcome_loss(y_val_hat, y_val) + prop_loss(
                        torch.log(torch.stack([1 - p_val, p_val], dim=1) + 1e-8),
                        w_val.long(),
                    )

                if self.early_stopping:
                    if val_loss < best_val:
                        best_val = float(val_loss.detach().cpu())
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.state_dict().items()
                        }
                        patience = 0
                    else:
                        patience += 1
                    if patience > self.patience and i > self.n_iter_min:
                        break

                if i % self.n_iter_print == 0:
                    log.info(
                        f"[RankLearner:nuisance] Epoch: {i}, current {val_string} loss: {val_loss}, train_loss: {torch.mean(torch.stack(train_loss))}"
                    )
        
        if best_state is not None:
            self.load_state_dict(best_state)
        self.nuisance_fitted_ = True
        return self, {"best_val_loss": best_val}

    @torch.no_grad()
    def _compute_orthogonal_targets(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m0, m1, p = self._nuisance_forward(X)
        tau = m1 - m0
        dr = self._compute_dr(y, w, m0, m1, p)
        return tau.detach(), dr.detach()

    @torch.no_grad()
    def prepare_ranker_targets(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self._check_tensor(X)
        y = self._check_tensor(y).squeeze()
        w = self._check_tensor(w).squeeze()
        return self._compute_orthogonal_targets(X, y, w)

    def fit_ranker(
        self,
        X: torch.Tensor,
        tau: torch.Tensor,
        dr: torch.Tensor,
    ) -> Tuple["RankLearner", dict]:
        self._set_seeds()
        self.train()

        X = self._check_tensor(X)
        tau = self._check_tensor(tau).squeeze()
        dr = self._check_tensor(dr).squeeze()

        n_all = X.shape[0]
        if self.val_split_prop > 0:
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(n_all)
            n_val = max(1, int(self.val_split_prop * n_all))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            X_val = X[val_idx]
            tau_val = tau[val_idx]
            dr_val = dr[val_idx]
            X = X[train_idx]
            tau = tau[train_idx]
            dr = dr[train_idx]
            val_string = "validation"
        else:
            X_val, tau_val, dr_val = X, tau, dr
            val_string = "training"

        n = X.shape[0]
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.ceil(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        params = list(self._rank_repr.parameters()) + list(self._rank_estimator.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        rank_loss = nn.BCEWithLogitsLoss()

        best_autoc = -LARGE_VAL
        best_state = None
        patience = 0

        for i in range(self.ranker_epochs):
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                optimizer.zero_grad()
                idx_next = train_indices[(b * batch_size) : min((b + 1) * batch_size, n)]

                X_next = X[idx_next]
                tau_next = tau[idx_next]
                dr_next = dr[idx_next]

                rep = self._rank_repr(X_next)
                score = self._rank_estimator(rep).squeeze()

                pair_i, pair_j = self._sample_pairs(len(idx_next))
                labels = self._orthogonal_pair_labels(tau_next, dr_next, pair_i, pair_j)
                logits = score[pair_i] - score[pair_j]
                loss = rank_loss(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.detach())

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    val_rep = self._rank_repr(X_val)
                    val_scores = self._rank_estimator(val_rep).squeeze()
                    val_autoc = self._compute_autoc(val_scores, dr_val)

                if self.early_stopping:
                    if val_autoc > best_autoc:
                        best_autoc = float(val_autoc.detach().cpu())
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.state_dict().items()
                        }
                        patience = 0
                    else:
                        patience += 1
                
                    if patience > self.patience and i > self.n_iter_min:
                        break

                if i % self.n_iter_print == 0:
                    log.info(
                        f"[RankLearner:rank] Epoch: {i}, current {val_string} AUTOC: {val_autoc}, train_loss: {torch.mean(torch.stack(train_loss))}"
                    )
        if best_state is not None:
            self.load_state_dict(best_state)
        
        return self, {"best_val_autoc": best_autoc}

    def fit(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> Tuple["RankLearner", dict]:
        self._set_seeds()

        X = self._check_tensor(X)
        y = self._check_tensor(y).squeeze()
        w = self._check_tensor(w).squeeze()

        # two-stage split: nuisance fold + ranking fold
        X_nuis, y_nuis, w_nuis, X_rank, y_rank, w_rank, _ = make_val_split(
            X, y, w=w, val_split_prop=1 - self.nuisance_split_prop, seed=self.seed
        )

        _, nuisance_info = self.fit_nuisance(X_nuis, y_nuis, w_nuis)

        with torch.no_grad():
            tau_rank, dr_rank = self._compute_orthogonal_targets(X_rank, y_rank, w_rank)

        _, ranker_info = self.fit_ranker(X_rank, tau_rank, dr_rank)
        return self, {
            "nuisance": nuisance_info,
            "ranker": ranker_info}

    def predict(
        self, X: torch.Tensor, return_po: bool = False, training: bool = False
    ) -> torch.Tensor:
        if not training:
            self.eval()

        X = self._check_tensor(X)

        with torch.no_grad():
            rank_rep = self._rank_repr(X)
            score = self._rank_estimator(rank_rep).squeeze()

            if return_po:
                m0, m1, _ = self._nuisance_forward(X)
                return score, m0, m1

        return score
