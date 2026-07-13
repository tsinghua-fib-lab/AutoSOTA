# ECSEL: Explainable Classification via Signomial Equation Learning
#
# Copyright (c) 2026 Adia Lumadjeng
# University of Amsterdam
#
# Authors: Adia Lumadjeng, Ilker Birbil, Erman Acar

"""
ECSEL: Explainable Classification via Signomial Equation Learning.

This module implements an interpretable classifier whose decision logits are
signomial functions of the input features, i.e. sums of signed monomials of the
form ``const * prod_j(x_j ** alpha_j)``. Training is performed by mini-batch
Adam on a cross-entropy objective with an L1 penalty that encourages sparse
exponents and constants, yielding a closed-form, human-readable equation per
class.

The module provides two components:

- :func:`adam_minibatch`: a standalone mini-batch Adam optimizer operating on a
  flat parameter vector with optional early stopping and gradient-norm clipping.
- :class:`SignomialClassifier`: a scikit-learn-style estimator that wraps the
  optimizer, handles preprocessing, multi-restart initialization, and formula
  extraction.

Notes
-----
Features are assumed to be positive (signomials are defined on the positive
orthant). Inputs are floored at ``1e-10`` internally so that powers and
logarithms remain finite; genuinely non-positive features should be rescaled to
a positive range before fitting (see the ``internal_scaling_range`` argument).
"""

import time
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def adam_minibatch(
    loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    theta0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    num_epochs: int = 10,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
    patience: int = 10,
    min_delta: float = 1e-6,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Minimize a loss over a flat parameter vector using mini-batch Adam.

    Runs Adam for up to ``num_epochs`` epochs, shuffling the data each epoch and
    updating ``theta`` once per mini-batch. Gradients are clipped to a maximum
    L2 norm of 1.0 before the update to prevent exploding gradients. Early
    stopping monitors the validation loss when ``X_val``/``y_val`` are provided,
    otherwise the training loss; the parameters achieving the best monitored
    loss are returned.

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(theta, X_batch, y_batch) -> float``. Returns the scalar loss.
    grad_fn : callable
        ``grad_fn(theta, X_batch, y_batch) -> np.ndarray``. Returns the gradient
        of the loss with respect to ``theta``, shaped like ``theta``.
    theta0 : np.ndarray
        Initial parameter vector. Copied internally; not modified in place.
    X, y : np.ndarray
        Training features and targets.
    X_val, y_val : np.ndarray, optional
        Validation features and targets used only for monitoring and early
        stopping. If either is ``None``, the training loss is monitored instead.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=1e-3
        Learning rate (Adam step size).
    betas : tuple of float, default=(0.9, 0.999)
        Adam first- and second-moment decay rates ``(beta1, beta2)``.
    eps : float, default=1e-8
        Numerical stabilizer added to the denominator of the Adam update.
    weight_decay : float, default=0.0
        Coefficient of decoupled L2 weight decay added to the gradient. Set to
        0.0 to disable.
    num_epochs : int, default=10
        Maximum number of epochs.
    shuffle : bool, default=True
        Whether to shuffle the training indices at the start of each epoch.
    rng : np.random.Generator, optional
        Random generator used for shuffling. A fresh default generator is
        created if ``None``.
    patience : int, default=10
        Number of consecutive epochs without improvement (greater than
        ``min_delta``) tolerated before early stopping.
    min_delta : float, default=1e-6
        Minimum decrease in the monitored loss that counts as an improvement.
    verbose : bool, default=True
        Whether to print per-epoch progress.

    Returns
    -------
    best_theta : np.ndarray
        Parameter vector achieving the lowest monitored loss.
    info : dict
        Diagnostics with keys ``train_losses``, ``val_losses``, ``epochs``,
        ``steps``, ``best_loss``, ``converged`` (whether early stopping
        triggered), and ``used_validation``.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = np.array(theta0, dtype=float).copy()
    m = np.zeros_like(theta)  # first moment estimate
    v = np.zeros_like(theta)  # second moment estimate

    n = X.shape[0]
    beta1, beta2 = betas
    t = 0  # global step counter, used for bias correction
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_theta = theta.copy()
    patience_counter = 0

    use_validation = X_val is not None and y_val is not None

    if verbose:
        if use_validation:
            print(f"Starting Adam: {n} train samples, {X_val.shape[0]} val samples, batch={batch_size}, lr={lr}")
        else:
            print(f"Starting Adam: {n} train samples, batch={batch_size}, lr={lr}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        epoch_train_loss = 0.0
        num_batches = 0

        # Training loop -- only uses training data.
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            Xb = X[batch_idx]
            yb = y[batch_idx]

            # Compute gradient on the training batch.
            g = grad_fn(theta, Xb, yb)

            # Clip by L2 norm to prevent exploding gradients. Rescaling
            # preserves the gradient direction and only caps its magnitude.
            max_grad_norm = 1.0
            grad_norm = np.linalg.norm(g)
            if grad_norm > max_grad_norm:
                g = g * (max_grad_norm / grad_norm)

            if weight_decay != 0.0:
                g = g + weight_decay * theta

            if not np.all(np.isfinite(g)):
                if verbose:
                    print(f"Warning: Invalid gradients at step {t}")
                continue

            # Adam update with bias-corrected moment estimates.
            t += 1
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

            # Track training loss (sample-weighted).
            batch_loss = loss_fn(theta, Xb, yb)
            if np.isfinite(batch_loss):
                epoch_train_loss += batch_loss * len(batch_idx)
                num_batches += 1

        # Compute average epoch training loss.
        if num_batches > 0:
            avg_train_loss = epoch_train_loss / n
        else:
            avg_train_loss = float('inf')

        train_losses.append(avg_train_loss)

        # Compute validation loss (monitoring only, not used for updates).
        if use_validation:
            val_loss = loss_fn(theta, X_val, y_val)
            val_losses.append(val_loss)
            monitoring_loss = val_loss
            loss_type = "Val"
        else:
            monitoring_loss = avg_train_loss
            loss_type = "Train"

        epoch_time = time.time() - epoch_start

        if verbose:
            if num_epochs <= 5 or epoch == 0 or (epoch + 1) % (num_epochs // 5) == 0 or epoch == num_epochs - 1:
                if use_validation:
                    print(f"Epoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.6f}, Val={val_loss:.6f}, Time={epoch_time:.2f}s")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.6f}, Time={epoch_time:.2f}s")

        # Early stopping based on the monitored loss; keep the best parameters.
        if monitoring_loss < best_loss - min_delta:
            best_loss = monitoring_loss
            best_theta = theta.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} ({loss_type} loss stopped improving)")
            break

    info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs": epoch + 1,
        "steps": t,
        "best_loss": best_loss,
        "converged": patience_counter >= patience,
        "used_validation": use_validation,
    }

    return best_theta, info


class SignomialClassifier:
    """Interpretable classifier with signomial decision logits (ECSEL).

    Each class logit is modeled as a signomial function of the features:

        z_c(x) = sum_{k=1..K} const_{c,k} * prod_j (x_j ** alpha_{c,k,j})

    For binary problems a single logit feeds a sigmoid; for multi-class problems
    one logit per class feeds a softmax. Parameters (constants and exponents) are
    fit by mini-batch Adam on an L1-regularized cross-entropy loss, which drives
    many exponents and constants toward zero and yields a compact, closed-form
    equation that can be read off with :meth:`get_learned_formula`.

    Parameters
    ----------
    K : int, default=2
        Number of signomial terms (monomials) per class logit.
    l1_strength : float, default=0.1
        Coefficient of the L1 penalty applied to all parameters.
    batch_size : int, default=256
        Mini-batch size used during Adam optimization.
    lr : float, default=1e-3
        Learning rate.
    num_epochs : int, default=50
        Maximum number of training epochs per restart.
    n_restarts : int, default=1
        Number of random restarts; the restart with the lowest loss is kept.
    patience : int, default=10
        Early-stopping patience (epochs without improvement).
    min_delta : float, default=1e-6
        Minimum loss improvement counted as progress for early stopping.
    gradient_method : {'analytical', 'finite_diff'}, default='analytical'
        Whether to use the closed-form gradient or finite differences. The
        analytical gradient is far faster; finite differences are mainly useful
        for verifying the analytical implementation.
    use_sigmoid : bool or None, default=None
        Force sigmoid (``True``) or softmax (``False``) output. If ``None``,
        sigmoid is selected automatically for binary problems and softmax for
        multi-class problems.
    sigmoid_threshold : float, default=0.5
        Probability threshold applied to the positive class in the binary
        (sigmoid) case.
    random_state : int or None, default=None
        Base seed. Restart ``i`` uses seed ``random_state + i`` for reproducible
        initialization; if ``None``, initialization is non-deterministic.
    verbose : bool, default=False
        Whether to print progress during fitting.
    internal_scaling_range : tuple of float or None, default=None
        If a ``(low, high)`` tuple is given, a :class:`MinMaxScaler` is fit on
        the training data (only) and applied to all inputs, mapping features
        into a positive range suitable for signomials. If ``None``, no internal
        scaling is performed and inputs are assumed to be pre-scaled by the
        caller.

    Attributes
    ----------
    scaler_ : MinMaxScaler or None
        The fitted internal scaler, or ``None`` when ``internal_scaling_range``
        is ``None``.
    label_encoder_ : LabelEncoder
        Encoder mapping raw labels to integer class indices.
    best_params_ : np.ndarray
        Flat parameter vector of the best restart.
    classes_ : np.ndarray
        Sorted array of class labels.
    n_classes_ : int
        Number of classes.
    n_features_ : int
        Number of input features.
    training_info_ : dict
        Diagnostics returned by :func:`adam_minibatch` for the best restart.

    Notes
    -----
    Preprocessing (scaler and label encoder) is fit exclusively on the training
    data passed to :meth:`fit` and merely applied at prediction time, so there
    is no information leakage from held-out data through this estimator.
    """

    def __init__(self,
                 K=2,
                 l1_strength=0.1,
                 batch_size=256,
                 lr=1e-3,
                 num_epochs=50,
                 n_restarts=1,
                 patience=10,
                 min_delta=1e-6,
                 gradient_method='analytical',  # or 'finite_diff'
                 use_sigmoid=None,  # None=auto (sigmoid for binary, softmax for multi), True=force sigmoid, False=force softmax
                 sigmoid_threshold=0.5,
                 random_state=None,
                 verbose=False,
                 internal_scaling_range=None):
        self.K = K
        self.l1_strength = l1_strength
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_restarts = n_restarts
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_method = gradient_method
        self.use_sigmoid = use_sigmoid
        self.sigmoid_threshold = sigmoid_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.internal_scaling_range = internal_scaling_range

        if gradient_method not in ['analytical', 'finite_diff']:
            raise ValueError("gradient_method must be 'analytical' or 'finite_diff'")

        # Model state (populated by fit).
        self.scaler_ = None
        self.label_encoder_ = None
        self.best_params_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.training_info_ = None
        self._is_fitted = False
        self._using_sigmoid = False  # Set during fit.

    def _sigmoid(self, z):
        """Numerically stable logistic sigmoid, with inputs clipped to +/-500."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _softmax(self, Z):
        """Numerically stable row-wise softmax over a 2D logit array."""
        Z = np.clip(Z, -500, 500)
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _compute_predictions(self, params, X):
        """Compute signomial logits from a flat parameter vector.

        Evaluates ``z_c = sum_k const_{c,k} * prod_j(x_j ** alpha_{c,k,j})`` for
        each logit. The flat ``params`` vector is laid out per class, then per
        term, as ``[const, alpha_0, ..., alpha_{m-1}]`` blocks of length
        ``m + 1``. Features are floored at ``1e-10`` so powers stay finite, and
        non-finite term values are sanitized to large finite numbers.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_logits)
            Logits, with one column for the sigmoid case and ``n_classes_``
            columns for the softmax case.
        """
        n_samples, m = X.shape
        params_per_term = m + 1

        # One logit for binary (sigmoid); one per class for multi-class (softmax).
        n_logits = 1 if self._using_sigmoid else self.n_classes_
        params_per_class = self.K * params_per_term

        Z = np.zeros((n_samples, n_logits))

        for c in range(n_logits):
            class_start = c * params_per_class
            for k in range(self.K):
                start_idx = class_start + k * params_per_term
                if start_idx + params_per_term > len(params):
                    continue

                const = params[start_idx]
                exponents = params[start_idx + 1:start_idx + params_per_term]

                X_safe = np.maximum(X, 1e-10)
                term_value = const * np.prod(np.power(X_safe, exponents), axis=1)

                if np.any(~np.isfinite(term_value)):
                    term_value = np.nan_to_num(term_value, nan=0.0, posinf=1e10, neginf=-1e10)

                Z[:, c] += term_value

        return Z

    def _loss_function(self, params, X_batch, y_batch):
        """L1-regularized cross-entropy loss for the given parameters.

        Uses binary cross-entropy with a sigmoid in the binary case and
        categorical cross-entropy with a softmax otherwise, plus an L1 penalty
        ``l1_strength * sum(|params|)``. Returns a large finite value (``1e10``)
        if evaluation fails, so the optimizer can continue.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.
        X_batch, y_batch : np.ndarray
            Mini-batch features and (encoded) targets.

        Returns
        -------
        float
            Scalar loss (cross-entropy plus L1 penalty).
        """
        try:
            Z = self._compute_predictions(params, X_batch)

            if self._using_sigmoid:
                # Binary classification with sigmoid.
                P_pos = self._sigmoid(Z.ravel())
                P_pos = np.clip(P_pos, 1e-15, 1 - 1e-15)

                ce_loss = -np.mean(y_batch * np.log(P_pos) + (1 - y_batch) * np.log(1 - P_pos))
            else:
                # Multi-class classification with softmax.
                P = self._softmax(Z)
                P = np.clip(P, 1e-15, 1 - 1e-15)

                n_samples = len(y_batch)
                y_onehot = np.zeros((n_samples, self.n_classes_))
                y_onehot[np.arange(n_samples), y_batch.astype(int)] = 1

                ce_loss = -np.mean(y_onehot * np.log(P))  # Divides cross-entropy by batch size.

            # L1 penalty on both constants and exponents.
            l1_penalty = self.l1_strength * np.sum(np.abs(params))

            return ce_loss + l1_penalty

        except Exception:
            return 1e10

    def _analytical_gradient(self, params, X_batch, y_batch):
        """Closed-form gradient of the L1-regularized cross-entropy loss.

        Backpropagates the cross-entropy gradient ``dL/dZ`` through each
        signomial term to its constant and exponents. For a term
        ``const * prod_j(x_j ** alpha_j)`` the derivative w.r.t. the constant is
        the monomial value, and w.r.t. exponent ``alpha_j`` it carries an extra
        ``const * log(x_j)`` factor. The L1 subgradient ``sign(.)`` is added per
        parameter.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.
        X_batch, y_batch : np.ndarray
            Mini-batch features and (encoded) targets.

        Returns
        -------
        grad : np.ndarray
            Gradient with the same shape as ``params``.
        """
        n_samples, m = X_batch.shape
        params_per_term = m + 1
        params_per_class = self.K * params_per_term

        Z = self._compute_predictions(params, X_batch)

        if self._using_sigmoid:
            # Binary classification with sigmoid.
            P_pos = self._sigmoid(Z.ravel())

            # Gradient of binary cross-entropy w.r.t. the logit.
            dL_dz = (P_pos - y_batch) / n_samples  # Shape: (n_samples,)
            dL_dZ = dL_dz.reshape(-1, 1)  # Shape: (n_samples, 1)
            n_logits = 1
        else:
            # Multi-class classification with softmax.
            P = self._softmax(Z)

            y_onehot = np.zeros((n_samples, self.n_classes_))
            y_onehot[np.arange(n_samples), y_batch.astype(int)] = 1

            dL_dZ = (P - y_onehot) / n_samples  # Divides by batch size.
            n_logits = self.n_classes_

        grad = np.zeros_like(params)
        X_safe = np.maximum(X_batch, 1e-10)

        for c in range(n_logits):
            class_start = c * params_per_class
            for k in range(self.K):
                start_idx = class_start + k * params_per_term
                if start_idx + params_per_term > len(params):
                    continue

                const = params[start_idx]
                exponents = params[start_idx + 1:start_idx + params_per_term]

                term_values = np.prod(np.power(X_safe, exponents), axis=1)

                # Gradient w.r.t. the constant.
                grad[start_idx] = np.sum(dL_dZ[:, c] * term_values)
                grad[start_idx] += self.l1_strength * np.sign(const)

                # Gradients w.r.t. each exponent.
                for j in range(m):
                    exp_grad = const * np.log(X_safe[:, j]) * term_values
                    grad[start_idx + 1 + j] = np.sum(dL_dZ[:, c] * exp_grad)
                    grad[start_idx + 1 + j] += self.l1_strength * np.sign(exponents[j])

        return grad

    def _finite_diff_gradient(self, params, X_batch, y_batch):
        """Forward finite-difference gradient (step ``h = 1e-8``).

        Reference implementation for validating :meth:`_analytical_gradient`.
        Evaluates one extra loss per parameter, so it is slow and intended only
        for testing.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.
        X_batch, y_batch : np.ndarray
            Mini-batch features and (encoded) targets.

        Returns
        -------
        grad : np.ndarray
            Approximate gradient with the same shape as ``params``.
        """
        grad = np.zeros_like(params)
        h = 1e-8
        base_loss = self._loss_function(params, X_batch, y_batch)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += h
            loss_plus = self._loss_function(params_plus, X_batch, y_batch)
            grad[i] = (loss_plus - base_loss) / h

        return grad

    def _gradient_function(self, params, X_batch, y_batch):
        """Dispatch to the analytical or finite-difference gradient."""
        if self.gradient_method == 'analytical':
            return self._analytical_gradient(params, X_batch, y_batch)
        else:
            return self._finite_diff_gradient(params, X_batch, y_batch)

    def _initialize_parameters(self, rng):
        """Draw an initial parameter vector with sparse random exponents.

        Each term's constant is drawn from ``Normal(1, 5)``. Each exponent is
        set to zero with probability 0.7 and otherwise drawn from
        ``Normal(0, 1)``, producing a sparse starting point.

        Parameters
        ----------
        rng : np.random.Generator
            Random generator for reproducibility.

        Returns
        -------
        params : np.ndarray
            Initialized flat parameter vector.
        """
        m = self.n_features_
        params_per_term = m + 1
        params_per_class = self.K * params_per_term

        # One logit for binary (sigmoid); one per class for multi-class (softmax).
        n_logits = 1 if self._using_sigmoid else self.n_classes_
        n_params = n_logits * params_per_class

        params = np.zeros(n_params)

        for c in range(n_logits):
            class_start = c * params_per_class
            for k in range(self.K):
                term_start = class_start + k * params_per_term

                # Random constant.
                params[term_start] = rng.normal(1, 5)

                # Sparse random exponents (kept with probability 0.3).
                for j in range(m):
                    if rng.random() < 0.3:
                        params[term_start + 1 + j] = rng.normal(0, 1)

        return params

    def fit(self, X, y, validation_split=0.0):
        """Fit the classifier using multi-restart Adam with early stopping.

        Fits the (optional) internal scaler and the label encoder on the
        training data, optionally reserves a stratified internal validation
        split for early-stopping monitoring, then runs ``n_restarts``
        independent Adam optimizations and keeps the parameters with the lowest
        monitored loss. Each restart ``i`` is seeded with ``random_state + i``
        (when ``random_state`` is set) so the full procedure is reproducible.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features. Assumed positive (or mapped to a positive range
            via ``internal_scaling_range``).
        y : array-like of shape (n_samples,)
            Training targets (any hashable labels; encoded internally).
        validation_split : float, default=0.0
            Fraction of the training data held out as an internal, stratified
            validation set used solely for early-stopping monitoring. The
            default of 0.0 trains on all data and early-stops on the training
            loss, reproducing the published experiments. Setting a positive
            value (e.g. 0.2) reserves an internal validation split for early
            stopping instead.

        Returns
        -------
        self : SignomialClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            If sigmoid output is forced on a non-binary problem, or if every
            restart fails.
        """
        start_time = time.time()

        # Base seed for per-restart RNGs. Each restart uses base_seed + restart
        # so initialization is reproducible while still differing across
        # restarts; if random_state is None, each restart is non-deterministic.
        base_seed = self.random_state if self.random_state is not None else None

        X = np.array(X, dtype=float)
        y = np.array(y)

        # Internal feature scaling, fit on the training data only.
        if self.internal_scaling_range is not None:
            self.scaler_ = MinMaxScaler(feature_range=self.internal_scaling_range)
            self.scaler_.fit(X)
            X_scaled = self.scaler_.transform(X)
            if self.verbose:
                print(f"[SignomialClassifier] Internal MinMax scaling applied with range={self.internal_scaling_range}")
        else:
            # No internal scaling: inputs are assumed pre-scaled by the caller.
            self.scaler_ = None
            X_scaled = X
            if self.verbose:
                print("[SignomialClassifier] No internal scaling (using raw/external scaled inputs)")

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Auto mode: sigmoid for binary, softmax for multi-class.
        self._using_sigmoid = self.use_sigmoid if self.use_sigmoid is not None else (self.n_classes_ == 2)

        if self._using_sigmoid and self.n_classes_ != 2:
            raise ValueError("Sigmoid activation can only be used for binary classification")

        y_encoded = self.label_encoder_.transform(y)

        # Reserve a stratified internal validation split for early stopping.
        use_validation = validation_split and validation_split > 0.0
        if use_validation:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_scaled, y_encoded,
                test_size=validation_split,
                stratify=y_encoded,
                random_state=self.random_state,
            )
            if self.verbose:
                print(f"Training on {len(X_tr)} samples, {len(X_val)} held out for early stopping")
        else:
            X_tr, y_tr = X_scaled, y_encoded
            X_val, y_val = None, None
            if self.verbose:
                print(f"Using all {len(X_tr)} samples for training (no internal validation)")

        best_params = None
        best_loss = float('inf')
        best_info = None

        for restart in range(self.n_restarts):
            if self.verbose and self.n_restarts > 1:
                print(f"\nRestart {restart + 1}/{self.n_restarts}")

            # Per-restart RNG: deterministic when a base seed is given.
            if base_seed is not None:
                rng = np.random.default_rng(base_seed + restart)
            else:
                rng = np.random.default_rng()

            initial_params = self._initialize_parameters(rng)

            try:
                # Train with Adam, monitoring the internal validation split (if any).
                params, info = adam_minibatch(
                    loss_fn=self._loss_function,
                    grad_fn=self._gradient_function,
                    theta0=initial_params,
                    X=X_tr,
                    y=y_tr,
                    X_val=X_val,
                    y_val=y_val,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    num_epochs=self.num_epochs,
                    shuffle=True,
                    rng=rng,
                    patience=self.patience,
                    min_delta=self.min_delta,
                    verbose=self.verbose,
                )

                if info['best_loss'] < best_loss:
                    best_loss = info['best_loss']
                    best_params = params.copy()
                    best_info = info.copy()
                    if self.verbose and self.n_restarts > 1:
                        print(f"New best loss: {best_loss:.6f}")

            except Exception as e:
                if self.verbose:
                    print(f"Restart {restart + 1} failed: {e}")
                continue

        if best_params is None:
            raise ValueError("All optimization restarts failed")

        self.best_params_ = best_params
        self.training_info_ = best_info
        self._is_fitted = True

        total_time = time.time() - start_time
        if self.verbose:
            print(f"\nTraining completed in {total_time:.2f} seconds")
            print(f"Best loss: {best_loss:.6f}")
            print(f"Used validation: {use_validation}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Applies the internal scaler when present, evaluates the signomial
        logits, and maps them to probabilities via sigmoid (binary) or softmax
        (multi-class).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes_)
            Class probabilities. In the binary case the columns are
            ``[P(class_0), P(class_1)]``.

        Raises
        ------
        ValueError
            If called before the model is fitted.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X, dtype=float)

        # Apply the internal scaler if one was fit; otherwise assume pre-scaled.
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        Z = self._compute_predictions(self.best_params_, X_scaled)

        if self._using_sigmoid:
            P_pos = self._sigmoid(Z.ravel())
            return np.column_stack([1 - P_pos, P_pos])
        else:
            return self._softmax(Z)

    def predict(self, X):
        """Predict class labels.

        In the binary case the positive class is assigned when its probability
        meets ``sigmoid_threshold``; otherwise the highest-probability class is
        chosen by argmax. Returns labels in the original (un-encoded) space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)

        if self._using_sigmoid:
            # Threshold the positive-class probability.
            y_pred_encoded = (proba[:, 1] >= self.sigmoid_threshold).astype(int)
        else:
            y_pred_encoded = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def score(self, X, y):
        """Return the mean accuracy of :meth:`predict` against ``y``."""
        return np.mean(self.predict(X) == y)

    def get_model_summary(self):
        """Return a summary of the fitted model.

        Counts a parameter as active when its absolute value exceeds 0.01 and
        reports the resulting sparsity, along with architecture and training
        diagnostics.

        Returns
        -------
        dict
            Keys: ``n_classes``, ``n_features``, ``activation``,
            ``n_terms_per_class``, ``total_parameters``, ``active_parameters``,
            ``sparsity``, ``final_loss``, ``converged``, ``epochs_trained``.

        Raises
        ------
        ValueError
            If called before the model is fitted.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting summary")

        # Count active (non-negligible) parameters.
        total_params = len(self.best_params_)
        active_params = np.sum(np.abs(self.best_params_) > 0.01)
        sparsity = 1 - active_params / total_params

        return {
            'n_classes': self.n_classes_,
            'n_features': self.n_features_,
            'activation': 'sigmoid' if self._using_sigmoid else 'softmax',
            'n_terms_per_class': self.K,
            'total_parameters': total_params,
            'active_parameters': active_params,
            'sparsity': sparsity,
            'final_loss': self.training_info_['train_losses'][-1] if self.training_info_['train_losses'] else None,
            'converged': self.training_info_['converged'],
            'epochs_trained': self.training_info_['epochs'],
        }

    def get_learned_formula(self, feature_names=None, threshold=0.01):
        """Return a human-readable string of the learned signomial equation(s).

        Builds one logit expression per class (a single expression in the
        binary case) by listing the surviving terms, where constants and
        exponents below ``threshold`` in absolute value are dropped. Exponents
        close to 1 are shown without an exponent, and unit constants are
        omitted. The output ends with the sigmoid or softmax mapping.

        Parameters
        ----------
        feature_names : list of str, optional
            Names for the features. Defaults to ``X_0, X_1, ...``.
        threshold : float, default=0.01
            Absolute-value cutoff below which constants and exponents are
            treated as zero and hidden from the formula.

        Returns
        -------
        str
            The formatted formula, with the activation mapping appended.

        Raises
        ------
        ValueError
            If called before fitting, or if ``feature_names`` has the wrong
            length.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before extracting formula")

        if feature_names is None:
            feature_names = [f'X_{i}' for i in range(self.n_features_)]
        elif len(feature_names) != self.n_features_:
            raise ValueError(f"feature_names length ({len(feature_names)}) must match n_features ({self.n_features_})")

        m = self.n_features_
        params_per_term = m + 1
        params_per_class = self.K * params_per_term

        class_formulas = []

        if self._using_sigmoid:
            # Binary classification with sigmoid: a single logit expression.
            terms = []

            for k in range(self.K):
                start_idx = k * params_per_term
                if start_idx + params_per_term > len(self.best_params_):
                    continue

                const = self.best_params_[start_idx]
                exponents = self.best_params_[start_idx + 1:start_idx + params_per_term]

                # Skip terms with a negligible constant.
                if abs(const) < threshold:
                    continue

                # Build the variable part of the term.
                term_parts = []
                for j, exp in enumerate(exponents):
                    if abs(exp) > threshold:
                        if abs(exp - 1.0) < threshold:
                            term_parts.append(feature_names[j])
                        else:
                            term_parts.append(f"{feature_names[j]}^{exp:.2f}")

                # Assemble the full term, hiding a unit constant.
                if term_parts:
                    if abs(const - 1.0) < threshold:
                        term_str = " * ".join(term_parts)
                    else:
                        term_str = f"{const:.2f} * " + " * ".join(term_parts)
                    terms.append(term_str)
                elif abs(const) > threshold:  # Constant-only term.
                    terms.append(f"{const:.2f}")

            # Join the terms into a single expression, handling signs.
            if terms:
                if len(terms) == 1:
                    formula = f"z = {terms[0]}"
                else:
                    formula_parts = [terms[0]]
                    for term in terms[1:]:
                        if term.startswith('-'):
                            formula_parts.append(f" - {term[1:]}")
                        else:
                            formula_parts.append(f" + {term}")
                    formula = f"z = " + "".join(formula_parts)
            else:
                formula = f"z = 0"

            class_formulas.append(formula)

            final_formula = f"\nP({self.classes_[1]}) = sigmoid(z) = 1 / (1 + exp(-z))\n"

        else:
            # Multi-class classification with softmax: one expression per class.
            for c, class_name in enumerate(self.classes_):
                class_start = c * params_per_class
                terms = []

                for k in range(self.K):
                    start_idx = class_start + k * params_per_term
                    if start_idx + params_per_term > len(self.best_params_):
                        continue

                    const = self.best_params_[start_idx]
                    exponents = self.best_params_[start_idx + 1:start_idx + params_per_term]

                    # Skip terms with a negligible constant.
                    if abs(const) < threshold:
                        continue

                    # Build the variable part of the term.
                    term_parts = []
                    for j, exp in enumerate(exponents):
                        if abs(exp) > threshold:
                            if abs(exp - 1.0) < threshold:
                                term_parts.append(feature_names[j])
                            else:
                                term_parts.append(f"{feature_names[j]}^{exp:.2f}")

                    # Assemble the full term, hiding a unit constant.
                    if term_parts:
                        if abs(const - 1.0) < threshold:
                            term_str = " * ".join(term_parts)
                        else:
                            term_str = f"{const:.2f} * " + " * ".join(term_parts)
                        terms.append(term_str)
                    elif abs(const) > threshold:  # Constant-only term.
                        terms.append(f"{const:.2f}")

                # Join the terms into the class expression, handling signs.
                if terms:
                    if len(terms) == 1:
                        class_formula = f"z_{class_name} = {terms[0]}"
                    else:
                        formula_parts = [terms[0]]
                        for term in terms[1:]:
                            if term.startswith('-'):
                                formula_parts.append(f" - {term[1:]}")
                            else:
                                formula_parts.append(f" + {term}")
                        class_formula = f"z_{class_name} = " + "".join(formula_parts)
                else:
                    class_formula = f"z_{class_name} = 0"

                class_formulas.append(class_formula)

            logit_names = [f"z_{class_name}" for class_name in self.classes_]
            final_formula = f"\nP(class) = softmax([{', '.join(logit_names)}])\n"

        return "\n".join(class_formulas) + final_formula

    def get_params(self, deep=True):
        """Return estimator parameters as a dict (scikit-learn compatibility)."""
        return {
            'K': self.K,
            'l1_strength': self.l1_strength,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'num_epochs': self.num_epochs,
            'n_restarts': self.n_restarts,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'gradient_method': self.gradient_method,
            'use_sigmoid': self.use_sigmoid,
            'sigmoid_threshold': self.sigmoid_threshold,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        """Set estimator parameters in place (scikit-learn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self