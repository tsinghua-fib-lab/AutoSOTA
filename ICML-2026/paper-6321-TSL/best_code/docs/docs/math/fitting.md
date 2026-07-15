# Fitting

TSL fitting combines **forward stagewise additive modeling** (boosting-style residual
fitting) with **CART-style partition refinement**. The structure mirrors the code exactly:

- the per-stage grid refinement lives in [`GridTensor`](../code/grid-tensor.md);
- the per-stage bagging + aggregation lives in [`StagePredictor`](../code/stage-predictor.md);
- the outer boosting loop and coefficient backfit live in [`TSL`](../code/forest.md).

## Forward stagewise additive modeling

At stage $\ell$ the previous stages are frozen and a new stage predictor
$\hat{m}^{(\ell)}$ is fit to the **outer residuals**

$$
R_i^{(\ell-1)} \coloneqq y^{(i)} - \sum_{k=1}^{\ell-1}\bigl[\hat{m}_+^{(k)}(\mathbf{x}^{(i)}) - \hat{m}_-^{(k)}(\mathbf{x}^{(i)})\bigr]
$$

(the empty sum at $\ell=1$ means the first stage is fit directly to $y$). After fitting
stage $\ell$, **all** stage scalars $\{\lambda_{\pm}^{(k)}\}_{k=1}^{\ell}$ are refit jointly
by least squares against $y$ (the [backfit](#coefficient-backfitting) below).

Within a stage, the **within-stage residual** is $r_i \coloneqq R_i^{(\ell-1)} - \hat{m}^{(\ell)}(\mathbf{x}^{(i)})$,
where $\hat{m}^{(\ell)}$ is the current in-progress stage prediction.

!!! note "Implementation note — unscaled products"
    Pseudocode and code use the **unscaled** per-sample product
    $\tilde{m}_{\pm}^{(i)} \coloneqq \prod_{j=1}^p \hat{m}_{\pm,j}^{(\ell)}(x_j^{(i)})$, so the
    full stage prediction is $\lambda_{+}^{(\ell)}\tilde{m}_{+}^{(i)} - \lambda_{-}^{(\ell)}\tilde{m}_{-}^{(i)}$.
    In the codebase the $\lambda$'s are applied once at the `StagePredictor` level as
    `scaling_plus`/`scaling_minus`; `GridTensor::predict_unscaled` and
    `extract_two_tensor_predictions_unscaled` deliberately return unscaled $\tilde{m}_+,\tilde{m}_-$.
    See the [architecture invariants](../code/architecture.md#two-critical-invariants).

## The grid tensor

A single stage product is fit on a shared adaptive partition. A **grid tensor** is the
tuple $\mathcal{G} = \bigl(\prod_{j=1}^p \mathcal{I}_j, \{\hat{v}_{\pm,j,I}\}\bigr)$, where
$\mathcal{I}_j = \{I_{j,1},\dots,I_{j,L_j}\}$ partitions the domain of $x_j$ and
$\hat{v}_{\pm,j,I}>0$ is the value on interval $I$. Each univariate factor is
piecewise-constant on its partition:

$$
\hat{m}_{\pm,j}^{(\ell)}(x_j) = \sum_{I\in\mathcal{I}_j^{(\ell)}} \hat{v}_{\pm,j,I}^{(\ell)}\,\mathbbm{1}_I(x_j),
$$

so each $(\pm)$ product is constant on each grid cell $\prod_j I_j$. Unlike CART — which
splits one leaf rectangle — a grid split partitions **all** intervals along an axis $j$ at
once, creating $\prod_{k\ne j} L_k$ new cells while preserving separability.

## Greedy split scoring

The grid is refined by greedily splitting intervals. A candidate split on axis $j$ at
threshold $s$ partitions an interval $I=[a,b)$ into $I_L=[a,s)$ and $I_R=[s,b)$. For each
region $S\in\{L,R\}$ the bin value is rescaled multiplicatively,
$\hat{v}_{\pm,j,I_S} = \hat{v}_{\pm,j,I}\,u_\pm^S$, and the updates $u_\pm^S$ minimize a
regularized least-squares objective:

$$
\mathcal{L}_S(u_+^S, u_-^S) = \sum_{x_j^{(i)}\in I_S} w_i\bigl(R_i^{(\ell-1)} - (u_+^S \tilde{m}_{+}^{(i)} - u_-^S \tilde{m}_{-}^{(i)})\bigr)^2 + \alpha\bigl((u_+^S-1)^2 + (u_-^S-1)^2\bigr),
$$

with stabilizing weights $w_i\ge 0$ and ridge strength $\alpha\ge 0$. The split that
maximizes the total reduction $\Delta_{\text{split}} = \Delta_L + \Delta_R$, where
$\Delta_S \coloneqq \mathcal{L}_S(1,1) - \mathcal{L}_S(u_+^S,u_-^S)$, is selected.

### The closed-form 2x2 solver

It is convenient to work with the **delta** $\hat{u}_\pm = u_\pm - 1$ (so the baseline
"no update" is $\hat{u}_\pm=0$). The objective becomes

$$
\mathcal{L}_S(\hat{u}_+,\hat{u}_-) = \sum_{i\in S} w_i\bigl(r_i - (\hat{u}_+\tilde{m}_{+}^{(i)} - \hat{u}_-\tilde{m}_{-}^{(i)})\bigr)^2 + \alpha(\hat{u}_+^2 + \hat{u}_-^2),
$$

minimized by a $2\times2$ linear system built from five **sufficient statistics**:

$$
\begin{aligned}
S_{11} &= \sum_{i\in S} w_i\bigl(\tilde{m}_{+}^{(i)}\bigr)^2, &
S_{22} &= \sum_{i\in S} w_i\bigl(\tilde{m}_{-}^{(i)}\bigr)^2, &
S_{12} &= -\sum_{i\in S} w_i\,\tilde{m}_{+}^{(i)}\tilde{m}_{-}^{(i)}, \\
t_1 &= \sum_{i\in S} w_i\,r_i\,\tilde{m}_{+}^{(i)}, &
t_2 &= -\sum_{i\in S} w_i\,r_i\,\tilde{m}_{-}^{(i)}. & &
\end{aligned}
$$

!!! note "Implementation note — tilt coupling (τ, ρ)"
    The solver `solve_two_tensor` (`src/grid_tensor/two_tensor_solver.rs`) optimizes a
    slightly more general objective than the ridge above, adding two regularizers that
    couple the two branches:
    $\;+\,\tau(\hat{u}_+ - \hat{u}_-)^2 + \rho\,|\hat{u}_+ - \hat{u}_-|$. The system matrix is
    then

    $$
    A = \begin{pmatrix} S_{11}+\alpha+\tau & S_{12}-\tau \\ S_{12}-\tau & S_{22}+\alpha+\tau \end{pmatrix},
    \qquad t = \begin{pmatrix} t_1 \\ t_2 \end{pmatrix}.
    $$

    Setting $\tau=\rho=0$ recovers the ridge form above. With $\rho=0$ the solution is the
    explicit inverse $\hat{u}=A^{-1}t$; with $\rho>0$ an iterative soft-thresholding step
    handles the $\ell_1$ term. Defaults keep $\tau$ small and $\rho=0$
    (see [Hyperparameters](../guides/hyperparameters.md)).

After solving, the updates are **clamped** to preserve positivity and stability:

$$
v_\pm^S = \mathrm{clamp}(1 + \hat{u}_\pm^S, v_{\min}, v_{\max}),
\qquad \mathrm{clamp}(x,a,b) = \max(a,\min(b,x)).
$$

Crucially the candidate is scored at the **clamped** update actually applied: writing
$\tilde{u}_\pm^S = v_\pm^S - 1$,

$$
\Delta_S = \mathcal{L}_S(0,0) - \mathcal{L}_S(\tilde{u}_+^S, \tilde{u}_-^S).
$$

The L2 and Huber refinement strategies (`RefinementStrategy::L2` / `Huber`) differ only in
the weights $w_i$: L2 uses $w_i=1$; Huber uses $w_i=\min(1, c/|r_i|)$ with $c\approx1.345$.

## Initialization and the positive-only special case

A stage starts with both products constant ($\hat{m}_{\pm,j}=1$ everywhere) and the scalars
initialized from residual signs. When **all** outer residuals are non-negative
($R_i\ge0$ for all $i$ — e.g. stage 1 with non-negative targets), TSL uses the simplified
**positive-only** mode: $\lambda_{-}^{(\ell)}=0$, the $(-)$ product stays at 1, and the
objective reduces to 1D ridge least squares on the $(+)$ product,

$$
\hat{u}_+^S = \frac{\sum_{i\in S} w_i\,r_i\,\tilde{m}_{+}^{(i)}}{\sum_{i\in S} w_i\,(\tilde{m}_{+}^{(i)})^2 + \alpha}.
$$

This invariant — positive-only stage 1 produces non-negative predictions with $d_j=0$ — is
tested in `tests/stage1_positive_only.rs`.

## Candidate sampling, stopping, and binning

- **Sampling.** Two parameters bound the search per iteration:
  $\texttt{split_try}$ split positions are sampled per (feature, interval) from the
  *valid* positions (those leaving $\ge\texttt{min_interval_samples}$ on both sides), and
  $\texttt{colsample}\in[0,1]$ samples a fraction of features. The `SplitStrategy` enum
  offers `Random` (default), `Best`, and `TopK`.
- **Stopping.** Refinement stops when `n_iter` iterations are reached, no split exceeds the
  minimum error-reduction threshold, the `min_interval_samples` constraint would be
  violated, or no valid split remains.
- **Histogram binning.** Setting `max_bins` restricts candidate positions to quantile bin
  edges and replaces per-position prefix sums with per-bin cumulative sums, cutting the
  per-candidate cost from $O(n)$ to $O(\text{bins})$. See
  [GridTensor → histogram binning](../code/grid-tensor.md#histogram-binning).

Efficient evaluation relies on **prefix-sum caches** per axis, giving $O(1)$ queries for
the sufficient statistics of any candidate.

## Bagging

To reduce variance, $n_{\text{grids}}$ grid tensors are fit independently (embarrassingly
parallel) and aggregated into one stage component. Because an average of products is not a
product, aggregation works on the per-feature components in the backbone/tilt gauge — see
the dedicated page on [Bagging & aggregation](bagging-aggregation.md).

## Coefficient backfitting

After stage $\ell$ is fit, all stage scalars are refit by ordinary least squares. Build the
design matrix $M\in\mathbb{R}^{n\times 2\ell}$ whose columns are the per-stage unscaled
products,

$$
M_{i,k} = \prod_{j=1}^p \hat{m}_{+,j}^{(k)}(x_j^{(i)}), \qquad
M_{i,\ell+k} = -\prod_{j=1}^p \hat{m}_{-,j}^{(k)}(x_j^{(i)}), \qquad k=1,\dots,\ell,
$$

and solve $\boldsymbol{\lambda} = (M^\top M)^{-1} M^\top \mathbf{y}$ with
$\boldsymbol{\lambda} = [\lambda_{+}^{(1)},\dots,\lambda_{+}^{(\ell)},\lambda_{-}^{(1)},\dots,\lambda_{-}^{(\ell)}]^\top$.
This **orthogonal-greedy** refit makes the coefficients jointly optimal given all stages so
far, improving on frozen-coefficient greedy fitting.

!!! note "Implementation note — where the coefficients live"
    The backfit is solved in `src/forest/fitter.rs` (via `LeastSquaresSvd`) over the
    $[\tilde{m}_+^{(k)}, -\tilde{m}_-^{(k)}]$ columns of every completed stage. The solved coefficients are
    stored as `scaling_plus`/`scaling_minus` on each `StagePredictor` rather than mutating
    the grid's `lambda_plus`/`lambda_minus` — mathematically equivalent, and the reason
    scaling is applied exactly once at the stage level.

## Cost

Total training cost is $\mathcal{O}(R\, n_{\text{grids}}\, T\, n\, p)$ for $R$ stages,
$n_{\text{grids}}$ bagged grids, $T$ split iterations per grid, $n$ samples, $p$ features.
Bagged grids are parallel, and histogram binning trades the $n$ factor for the bin count.
