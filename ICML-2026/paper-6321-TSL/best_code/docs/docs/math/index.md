# Notation

This page collects the symbols used throughout the *Under the hood* section. TSL solves a
supervised regression problem: estimate the conditional mean
$m(\mathbf{x}) = \mathbb{E}[Y \mid X = \mathbf{x}]$ from i.i.d. observations
$\mathcal{D}_n = \{(y^{(i)}, \mathbf{x}^{(i)})\}_{i=1}^n \subseteq \mathbb{R}\times\mathcal{X}$,
where the feature support $\mathcal{X}\subseteq\mathbb{R}^p$ may be **non-rectangular**.

## Indices and data

| Symbol | Meaning |
|--------|---------|
| $p$ | number of features |
| $n$ | number of training samples |
| $[k]$ | the set $\{1,\dots,k\}$ |
| $j \in [p]$ | feature / coordinate index |
| $\ell, k \in [R]$ | stage index |
| $x_j$ / $X_j$ | realized value / random variable for coordinate $j$ |
| $\mathbf{x},\,\mathbf{x}^{(i)}$ | feature vector (the $i$-th sample) |
| $X_{(-j)}$ | random subvector over all coordinates except $j$ |
| $y^{(i)}$ / $Y$ | response (sample $i$) / response random variable |
| $m(\mathbf{x})$ | true conditional mean $\mathbb{E}[Y\mid X=\mathbf{x}]$ |
| $\hat{m}(\mathbf{x})$ | the fitted TSL estimator |

## Model

| Symbol | Meaning |
|--------|---------|
| $R$ | number of stages (separation rank $\le 2R$) |
| $\hat{m}^{(\ell)}(\mathbf{x})$ | stage-$\ell$ predictor, $\hat{m}_{+}^{(\ell)} - \hat{m}_{-}^{(\ell)}$ |
| $\hat{m}_{\pm,j}^{(\ell)}(x_j)$ | positive univariate factor for feature $j$, sign branch $\pm$ ($>0$) |
| $\hat{m}_{\pm}^{(\ell)}(\mathbf{x})$ | **scaled** signed branch $\lambda_{\pm}^{(\ell)}\prod_j \hat{m}_{\pm,j}^{(\ell)}(x_j) = \lambda_{\pm}^{(\ell)}\,\tilde{m}_{\pm}^{(\ell)}\ (\ge 0)$ |
| $\lambda_{+}^{(\ell)},\lambda_{-}^{(\ell)}$ | non-negative stage scalars |
| $b_j^{(\ell)}(x_j)$ | **backbone** (magnitude / activity gate), $>0$ |
| $d_j^{(\ell)}(x_j)$ | **tilt** (signed log-imbalance), $\in\mathbb{R}$ |
| $b^{(\ell)}(\mathbf{x})$ | stage backbone $b_0^{(\ell)}\prod_j b_j^{(\ell)}(x_j)$ |
| $d^{(\ell)}(\mathbf{x})$ | stage tilt $d_0^{(\ell)} + \sum_j d_j^{(\ell)}(x_j)$ |
| $b_0^{(\ell)},\,d_0^{(\ell)}$ | stage-level backbone scale / tilt intercept (absorb $\lambda_\pm$) |

## Grid tensor & fitting

| Symbol | Meaning |
|--------|---------|
| $\mathcal{G}^{(\ell)}$ | a grid tensor: interval partitions + per-bin values |
| $\mathcal{I}_j^{(\ell)}$ | interval partition of axis $j$; $L_j^{(\ell)} = \#\mathcal{I}_j^{(\ell)}$ |
| $\hat{v}_{\pm,j,I}^{(\ell)}$ | value of $\hat{m}_{\pm,j}^{(\ell)}$ on interval $I$ ($>0$) |
| $\mathbbm{1}_I(x_j)$ | indicator that $x_j$ falls in interval $I$ |
| $R_i^{(\ell-1)}$ | outer residual at stage $\ell$ (target minus previous stages) |
| $r_i$ | within-stage residual (outer residual minus current stage) |
| $\tilde{m}_{\pm}^{(\ell)},\ \tilde{m}_{\pm}^{(i)}$ | **unscaled** product $\prod_j \hat{m}_{\pm,j}^{(\ell)}(x_j)$ (per-sample at $\mathbf{x}^{(i)}$); the scaled branch is $\hat{m}_{\pm}^{(\ell)} = \lambda_{\pm}^{(\ell)}\,\tilde{m}_{\pm}^{(\ell)}$ |
| $w_i$ | stabilizing weight ($=1$ for L2; Huber-dependent otherwise) |
| $u_\pm^S,\ \hat{u}_\pm^S$ | multiplicative bin update on region $S$, and its delta $u_\pm-1$ |
| $\alpha$ | ridge regularization on the bin update |
| $\tau,\rho$ | tilt-coupling regularizers (implementation; see [Fitting](fitting.md)) |
| $v_{\min},v_{\max}$ | clamp bounds keeping updates positive/stable |
| $S_{11},S_{22},S_{12},t_1,t_2$ | sufficient statistics of the $2\times2$ split system |
| $\Delta_S,\ \Delta_{\text{split}}$ | per-region / total error reduction from a split |
| $n_{\text{grids}}$ | number of bagged grids per stage (`n_trees`) |
| $\xi$ | similarity trim threshold; keep top $K=\lceil(1-\xi)n_{\text{grids}}\rceil$ |

## Interpretation

| Symbol | Meaning |
|--------|---------|
| $\mathrm{PD}_j(x_j)$ | partial dependence $\mathbb{E}_{X_{(-j)}}[m(x_j, X_{(-j)})]$ |
| $\mathrm{PD}_{\pm,j}^{(\ell)}(x_j)$ | signed-branch partial dependence of stage $\ell$ |
| $C^{(\ell)}_{\pm,j}$ | per-stage/feature/branch PD constant $c^{(\ell)}_{\pm,j}\lambda_\pm^{(\ell)}$ |

!!! note "Branch sign convention"
    The "$+$" and "$-$" branches are **not** positive/negative values — both products are
    non-negative ($\hat{m}_{\pm}^{(\ell)} \ge 0$). Signed effects arise only from the
    *difference* $\hat{m}_{+}^{(\ell)} - \hat{m}_{-}^{(\ell)}$. See [The model](model.md).
