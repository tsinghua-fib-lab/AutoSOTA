import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import softjax as sj

jnp.set_printoptions(precision=4, suppress=True)
jax.config.update("jax_enable_x64", True)

# 1. Median regression
# Minimize the median absolute residual to be robust to outliers.

key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (20, 3))
w_true = jnp.array([1.0, -2.0, 0.5])
y = X @ w_true
y = y.at[0].set(1e6)  # inject outlier


def median_regression_loss(w, X, y, mode="smooth"):
    residuals = y - X @ w
    return sj.median(sj.abs(residuals, mode=mode), mode=mode)

w = jnp.zeros(3)
print("=== 1. Robust median regression ===")
print("Hard grad:", jax.grad(median_regression_loss)(w, X, y, mode="hard"))
print("Soft grad:", jax.grad(median_regression_loss)(w, X, y, mode="smooth"))

ws = []
for _ in range(50):
    ws.append(w.tolist())
    grad = jax.grad(median_regression_loss)(w, X, y)
    w = w - 0.1 * grad
print("Learned w:", w, " (true:", w_true, ")")


# 2. Top-k feature selection
# Discover which features of a trained model are important.
# 10 features total, only 3 informative — learn gating scores to find them.

n_features, k_sel = 10, 3
k1, k2 = jax.random.split(jax.random.PRNGKey(42))
X_fs = jax.random.normal(k1, (100, n_features))
w_model = jnp.array([0, 2.0, 0, -1.5, 0, 0, 0, 5.0, 0, 0])  # trained model
y_fs = X_fs @ w_model + 0.1 * jax.random.normal(k2, (100,))


def feature_selection_loss(g, X, y, w_model, mode="smooth"):
    _, soft_idx = sj.top_k(g, k=k_sel, mode=mode, gated_grad=False)
    mask = soft_idx.sum(axis=0)
    y_pred = (X * mask) @ w_model
    return jnp.mean(sj.abs(y_pred - y))


g = jnp.zeros(n_features)
print("\n=== 2. Top-k feature selection ===")
print("Hard grad:", jax.grad(feature_selection_loss)(g, X_fs, y_fs, w_model, mode="hard"))
print("Soft grad:", jax.grad(feature_selection_loss)(g, X_fs, y_fs, w_model, mode="smooth"))

gs = []
for _ in range(5):
    gs.append(g.tolist())
    grad = jax.grad(feature_selection_loss)(g, X_fs, y_fs, w_model)
    g = g - 0.001 * grad
    
print("Selected features:", jax.lax.top_k(g, k=k_sel)[1])
print("Feature scores:", g)

# 3. Differentiable filter
# Learn a threshold that gates inputs.

x_filt = jnp.array([0.2, 0.8, 0.5, 1.2, 0.1])
target_sum = 2.0  # sum of values above threshold should equal 2.0 (= 0.8 + 1.2)


def filter_loss(t, x, target, mode="smooth"):
    mask = sj.greater(x, t, mode=mode)
    return (jnp.sum(mask * x) - target) ** 2


t = jnp.array(0.0)
print("\n=== 3. Differentiable threshold filtering ===")
print("Hard grad:", jax.grad(filter_loss)(t, x_filt, target_sum, mode="hard"))
print("Soft grad:", jax.grad(filter_loss)(t, x_filt, target_sum, mode="smooth"))

ts = []
for _ in range(20):
    ts.append(float(t))
    grad = jax.grad(filter_loss)(t, x_filt, target_sum)
    t = t - 0.1 * grad
print("Learned threshold:", t)


# 4. Differentiable rule-based classifier
# Learn decision boundaries: classify positive if ANY feature is in [lo, hi].
# The rule is true if any element of a feature is inside `[lo, hi]`.
x_rules = jnp.array([[0.2, 0.8], [0.5, 0.3], [0.9, 0.1], [0.4, 0.7],
                     [0.1, 0.4], [0.2, 0.7], [0.4, 0.1], [0.4, 0.7],
                     [0.7, 0.29], [0.3, 0.3], [0.61, 0.25], [0.4, 0.6],
                     [0.0, 0.1], [0.5, 0.3], [0.4, 0.9], [0.1, 0.57],
                     ])
labels = jnp.array([0.0, 1.0, 0.0, 1.0, 
                    1.0, 0.0, 1.0, 1.0,
                    0.0, 1.0, 0.0, 1.0, 
                    0.0, 1.0, 1.0, 1.0])


@sj.st
def rule_loss(params, x, labels, mode="smooth"):
    lo, hi = params[0], params[1]
    above = sj.greater(x, lo, mode=mode)
    below = sj.less(x, hi, mode=mode)
    in_range = sj.logical_and(above, below)
    preds = sj.any(in_range, axis=-1)
    return ((preds - labels) ** 2).sum()


params = jnp.array([0.0, 1.0])  # start with wide range [0, 1]
print("\n=== 4. Differentiable rule-based classifier ===")
print("Hard loss:", rule_loss(params, x_rules, labels, mode="hard"))
print("Soft loss:", rule_loss(params, x_rules, labels, mode="smooth"))
print("Hard grad:", jax.grad(rule_loss)(params, x_rules, labels, mode="hard"))
print("Soft grad:", jax.grad(rule_loss)(params, x_rules, labels, mode="smooth"))

params_hist = []
for _ in range(20):
    params_hist.append(params.tolist())
    grad = jax.grad(rule_loss)(params, x_rules, labels)
    params = params - 0.01 * grad
print("Learned [lo, hi]:", params)


# ── Plot ─────────────────────────────────────────────────────────────────────
palette = ["#00bfff","#e7a1e5", "#6dd1ac", "#e1be6a", "#368f80",  "#889fd9", "#f4836d", "#cecece"]
informative = {i for i, v in enumerate(w_model) if v != 0}

fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Iteration", fontsize=7)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.margins(x=0)

ws = jnp.array(ws)
for i in range(ws.shape[1]):
    axes[0].plot(ws[:, i], color=palette[i], label=f"w[{i}]")
    axes[0].axhline(w_true[i], color=palette[i], ls="--", alpha=0.3)
axes[0].set_title("Median regression", fontsize=8)
axes[0].legend(fontsize=6)

gs = jnp.array(gs)
for i in range(gs.shape[1]):
    if i in informative:
        if i == 1:
            kw = {"lw": 1.5, "color": "#6dd1ac", "label": "Informative"}
        else: 
            kw = {"lw": 1.5, "color": "#6dd1ac", "label": None}
    else:
        if i == 4:
            kw = {"alpha": 0.2, "color": "#889fd9", "label": "Uninformative"}
        else:
            kw = {"alpha": 0.2, "color": "#889fd9", "label": None}
    axes[1].plot(gs[:, i], **kw)
axes[1].set_title("Top-k feature selection", fontsize=8)
axes[1].legend(fontsize=6, title="Feature scores", title_fontsize=6)

axes[2].plot(ts, color=palette[0])
for xi in x_filt:
    axes[2].axhline(xi, ls="--", color=palette[-1], alpha=0.5)
axes[2].set_title("Threshold filtering", fontsize=8)

params_hist = jnp.array(params_hist)
axes[3].plot(params_hist[:, 1], color=palette[0], label="higher bound")
axes[3].plot(params_hist[:, 0], color=palette[2], label="lower bound")
axes[3].axhline(0.3, ls="--", color=palette[2], alpha=0.5)
axes[3].axhline(0.6, ls="--", color=palette[0], alpha=0.5)
axes[3].set_title("Rule classifier", fontsize=8)
axes[3].legend(fontsize=6)

fig.tight_layout()
fig.savefig("docs/examples/quick_example_optimization.svg", bbox_inches="tight", transparent=True)