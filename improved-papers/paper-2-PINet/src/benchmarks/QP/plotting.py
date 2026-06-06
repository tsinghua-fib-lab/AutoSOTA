"""Plotting for benchmarks."""

import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_rs_vs_cv(
    obj_fun_test: jnp.ndarray,
    obj_test: jnp.ndarray,
    eq_viol_test: jnp.ndarray,
    ineq_viol_test: jnp.ndarray,
    cvthres: float,
    rsthres: float,
):
    """Plot the relative suboptimality against constraint violation.

    Args:
        obj_fun_test (jnp.ndarray): The objective function values from the test set.
        obj_test (jnp.ndarray): The objective values estimated.
        eq_viol_test (jnp.ndarray): The equality constraint violations.
        ineq_viol_test (jnp.ndarray): The inequality constraint violations.
        cvthres (float): The threshold value for constraint violation used for plotting.
        rsthres (float): The threshold value for relative suboptimality used for plotting.

    Returns:
        tuple:
            fig (matplotlib.figure.Figure): The created figure.
            rs (jnp.ndarray or float): Computed relative suboptimality.
            cv (jnp.ndarray or float): Computed constraint violations.
    """
    fig, ax = plt.subplots()
    # Relative suboptimality
    rs = (obj_fun_test - obj_test) / jnp.abs(obj_test)
    cv = jnp.maximum(eq_viol_test, ineq_viol_test)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(jnp.minimum(1e-14, jnp.min(cv)), jnp.maximum(1e-1, jnp.max(cv)))
    ax.set_ylim(jnp.minimum(1e-5, jnp.min(rs)), jnp.maximum(1e-1, jnp.max(rs)))
    # Plot thresholds

    ax.hlines(
        y=rsthres,
        xmin=ax.get_xlim()[0],
        xmax=cvthres,
        color="red",
        linestyle="--",
    )
    ax.plot(
        [cvthres, cvthres],
        [ax.get_ylim()[0], rsthres],
        color="red",
        linestyle="--",
    )
    ax.scatter(cv, rs, label="Πnet")
    ax.set_xlabel("Constraint Violation")
    ax.set_ylabel("Relative Suboptimality")
    ax.set_title("Πnet")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, rs, cv


def plot_inference_boxes(
    single_inference_times: jnp.ndarray, batch_inference_times: jnp.ndarray
):
    """Plot box plots for single and batch inference.

    Args:
        single_inference_times (jnp.ndarray):
            A sequence of inference times (in seconds) for single inference.
        batch_inference_times (jnp.ndarray):
            A sequence of inference times (in seconds) for batch inference.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    bp1 = ax1.boxplot(
        [single_inference_times],
        positions=[1],
        widths=0.6,
        patch_artist=True,
        showfliers=True,
    )
    bp2 = ax2.boxplot(
        [batch_inference_times],
        positions=[2],
        widths=0.6,
        patch_artist=True,
        showfliers=True,
    )

    bp1["boxes"][0].set_facecolor("#1f77b4")
    bp2["boxes"][0].set_facecolor("#ff7f0e")

    def pad_ylim(ax, data, pad_ratio=0.10):
        ymin, ymax = jnp.min(data), jnp.max(data)
        span = ymax - ymin
        pad = span * pad_ratio if span else 1.0
        ax.set_ylim(ymin - pad, ymax + pad)

    pad_ylim(ax1, single_inference_times)
    pad_ylim(ax2, batch_inference_times)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Single Inference", "Batch Inference"])

    ax1.set_ylabel("Inference Time [s]")
    ax2.set_ylabel("Inference Time [s]")
    ax1.set_title("Inference Time")

    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig
