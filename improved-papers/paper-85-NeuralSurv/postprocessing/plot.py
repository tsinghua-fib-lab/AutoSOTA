import matplotlib.pyplot as plt
from scipy.stats import gamma
import numpy as np


def plot_survival_function_vs_km_curve(
    survival, km_surv_time, km_surv, plot_dir=None, label=""
):

    # Posterior survival function
    surv_time = survival["times"]
    surv_median = survival["median"]

    # Plot
    plt.figure(figsize=(10, 6))
    if "q025" in survival:
        surv_q025 = survival["q025"]
        surv_q975 = survival["q975"]
        plt.fill_between(surv_time, surv_q025, surv_q975, color="blue", alpha=0.2)
    plt.plot(surv_time, surv_median, color="blue", label="Posterior Approximation")
    plt.plot(km_surv_time, km_surv, color="red", label="KM Estimate")

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Survival Function")
    plt.legend()

    # Save or display the plot
    if plot_dir is not None:
        file_name = "/survival_function_posterior_vs_km_" + label + ".pdf"
        plt.savefig(plot_dir + file_name, dpi=300, bbox_inches="tight", format="pdf")
    else:
        plt.show()


def plot_survival_function(survival, plot_dir=None, label=""):

    # Posterior survival function
    surv_time = survival["times"]
    surv_median = survival["median"]

    # Plot
    plt.figure(figsize=(10, 6))
    if "q025" in survival:
        surv_q025 = survival["q025"]
        surv_q975 = survival["q975"]
        plt.fill_between(surv_time, surv_q025, surv_q975, color="blue", alpha=0.2)
    plt.plot(surv_time, surv_median, color="blue", label="Posterior Approximation")

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Survival Function")
    plt.legend()

    # Save or display the plot
    if plot_dir is not None:
        file_name = "/survival_function_posterior" + label + ".pdf"
        plt.savefig(plot_dir + file_name, dpi=300, bbox_inches="tight", format="pdf")
    else:
        plt.show()


def plot_hazard_function(hazard, plot_dir=None, label=""):

    # Posterior survival function
    hazard_time = hazard["times"]
    hazard_median = hazard["median"]

    # Plot
    plt.figure(figsize=(10, 6))
    if "q025" in hazard:
        hazard_q025 = hazard["q025"]
        hazard_q975 = hazard["q975"]
        plt.fill_between(hazard_time, hazard_q025, hazard_q975, color="blue", alpha=0.2)
    plt.plot(hazard_time, hazard_median, color="blue", label="Posterior Approximation")

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Hazard Function")
    plt.legend()

    # Save or display the plot
    if plot_dir is not None:
        file_name = "/hazard_function_posterior_" + label + ".pdf"
        plt.savefig(plot_dir + file_name, dpi=300, bbox_inches="tight", format="pdf")
    else:
        plt.show()


def plot_glin_function(glin, plot_dir=None, label=""):

    # Posterior survival function
    glin_time = glin["times"]
    glin_median = glin["median"]

    # Plot
    plt.figure(figsize=(10, 6))
    if "q025" in glin:
        glin_q025 = glin["q025"]
        glin_q975 = glin["q975"]
        plt.fill_between(glin_time, glin_q025, glin_q975, color="blue", alpha=0.2)
    plt.plot(glin_time, glin_median, color="blue", label="Approximation")

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("glin Function")
    plt.legend()

    # Save or display the plot
    if plot_dir is not None:
        file_name = "/glin_function_" + label + ".pdf"
        plt.savefig(plot_dir + file_name, dpi=300, bbox_inches="tight", format="pdf")
    else:
        plt.show()


def plot_posterior_phi(alpha, beta, plot_dir=None):

    # Create an array of x values for the plot
    x = np.linspace(0, 2, 1000)

    # Compute the PDF of the Gamma distribution for these x values
    pdf = gamma.pdf(x, alpha, scale=1 / beta)

    # Calculate the 95% quantile range
    lower_95 = gamma.ppf(0.025, alpha, scale=1 / beta)  # 2.5% quantile
    upper_95 = gamma.ppf(0.975, alpha, scale=1 / beta)  # 97.5% quantile

    # Calculate the median (50% quantile)
    median = gamma.ppf(0.5, alpha, scale=1 / beta)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, label=f"Gamma Distribution", color="blue")
    plt.fill_between(
        x,
        pdf,
        where=(x >= lower_95) & (x <= upper_95),
        color="blue",
        alpha=0.2,
        label="95% Quantile Range",
    )
    plt.plot(
        median,
        0,
        "rx",
        label=f"Median (x={median:.2f})",
    )

    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("Posterior distribution phi")
    plt.legend()

    # Save or display the plot
    if plot_dir is not None:
        file_name = "/posterior_distribution_phi.pdf"
        plt.savefig(plot_dir + file_name, dpi=300, bbox_inches="tight", format="pdf")
    else:
        plt.show()
