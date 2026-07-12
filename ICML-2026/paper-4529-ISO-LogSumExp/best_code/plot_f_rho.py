import numpy as np
import matplotlib.pyplot as plt


def f_rho(t, rho):
    return t * np.log(t) + 1 + ((1 - rho * t) / rho) * np.log(1 - rho * t)


def f_0(t):
    return t * np.log(t) - t + 1


def main():
    plt.figure(figsize=(10, 8))

    rhos = [0.1, 0.05, 0.01]
    t_dense = np.linspace(1e-6, 11, 1000)  # for f_0(t)

    # Plot f_rho(t) for each rho
    for rho in rhos:
        t_vals = np.linspace(1e-6, min(1/rho - 1e-6, 11), 1000)
        plt.plot(t_vals, f_rho(t_vals, rho), label=fr"$\rho={rho}$")

    # t log t − t + 1
    lbl = r"$f_0(t) = t\ \log t − t + 1$"
    plt.plot(t_dense, f_0(t_dense), label=lbl, linewidth=2)

    plt.xlabel(r"$t$", fontsize=15)
    plt.ylabel(r"$f_{\rho}(t)$", fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.xlim(-0.1, 10.3)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig('f_rho.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Plot of f_rho has been saved to f_rho.png.")


if __name__ == "__main__":
    main()
