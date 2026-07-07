import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


def save_beta_distribution_svg(alpha, beta_param, theta=None):
    x = np.linspace(0, 1, 500)
    y = beta.pdf(x, alpha, beta_param)

    plt.figure(figsize=(3, 2))
    plt.plot(x, y, linewidth=2.5)

    if theta is not None:
        y_theta = beta.pdf(theta, alpha, beta_param)

        plt.plot(theta, y_theta, 'ro', markersize=9)

        plt.vlines(theta, 0, y_theta,
                   linestyles='dashed',
                   colors='black',
                   linewidth=1.2)
        plt.text(theta, y_theta * 0.7, r'$\hat{\theta}$',
                 color='red',
                 fontsize=15,
                 ha='center',
                 va='bottom')
        plt.text(theta, -0.025 * max(y),
                 f'{theta:.2f}',
                 ha='center',
                 va='top',
                 fontsize=12)

    plt.ylim(0, max(y) * 1.15)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")

    filename = f"alpha_{alpha:.1f}_beta_{beta_param:.1f}_theta_{theta:.1f}.svg"
    plt.savefig(filename, format='svg', bbox_inches='tight', transparent=True)
    plt.close()


save_beta_distribution_svg(alpha=2, beta_param=3, theta=0.4)
save_beta_distribution_svg(alpha=3, beta_param=2, theta=0.7)
save_beta_distribution_svg(alpha=4, beta_param=2, theta=0.8)
save_beta_distribution_svg(alpha=1, beta_param=1, theta=0.3)
