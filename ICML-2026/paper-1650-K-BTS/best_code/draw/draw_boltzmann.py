import numpy as np
import matplotlib.pyplot as plt


def plot_boltzmann_svg(
    energies,
    labels,
    T=2.0,
    x_range=(-10, -5)
):

    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.exp(-x / T)

    plt.figure(figsize=(3, 2))
    plt.plot(x, y, linewidth=2.5)

    for xm, lab in zip(energies, labels):
        ym = np.exp(-xm / T)

        plt.plot(xm, ym, 'ro', markersize=9)

        plt.text(xm,
                 ym * 1.08,
                 lab,
                 color='red',
                 fontsize=11,
                 ha='center',
                 va='bottom')

        plt.text(xm,
                 ym * 0.82,
                 f'{ym:.2f}',
                 fontsize=10,
                 ha='center',
                 va='top')

    plt.ylim(0, max(y) * 1.2)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")

    filename = f"boltzmann_T_{T:.1f}.svg"
    plt.savefig(filename, format='svg', bbox_inches='tight', transparent=True)
    plt.close()

energies = [-9, -8, -7, -6]
labels = ['A', 'B', 'C', 'D']

plot_boltzmann_svg(
    energies=energies,
    labels=labels,
    T=2.0
)
