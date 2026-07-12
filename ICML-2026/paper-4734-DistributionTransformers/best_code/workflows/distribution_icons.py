import torch
from torch import tensor

from distributions.distributions import GaussianMixtureModel
from distributions.utils import decode_gmm_sample, encode_gmm_sample

import matplotlib.pyplot as plt

prior_dict = {'weights': tensor([0.4000, 0.3000, 0.3000]), 'loc': tensor([[-0.5000],
    [ 0.2000],
    [ 1.0000]]), 'scale_tril': tensor([[[0.3162]],
    [[0.2236]],
    [[0.7071]]])}

posterior_dict = {'weights': tensor([0.2294, 0.3610, 0.4096]), 'loc': tensor([[-0.3636],
    [ 0.2381],
    [ 1.0000]]), 'scale_tril': tensor([[[0.3015]],
    [[0.2182]],
    [[0.5774]]])}


model_prior_dict = {'weights': tensor([0.3864, 0.3101, 0.3035]), 'loc': tensor([[-0.4856],
    [ 0.2172],
    [ 0.8936]]), 'scale_tril': tensor([[[0.3570]],
    [[0.2468]],
    [[0.7722]]])}


model_posterior_dict = {'weights': tensor([0.2233, 0.3361, 0.4407]), 'loc': tensor([[-0.3459],
    [ 0.2355],
    [ 1.0038]]), 'scale_tril': tensor([[[0.2981]],
    [[0.2173]],
    [[0.5633]]])}

x = torch.linspace(-2, 3, 1000)

def plot(true: dict, approximate: dict, label: str):
    fig, ax = plt.subplots()
    ax.plot(GaussianMixtureModel(**true).log_prob(x.unsqueeze(-1)).squeeze(-1).exp(), "k", linewidth=7)
    ax.set_axis_off()
    plt.show()
    fig.savefig(f"../clipart/true_{label}.svg", format="svg")

    fig, ax = plt.subplots()
    ax.plot(GaussianMixtureModel(**approximate).log_prob(x.unsqueeze(-1)).squeeze(-1).exp(), "k", linewidth=7)
    ax.set_axis_off()
    plt.show()
    fig.savefig(f"../clipart/approximate_{label}.svg", format="svg")

    phi = encode_gmm_sample(approximate)
    for i, component in enumerate(phi):
        fig, ax = plt.subplots()
        ax.plot(component[0] * GaussianMixtureModel(**decode_gmm_sample(component.unsqueeze(0))).log_prob(x.unsqueeze(-1)).squeeze(-1).exp(), "k",
                        linewidth=7)
        ax.set_axis_off()
        plt.show()
        fig.savefig(f"../clipart/approximate_{label}_component_{i}.svg", format="svg")

plot(prior_dict, model_prior_dict, "prior")
plot(posterior_dict, model_posterior_dict, "posterior")
