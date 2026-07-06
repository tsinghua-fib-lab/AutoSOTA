import torch


def sample_gmm(means, covs, weights, num_samples=500):
    num_components = means.shape[0]
    samples = []
    component_indices = torch.multinomial(weights, num_samples, replacement=True)

    for i in range(num_components):
        num_i = (component_indices == i).sum().item()
        if num_i > 0:
            mvn = torch.distributions.MultivariateNormal(means[i], covariance_matrix=covs[i])
            samples.append(mvn.sample((num_i,)))

    return torch.cat(samples, dim=0)


