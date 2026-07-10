import torch
import numpy as np
import quadprog
import sklearn.covariance


# Learn the importance weighting assuming two distributions are gaussian
def estimate_gauss_ratio(x_test, x_train):
    cov_est = sklearn.covariance.LedoitWolf()

    def p_gauss(x, mu, S2):
        denom = torch.sqrt(pow(2 * torch.pi, len(x)) * torch.det(S2))
        return (1 / denom) * torch.exp(
            -0.5 * (x - mu) @ torch.linalg.inv(S2) @ (x - mu)
        )

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    mu_test = torch.sum(x_test, dim=0) / num_test
    S2_test = torch.tensor(cov_est.fit(x_test - mu_test).covariance_).float()
    mu_train = torch.sum(x_train, dim=0) / num_train
    S2_train = torch.tensor(cov_est.fit(x_train - mu_train).covariance_).float()

    def p_test(x):
        return p_gauss(x, mu_test, S2_test)

    def p_train(x):
        return p_gauss(x, mu_train, S2_train)

    w_hat = torch.vmap(p_test)(x_train) / torch.vmap(p_train)(x_train)
    return w_hat


# Kernel mean matching to estimate the ratio of p_test / p_train
# Lambda corresponds to kernel inverse regularizer
def kernel_mean_matching(
    x_test, x_train, normalize=False, bound=10.0, eps_scale=0.01, lamb=1e-3
):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    # Set sigma to the median distance between all samples
    dist_x = torch.cdist(x_train, x_train)
    sigma = dist_x.median()
    # Construct Kernel matrices between train and test
    ker_mat = torch.exp(-(dist_x**2) / (2 * (sigma**2)))
    ker_vec = torch.sum(
        torch.exp(-(torch.cdist(x_train, x_test) ** 2) / (2 * (sigma**2))), dim=1
    ) * (num_train / num_test)

    eps = eps_scale * bound / np.sqrt(num_train)
    G = ker_mat.double().numpy()
    G += lamb * np.eye(num_train)
    a = ker_vec.double().numpy()
    if normalize is True:
        C = (
            torch.cat(
                (
                    torch.ones(1, num_train),
                    -torch.ones(1, num_train),
                    torch.eye(num_train),
                    -torch.eye(num_train),
                )
            )
            .double()
            .numpy()
            .T
        )
        b = (
            torch.cat(
                (
                    torch.tensor([(1 - eps) * num_train, -(1 + eps) * num_train]),
                    torch.zeros(num_train),
                    -bound * torch.ones(num_train),
                )
            )
            .double()
            .numpy()
        )
    else:
        C = torch.cat((torch.eye(num_train), -torch.eye(num_train))).double().numpy().T
        b = (
            torch.cat((torch.zeros(num_train), -bound * torch.ones(num_train)))
            .double()
            .numpy()
        )
    w_hat = quadprog.solve_qp(G, a, C, b)[0]
    w_hat = np.maximum(np.zeros(num_train), w_hat)
    return torch.tensor(w_hat)


def prob_classifier_weighting(x_test, x_train, learning_rate=1e-3, num_epochs=1000):
    class log_reg(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.layer = torch.nn.Linear(dim, 2)

        def forward(self, x):
            return self.layer(x)

    num_train, num_dim = x_train.shape
    model = log_reg(dim=num_dim)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_test = x_test.shape[0]
    x_all = torch.cat((torch.tensor(x_train), torch.tensor(x_test))).float()
    x_all = x_all.reshape((num_train + num_test, num_dim))
    y_all = torch.cat((torch.zeros(num_train), torch.ones(num_test))).long()
    loss_fun = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        loss = loss_fun(model(x_all), y_all)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"loss: {loss:>7f} epoch:{epoch+1}")
    probs = torch.nn.Softmax()(model(x_all[:num_train]))
    w_est = (num_train / num_test) * probs[:, 1] / probs[:, 0]
    return w_est.detach().numpy(), model


def apply_importance_weighting(x_source, x_target, y_source, method, max_weight):
    num_dim = len(x_source.shape)
    if num_dim == 1:
        x_source = x_source[:, torch.newaxis]
        x_target = x_target[:, torch.newaxis]
        num_test = x_target.shape[0]
    else:
        num_dim = x_source.shape[-1]
        num_test = x_target.shape[0]
    num_train = y_source.shape[0]

    X_train = torch.hstack((torch.ones(num_train)[:, torch.newaxis], x_source))
    if (
        method == "de"
    ):  # direct density estimation of both p and q (assuming Gaussian distributions)
        w = estimate_gauss_ratio(x_target, x_source)
    elif method == "kmm":
        w = kernel_mean_matching(
            x_target,
            x_source,
            normalize=False,
            bound=max_weight,
            eps_scale=0.01,
            lamb=1e-3,
        )
    elif method == "logreg":  # logistic regression
        w, _ = prob_classifier_weighting(x_target, x_source)
    else:
        raise Exception("Unspecified ratio estimation method!")
    print(f"Stats. Max: {w.max()}, Mean: {w.mean()}, Min: {w.min()}")
    w = torch.clamp(torch.tensor(w), max=max_weight)
    Xbar = torch.diag(torch.sqrt(w)) @ X_train
    ybar = torch.diag(torch.sqrt(w)) @ y_source
    theta, res, _, _ = torch.linalg.lstsq(Xbar, ybar)
    Xtest = torch.hstack((torch.ones(num_test)[:, torch.newaxis], x_target))
    return theta, Xtest @ theta, w
