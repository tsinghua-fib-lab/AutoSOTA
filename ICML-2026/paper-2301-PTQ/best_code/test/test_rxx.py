import torch

if __name__ == "__main__":

    X = torch.randn(100, 10).to("cuda")  # 100 samples, 10 features

    Exx_manual = torch.zeros(10, 10, device=X.device)

    for x in X:
        Exx_manual += torch.ger(x, x)

    Exx_manual /= X.shape[0]

    Exx_einsum = torch.einsum("ij,ik->jk", X, X) / X.shape[0]

    print(torch.allclose(Exx_manual, Exx_einsum, atol=1e-6))
