import random
import time
import numpy as np
import torch
import sys
sys.path.insert(0, "/repo")
from libs import OnesidedStreamSW, SW
import ot

def compute_true_Wasserstein(X, Y, p=2):
    M = ot.dist(X.cpu().detach().numpy(), Y.cpu().detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)

def main():
    for _ in range(100):
        torch.randn(100)
        np.random.rand(100)

    A = np.load("reconstruct_random_50_shapenetcore55.npy")
    ind1, ind2 = 30, 31
    target = A[ind2]
    source = A[ind1]

    device = "cuda"
    learning_rate = 0.001
    N_step = 5000
    L = 100
    k = 100
    seeds = [1, 2, 3, 4, 5]

    Y = torch.from_numpy(target).to(device)
    N = target.shape[0]

    Z = (1 - (2 * torch.arange(1, L + 1) - 1) / L).view(-1, 1)
    theta1 = torch.arccos(Z)
    theta2 = torch.remainder(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    theta = torch.cat(
        [torch.sin(theta1) * torch.cos(theta2),
         torch.sin(theta1) * torch.sin(theta2),
         torch.cos(theta1)],
        dim=1).to(device)

    results = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        streamsw = OnesidedStreamSW(L=L, d=3, k=k, p=2, c=1.0,
                                     thetas=theta.detach().cpu().numpy())
        streamsw.update(Y.cpu().detach().numpy())

        X = torch.tensor(source, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([X], lr=learning_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.003, total_steps=N_step,
            pct_start=0.3, anneal_strategy="cos")

        for i in range(N_step):
            optimizer.zero_grad()
            a = np.ones(X.shape[0]) / X.shape[0]
            sw = streamsw.compute_distance_torch(X, a)
            loss = N * sw
            loss.backward()
            torch.nn.utils.clip_grad_norm_([X], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if i % 100 == 99 or i == 0:
                with torch.no_grad():
                    w2_check = compute_true_Wasserstein(X.detach(), Y)
                    print("  [step %5d] W2^2*1000 = %.4f" % (i+1, w2_check * 1000))

        final_distance = compute_true_Wasserstein(X, Y)
        scaled_distance = final_distance * 1000
        results.append(scaled_distance)
        print("StreamSW L=%d k=%d seed=%d step=%d: W2^2*1000 = %.4f" % (
            L, k, seed, N_step, scaled_distance))

    mean_val = np.mean(results)
    std_val = np.std(results)
    seed_strs = ["%.4f" % v for v in results]
    print("")
    print("=== FINAL RESULT ===")
    print("StreamSW L=%d k=%d W2^2*1000 mean: %.4f" % (L, k, mean_val))
    print("StreamSW L=%d k=%d W2^2*1000 std:  %.4f" % (L, k, std_val))
    print("Individual seeds: %s" % seed_strs)
    print("Paper reference: 1.93 +/- 0.03")
    print("Reproduction CI: [1.90, 1.96]")
    within_ci = 1.90 <= mean_val <= 1.96
    print("Within CI bounds: %s" % within_ci)

if __name__ == "__main__":
    main()
