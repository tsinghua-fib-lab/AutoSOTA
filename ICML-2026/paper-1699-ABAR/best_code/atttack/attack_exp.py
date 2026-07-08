"""
Streaming DP-prefix-sum toy + adversary, with:
1) accuracy curve vs round
2) sweep over different tree noise multipliers z

Assumes: PyTorch >= 2.0, torchvision installed & working.

If your environment throws a torchvision error (e.g., "torchvision::nms does not exist"),
install matching torch/torchvision versions, e.g.:
  pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision
(or cpu wheels if needed).
"""

import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# PyTorch 2.0+
from torch.func import functional_call, vmap, grad

# torchvision for MNIST
from torchvision import datasets, transforms


# ---------------------------
# Binary-tree cumulative noise (safe return + on-device sampling)
# ---------------------------
class CummuNoiseTorch:
    @torch.no_grad()
    def __init__(self, std: float, shapes: List[torch.Size], device: torch.device, test_mode: bool = False):
        assert std >= 0
        self.std = float(std)
        self.shapes = list(shapes)
        self.device = device
        self.step = 0
        self.binary = [0]
        self.noise_sum = [torch.zeros(s, device=self.device) for s in self.shapes]
        self.recorded = [[torch.zeros(s, device=self.device) for s in self.shapes]]
        self.test_mode = test_mode

    @torch.no_grad()
    def __call__(self) -> List[torch.Tensor]:
        self.step += 1
        if self.std <= 0 and not self.test_mode:
            return [ns.clone() for ns in self.noise_sum]

        idx = 0
        while idx < len(self.binary) and self.binary[idx] == 1:
            self.binary[idx] = 0
            for ns, re in zip(self.noise_sum, self.recorded[idx]):
                ns.sub_(re)
            idx += 1

        if idx >= len(self.binary):
            self.binary.append(0)
            self.recorded.append([torch.zeros(s, device=self.device) for s in self.shapes])

        for s, ns, re in zip(self.shapes, self.noise_sum, self.recorded[idx]):
            if self.test_mode:
                n = torch.ones(s, device=self.device)
            else:
                n = torch.randn(s, device=self.device) * self.std
            ns.add_(n)
            re.copy_(n)

        self.binary[idx] = 1
        return [ns.clone() for ns in self.noise_sum]


# ---------------------------
# Simple binary MLP
# ---------------------------
class MLPBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------
# Stream construction (MNIST 0/1)
# ---------------------------
def build_streams_from_mnist01(
    mnist_dataset,
    batch_size: int,
    rounds: int,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    Batches indexed from 0.

    Stream A:
      - all 0s, except the first element of each odd batch index (1,3,5,...) is a 1.
    Stream B:
      - delete the first element of A (shift left), append a 0 at the end.
    """
    rng = random.Random(seed)
    zeros, ones = [], []
    for idx in range(len(mnist_dataset)):
        y = int(mnist_dataset[idx][1])
        if y == 0:
            zeros.append(idx)
        elif y == 1:
            ones.append(idx)

    rng.shuffle(zeros)
    rng.shuffle(ones)

    N = batch_size * rounds
    need_ones = sum(1 for t in range(rounds) if (t % 2 == 1))  # batches 1,3,5,... have a leading 1
    need_zeros = N - need_ones + 1  # +1 for append in B

    if len(ones) < need_ones or len(zeros) < need_zeros:
        raise RuntimeError(
            f"Not enough MNIST 0/1: need ones={need_ones}, zeros={need_zeros}; "
            f"have ones={len(ones)}, zeros={len(zeros)}"
        )

    ones_ptr, zeros_ptr = 0, 0
    stream_A = []
    for t in range(rounds):
        for j in range(batch_size):
            if (j == 0) and (t % 2 == 1):
                stream_A.append(ones[ones_ptr]); ones_ptr += 1
            else:
                stream_A.append(zeros[zeros_ptr]); zeros_ptr += 1

    stream_B = stream_A[1:] + [zeros[zeros_ptr]]
    return stream_A, stream_B


def get_batch_indices(stream: List[int], batch_size: int, t: int) -> List[int]:
    return stream[t * batch_size : (t + 1) * batch_size]


# ---------------------------
# Per-example clipped gradient (torch.func)
# ---------------------------
def per_example_clipped_batch_grad(
    model: nn.Module,
    params_dict: Dict[str, torch.Tensor],
    buffers_dict: Dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    clip_C: float,
) -> Dict[str, torch.Tensor]:
    def loss_one(params, buffers, x1, y1):
        logits = functional_call(model, (params, buffers), (x1.unsqueeze(0),))
        return F.cross_entropy(logits, y1.unsqueeze(0))

    grad_fn = grad(loss_one)
    grads_batched = vmap(grad_fn, in_dims=(None, None, 0, 0))(params_dict, buffers_dict, x, y)

    B = x.shape[0]
    sq = None
    for g in grads_batched.values():
        g2 = (g.reshape(B, -1) ** 2).sum(dim=1)
        sq = g2 if sq is None else (sq + g2)
    norms = torch.sqrt(sq + 1e-12)

    scale = (clip_C / norms).clamp(max=1.0)  # [B]
    clipped_mean = {}
    for k, g in grads_batched.items():
        view = [B] + [1] * (g.ndim - 1)
        clipped = g * scale.view(*view)
        clipped_mean[k] = clipped.mean(dim=0)

    return clipped_mean


def l2_dist_list(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    s = 0.0
    for ta, tb in zip(a, b):
        diff = ta - tb
        s += float((diff * diff).sum().item())
    return math.sqrt(s)


# ---------------------------
# Streaming experiment
# ---------------------------
@dataclass
class ExpConfig:
    batch_size: int = 128
    rounds: int = 100
    clip_C: float = 1.0
    noise_multiplier_z: float = 1.0  # tree std (grad units) = z * C / sqrt(B)
    lr: float = 0.1                  # ONLY used in parameter update (separate from tree)
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run_streaming_one(
    cfg: ExpConfig,
    mnist,
    stream_A: List[int],
    stream_B: List[int],
    true_stream: str = "A",
) -> Tuple[List[int], List[float]]:
    """
    Returns:
      test_rounds: list of t where we tested (0,2,4,...)
      running_majority_acc: accuracy curve (same length)
    """
    device = torch.device(cfg.device)
    stream_true = stream_A if true_stream == "A" else stream_B

    model = MLPBinary().to(device)
    model.eval()

    param_names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]

    def get_params_dict():
        return {n: p for n, p in model.named_parameters()}

    def get_buffers_dict():
        return {n: b for n, b in model.named_buffers()}

    theta0 = {n: p.detach().clone() for n, p in get_params_dict().items()}

    # Prefix sums in GRADIENT space
    prefix = {n: torch.zeros_like(theta0[n], device=device) for n in param_names}
    prev_noisy_prefix = {n: torch.zeros_like(theta0[n], device=device) for n in param_names}

    # Tree noise std in GRADIENT units
    noise_std = cfg.noise_multiplier_z * cfg.clip_C / (cfg.batch_size)
    noise_gen = CummuNoiseTorch(std=noise_std, shapes=shapes, device=device, test_mode=False)

    votes: List[str] = []
    is_guess_correct: List[bool] = []
    test_rounds: List[int] = []



    for t in range(cfg.rounds):
        do_test = (t % 2 == 0)
        params_dict = get_params_dict()
        buffers_dict = get_buffers_dict()

        # --- adversary candidates at START of round t
        gA_list = gB_list = None
        if do_test:
            idxA = get_batch_indices(stream_A, cfg.batch_size, t)
            idxB = get_batch_indices(stream_B, cfg.batch_size, t)

            xA = torch.stack([mnist[i][0] for i in idxA]).to(device)
            yA = torch.tensor([int(mnist[i][1]) for i in idxA], device=device, dtype=torch.long)

            xB = torch.stack([mnist[i][0] for i in idxB]).to(device)
            yB = torch.tensor([int(mnist[i][1]) for i in idxB], device=device, dtype=torch.long)

            gA = per_example_clipped_batch_grad(model, params_dict, buffers_dict, xA, yA, cfg.clip_C)
            gB = per_example_clipped_batch_grad(model, params_dict, buffers_dict, xB, yB, cfg.clip_C)

            gA_list = [gA[n].detach() for n in param_names]
            gB_list = [gB[n].detach() for n in param_names]

        # --- mechanism processes TRUE batch at round t
        idxT = get_batch_indices(stream_true, cfg.batch_size, t)
        xT = torch.stack([mnist[i][0] for i in idxT]).to(device)
        yT = torch.tensor([int(mnist[i][1]) for i in idxT], device=device, dtype=torch.long)

        gT = per_example_clipped_batch_grad(model, params_dict, buffers_dict, xT, yT, cfg.clip_C)
        for n in param_names:
            prefix[n].add_(gT[n].detach())

        noise_list = noise_gen()
        noisy_prefix = {n: prefix[n] + noise_t for n, noise_t in zip(param_names, noise_list)}

        # observed noisy increment (gradient units)
        delta_noisy = [(noisy_prefix[n] - prev_noisy_prefix[n]).detach() for n in param_names]

        # model update (lr applied here, separate from tree)
        for n, p in model.named_parameters():
            p.copy_(theta0[n] - cfg.lr * noisy_prefix[n])

        for n in param_names:
            prev_noisy_prefix[n].copy_(noisy_prefix[n])

        # --- adversary vote + running majority acc
        if do_test:
            # 1. Calculate distances for THIS step
            dist_t_A = l2_dist_list(delta_noisy, gA_list)
            dist_t_B = l2_dist_list(delta_noisy, gB_list)

            # 2. Accumulate the squared distances (Log-Likelihood accumulation)
            # We use squared distance because for Gaussian noise, 
            # maximizing likelihood == minimizing sum of squared errors.

            # 3. Make the decision based on CUMULATIVE history
            # If total error for A is lower, we guess A.
            current_guess = "A" if dist_t_A <= dist_t_B else "B"

            votes.append(current_guess)

            major_guess = max(set(votes), key=votes.count)
            
            # 4. Check if the guess at this specific round t is correct
            is_guess_correct.append( (major_guess == true_stream))

            
            
            test_rounds.append(t)

    return test_rounds, is_guess_correct


def sweep_noise_levels_and_plot(
    z_list=(0.25, 0.5, 1.0, 2.0, 4.0),
    seed=0,
    trials=100,
    true_stream="A",
    batch_size=128,
    rounds=100,
    clip_C=1.0,
    lr=0.1,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'running device: {device}')

    # Load MNIST once
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Each (z, seed) gives an accuracy curve; we average over seeds for each z
    curves_mean = {}  # z -> (test_rounds, mean_acc)
    curves_all = {}   # z -> list of acc arrays

    
    for z in z_list:
        agg_correct_list = []
        trs_ref = None
        torch.manual_seed(seed)
        random.seed(seed)
        
        for trial in tqdm(range(trials), desc=f'Noise z={z}'):
            

            stream_A, stream_B = build_streams_from_mnist01(mnist, batch_size, rounds, seed=seed)

            cfg = ExpConfig(
                batch_size=batch_size,
                rounds=rounds,
                clip_C=clip_C,
                noise_multiplier_z=float(z),
                lr=lr,
                seed=seed,
                device=device,
            )
            trs, correct_list = run_streaming_one(cfg, mnist, stream_A, stream_B, true_stream=true_stream)
            if trs_ref is None:
                trs_ref = trs
            else:
                assert trs_ref == trs, "Test-round indices mismatch (should not happen)."
            agg_correct_list.append(correct_list)

        curves_all[z] = agg_correct_list
        #average over trials for each test round
        mean_acc = []
        for i in range(len(trs_ref)):
            acc = sum(agg_correct_list[trial][i] for trial in range(trials)) / trials
            mean_acc.append(acc)
        curves_mean[z] = (trs_ref, mean_acc)

    # Plot mean curves
    plt.figure()
    for z in z_list:
        trs, mean_acc = curves_mean[z]
        plt.plot(trs, mean_acc, label=f"z={z}")
    plt.xlabel("Round t (test rounds: 0,2,4,...)")
    plt.ylabel("Running majority accuracy")
    plt.title("Adversary accuracy vs round (mean over seeds)")
    plt.grid(True)
    plt.legend()
    #save the figure
    os.makedirs("./figures", exist_ok=True)
    plt.savefig("./figures/streaming_dp_prefix_sum_toy_exp.pdf")
    plt.savefig("./figures/streaming_dp_prefix_sum_toy_exp.png")
    plt.show()

    #also save the data
    import pickle
    with open("./figures/streaming_dp_prefix_sum_toy_exp_data.pkl", "wb") as f:
        pickle.dump(curves_mean, f) 


    # Also return raw data in case you want to save it
    return curves_mean, curves_all


if __name__ == "__main__":
    # Your defaults
    curves_mean, curves_all = sweep_noise_levels_and_plot(
        z_list=([0.1,0.5,1,2,4,8,16,32,64]),
        seed=0,
        trials=100,
        true_stream="A",
        batch_size=32,
        rounds=185,
        clip_C=1.0,
        lr=0.1,
    )

    # 
