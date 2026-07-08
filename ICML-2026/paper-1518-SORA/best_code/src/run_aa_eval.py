import torch
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

sys.path.insert(0, "/repo/src")
from architectures.get_model import get_model
from autoattack import AutoAttack

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, -1, 1, 1))
    def forward(self, x):
        return (x - self.mean) / self.std

def main():
    root_path = "/repo/output"
    model_name = "PreActResNet18"
    dataset_name = "CIFAR10"
    attack_name = "SORA"
    seed = 42
    epsilon = 8.0 / 255.0
    batch_size = 256
    num_classes = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2471, 0.2435, 0.2616]

    # NO normalization in transform - Normalize wrapper handles it
    test_transform = transforms.Compose([transforms.ToTensor()])
    testset = CIFAR10(root=f"{root_path}/Datasets/{dataset_name}", train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = get_model(model_name, num_classes, 32, 3)
    model = nn.Sequential(Normalize(mean=cifar10_mean, std=cifar10_std), net).to(device)
    model.eval()

    ckpt_path = f"{root_path}/Results/{dataset_name}/{model_name}/{attack_name}/checkpoints_{seed}/model030.pt"
    print("Loading checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded successfully.")

    # Verify clean accuracy first
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Clean accuracy (verified): {100.0 * correct / total:.2f}%")

    all_x = []
    all_y = []
    for x, y in testloader:
        all_x.append(x)
        all_y.append(y)
    x_test = torch.cat(all_x, 0)
    y_test = torch.cat(all_y, 0)
    print("Test data:", x_test.shape, "labels:", y_test.shape)

    log_path = f"{root_path}/Results/{dataset_name}/{model_name}/{attack_name}/raw_results_{seed}/aa_log_v2.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"Running AutoAttack L-inf with eps={epsilon}")
    adversary = AutoAttack(model, norm="Linf", eps=epsilon, log_path=log_path)
    adversary.fab.n_target_classes = min(num_classes - 1, adversary.fab.n_target_classes)
    adversary.apgd_targeted.n_target_classes = min(num_classes - 1, adversary.apgd_targeted.n_target_classes)

    adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    print("AutoAttack complete.")
    clean_acc = adv_complete.get("clean_acc", "N/A")
    robust_acc = adv_complete.get("robust_acc", "N/A")
    print("Clean accuracy:", clean_acc)
    print("Robust accuracy (AA):", robust_acc)

if __name__ == "__main__":
    main()
