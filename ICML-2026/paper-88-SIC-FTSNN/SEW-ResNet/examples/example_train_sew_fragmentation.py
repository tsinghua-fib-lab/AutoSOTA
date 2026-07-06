"""Example: training a SEW-ResNet with dynamic learnable fragmentation.

This is intentionally compact. It shows the exact insertion point of the fragmentation add-on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from sew_resnet_fragmentation_wrapper import build_fragmented_sew_resnet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_fragmented_sew_resnet(
        depth=18,
        num_classes=10,
        image_size=(32, 32),
        in_channels=3,
        stem="cifar",
        cnf="ADD",
        neuron_name="if",
        zero_init_residual=True,
        dynamic_candidates=(2, 4, 8),
        init_direction="horizontal",
        mask_scale=1.0,
        decoder="entropy",
        entropy_gamma=1.0,
        use_expected_poisson=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy minibatch
    images = torch.rand(8, 3, 32, 32, device=device)
    targets = torch.randint(0, 10, (8,), device=device)

    model.train()
    out = model(images, return_aux=True)

    main_loss = F.cross_entropy(out.logits, targets)
    loss = main_loss + 0.01 * out.fragmentation.balance_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("logits shape:", tuple(out.logits.shape))
    print("step_logits shape:", tuple(out.step_logits.shape))
    print("selected_t:", out.fragmentation.selected_t)
    if out.fragmentation.selector_probs is not None:
        print("selector_probs:", out.fragmentation.selector_probs.detach().cpu())
    print("balance_loss:", float(out.fragmentation.balance_loss.detach().cpu()))


if __name__ == "__main__":
    main()
