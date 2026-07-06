from __future__ import annotations

import torch
import torch.nn.functional as F

from spikformer_fragmentation_addon import (
    Spikformer,
    build_spikformer_preset,
    FixedLearnableFragmenter,
    DynamicLearnableFragmenter,
    FragmentedSpikformer,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-style Spikformer following the official repo's lightweight setting.
    backbone = build_spikformer_preset(
        "spikformer-cifar",
        image_size=(32, 32),
        in_channels=3,
        num_classes=10,
        spike_backend="native",   # switch to 'auto' or 'spikingjelly' if desired
    ).to(device)

    # Dynamic learnable fragmentation from the uploaded ICML-style paper.
    fragmenter = DynamicLearnableFragmenter(
        image_size=(32, 32),
        candidates=(2, 4, 8),
        gumbel_tau=1.0,
        sharpness=1.0,
        straight_through=True,
        selector_init=4,
    ).to(device)

    model = FragmentedSpikformer(backbone, fragmenter, decode="entropy", gamma=1.0).to(device)
    model.train()

    images = torch.rand(8, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (8,), device=device)

    logits, aux = model(images, return_aux=True)
    loss = F.cross_entropy(logits, labels) + 0.01 * aux["balance_loss"]
    loss.backward()

    print("train logits:", logits.shape)
    print("balance loss:", float(aux["balance_loss"].detach().cpu()))
    print("selected steps (current argmax):", aux["selected_steps"])
    print("selector probs:", None if aux["selector_probs"] is None else aux["selector_probs"].detach().cpu())

    # Switch to eval: the dynamic fragmenter now uses the selected T directly.
    model.eval()
    with torch.no_grad():
        logits_eval, aux_eval = model(images, return_aux=True)
    print("eval logits:", logits_eval.shape)
    print("eval selected steps:", aux_eval["selected_steps"])


if __name__ == "__main__":
    main()
