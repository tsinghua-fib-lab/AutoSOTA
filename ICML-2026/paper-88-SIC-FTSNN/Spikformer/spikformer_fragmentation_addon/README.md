# Spikformer + Learnable Fragmentation Add-on

This package contains a clean PyTorch implementation of the **ICLR 2023 Spikformer** backbone and a paper-faithful **learnable fragmentation** add-on that can be attached in front of it.

## Files

- `spiking_core.py`
  - Pure-PyTorch multi-step LIF node with sigmoid surrogate.
  - Optional SpikingJelly wrapper (`spike_backend='spikingjelly'` or `'auto'`).
- `spikformer.py`
  - Spikformer backbone.
  - `forward(x)` for ordinary static-image SNN usage.
  - `forward_sequence(x_seq)` for externally constructed temporal sequences.
- `fragmentation.py`
  - `FixedLearnableFragmenter`
  - `DynamicLearnableFragmenter`
  - `entropy_weighted_decode`
- `wrapper.py`
  - `FragmentedSpikformer`, which plugs the fragmenter in front of the backbone.
- `example_usage.py`
  - End-to-end training/eval example on random tensors.

## Minimal usage

```python
import torch
import torch.nn.functional as F

from spikformer_fragmentation_addon import (
    build_spikformer_preset,
    DynamicLearnableFragmenter,
    FragmentedSpikformer,
)

backbone = build_spikformer_preset(
    'spikformer-cifar',
    image_size=(32, 32),
    in_channels=3,
    num_classes=10,
    spike_backend='native',   # 'auto' or 'spikingjelly' also works when available
)

fragmenter = DynamicLearnableFragmenter(
    image_size=(32, 32),
    candidates=(2, 4, 8),
    selector_init=4,
)

model = FragmentedSpikformer(backbone, fragmenter, decode='entropy', gamma=1.0)

images = torch.rand(16, 3, 32, 32)
labels = torch.randint(0, 10, (16,))

logits, aux = model(images, return_aux=True)
loss = F.cross_entropy(logits, labels) + 0.01 * aux['balance_loss']
loss.backward()
```

## Design choices

### Spikformer side

- Spike-form Q/K/V.
- No softmax in SSA.
- Fixed paper scale (`0.125` by default).
- BatchNorm replaces LayerNorm.
- Residual `SSA + MLP` blocks.
- Static images are repeated along time when no external sequence is provided.

### Fragmentation side

- Learnable division lines `(h_k, v_k, r_k)`.
- Bounded line reparameterization.
- Straight-through binary masks in the forward pass.
- Dynamic fragment-count selection with Gumbel-Softmax.
- Balance regularizer.
- Entropy-weighted temporal decoding utility.

## Notes

- The code is self-contained and runs without SpikingJelly.
- If you have SpikingJelly installed and want to use its neuron kernels, set `spike_backend='auto'` or `spike_backend='spikingjelly'` when building the backbone.
- The backbone accepts **variable T** in `forward_sequence`, so the dynamic fragmenter can train with `Tmax` and infer with the selected `T` directly.
