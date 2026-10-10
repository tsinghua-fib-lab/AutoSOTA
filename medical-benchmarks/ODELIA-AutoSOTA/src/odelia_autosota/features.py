"""Extract the exact Stage 3/4/5 descriptors from frozen feature maps."""

import torch


@torch.no_grad()
def stage_descriptors(stage3, stage4, stage5):
    blocks = []
    for tensor, channels in zip((stage3, stage4, stage5), (256, 320, 320)):
        if tensor.ndim < 3 or tensor.shape[1] != channels:
            raise ValueError(
                "expected stage channels 256, 320, 320 and spatial dimensions"
            )
        flat = tensor.detach().float().flatten(2)
        blocks.extend((flat.mean(-1), flat.amax(-1), flat.std(-1, unbiased=False)))
    if len({block.shape[0] for block in blocks}) != 1:
        raise ValueError("stage batch sizes differ")
    return torch.cat(blocks, dim=-1)
