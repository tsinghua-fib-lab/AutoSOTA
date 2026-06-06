from .reins import LoRAReins
from .dino_v2 import DINOv2
from .peft import set_requires_grad, set_train, get_pyramid_feature


class ReinsDINOv2(DINOv2):
    def __init__(
        self,
        reins_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins: LoRAReins = LoRAReins(**reins_config)

    def forward(self, x):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, None)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return get_pyramid_feature(outs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])


def get_std_reins_dinov2_large():
    reins_config = dict(
        token_length=100,
        embed_dims=1024,
        num_layers=24,
        patch_size=16,
        lora_dim=16,
    )
    return ReinsDINOv2(
        reins_config=reins_config,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        img_size=512,
        ffn_layer="mlp",
        init_values=1.0e-5,
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
