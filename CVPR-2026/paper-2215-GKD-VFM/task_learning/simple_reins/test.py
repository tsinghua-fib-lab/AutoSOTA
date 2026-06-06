from .reins_clip import get_std_reins_clip_large
from .reins_dinov2 import get_std_reins_dinov2_large
import torch


def test_clip():
    # you can get checkpoint by:
    # wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
    checkpoint_file = "ViT-L-14.pt"
    model = get_std_reins_clip_large(checkpoint_file)
    model.train(True)
    random_image = torch.randn([2, 3, 512, 512])
    result = model.forward(random_image)
    for level, feats in enumerate(result):
        print(f"level {level} has shape {feats.shape}")


def test_dinov2():
    """you can get checkpoint by:
    wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
    python tools/convert_weight/convert_dinov2.py dinov2_vitl14_pretrain.pth dinov2_vitl14_512x512.pth
    """
    checkpoint_file = "dinov2_vitl14_512x512.pth"
    model = get_std_reins_dinov2_large()
    model.load_state_dict(torch.load(checkpoint_file, "CPU"),strict=False)
    model.train(True)
    random_image = torch.randn([2, 3, 512, 512])
    result = model.forward(random_image)
    for level, feats in enumerate(result):
        print(f"level {level} has shape {feats.shape}")


if __name__ == "__main__":
    test_clip()
    test_dinov2()
