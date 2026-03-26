from .clip_models import ClipModel


VALID_NAMES = {
    'CLIP:ViT-B/16_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/openai/clip-vit-base-patch16/',
    'CLIP:ViT-B/32_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/openai/clip-vit-base-patch32/',
    'CLIP:ViT-L/14_svd':'/tmp/clip_vitl14', 
    'SigLIP:ViT-L/16_256_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/huggingface/google/siglip-large-patch16-256/',
    'BEiTv2:ViT-L/16_svd':'/Youtu_Pangu_Security_Public/youtu-pangu-public/jeremiewang/pretrained_model/BEiT-v2/beitv2_large_patch16_224_pt1k_ft21k.pth',
}


def get_model(name, opt):
    assert name in VALID_NAMES.keys()
    if name.startswith("CLIP:"):
        return ClipModel(VALID_NAMES[name], opt)
    else:
        assert False 
