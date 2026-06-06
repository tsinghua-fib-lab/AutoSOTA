from .dino_v2 import DinoVisionTransformer
from .reins_dinov2 import ReinsDinoVisionTransformer
try:
    from .reins_eva_02 import ReinsEVA2
except (ImportError, ModuleNotFoundError) as e:
    print(f'Warning: Failed to import ReinsEVA2: {e}')
from .reins_resnet import ReinsResNetV1c
try:
    from .reins_convnext import ReinsConvNeXt
except:
    print('Fail to import ReinsConvNeXt, if you need to use it, please install mmpretrain')
from .clip import CLIPVisionTransformer
from .reins_sam_vit import ReinsSAMViT
from .sam_vit import SAMViT
from .reins_clip import ReinsCLIPVisionTransformer
from .synclr import SynclrVisionTransformer
