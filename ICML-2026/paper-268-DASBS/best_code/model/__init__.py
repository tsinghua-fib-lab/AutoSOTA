from .ema import EMA
from .vit_rope import RopeViT, get_rope_vit_model
from .sit import SiT, get_sit_model
from .sit_rope import RopeSiT, get_rope_sit_model
from .controller_wrapper import ControllerWrapper

def get_model(args, require_time=True):
    if args.model.name == 'ropevit':
        return get_rope_vit_model(args, require_time=require_time)
    elif args.model.name == 'sit':
        return get_sit_model(args, require_time=require_time)
    elif args.model.name == 'ropesit':
        return get_rope_sit_model(args, require_time=require_time)
    else:
        raise ValueError(f"Unknown model name: {args.model.name}")