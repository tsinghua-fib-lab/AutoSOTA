from .ulip_model import ULIP
from .text_encoder import TextEncoder


def create_clip_text_encoder(args):
    model = TextEncoder(args)
    return model


def create_ulip(args):
    model = ULIP(args)
    return model