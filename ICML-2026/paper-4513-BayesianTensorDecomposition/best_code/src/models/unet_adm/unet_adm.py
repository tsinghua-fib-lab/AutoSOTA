import torch
from .unet import UNetModel

NUM_CLASSES = 1000

def create_unet_adm(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    model_path='',
    grayscale=False
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    if isinstance(attention_resolutions, int):
        attention_ds.append(image_size // attention_resolutions)
    elif isinstance(attention_resolutions, str):
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
    else:
        raise NotImplementedError
    
    in_channels = 1 if grayscale else 3
    out_channels = 2*in_channels if learn_sigma else in_channels

    model= UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f"Got exception: {e} / Randomly initialize")
    return model
