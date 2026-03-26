import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    keys=[]
    for key, value in model.named_parameters():
        keys.append(key)
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':  # this setting is for the CLIP pre-trained models
            if not cfg.MODEL.FROZEN:
                if "base" in key:
                    if "adapter" not in key:
                        lr = 0.000005
        else:
            if not cfg.MODEL.FROZEN: # this setting is for the ImageNet pre-trained models
                if "base" in key:
                    if "adapter" not in key:
                        lr = cfg.SOLVER.BASE_LR*0.8

        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        if cfg.MODEL.ADD_SHARE is True:
            if "add_image_token" in key or "positional_embedding" in key:
                lr = cfg.SOLVER.BASE_LR

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
