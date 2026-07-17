import torch
import transformers
from foam_torch import GaLoreAdamW, Muon, FOAM, Adam_mini
import bitsandbytes as bnb

def configure_optimizer(args, logger, model, model_config):

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_total_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in trainable_params)
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {num_total_params / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {num_trainable_params / 1_000_000:.2f}M")

    def get_optimizer_params(model):
        return [
            p for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ], [
            p for name, p in model.named_parameters()
            if not (p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name)
        ]
    
    if args.optimizer.lower() == "adamw" or args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

    elif args.optimizer.lower() == "galore_adamw":
        scale_params, base_params = get_optimizer_params(model)
        param_groups = [{'params': base_params}, 
                        {'params': scale_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.scale, 'proj_type': args.proj_type}]
        
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    elif args.optimizer.lower() == "foam":
        scale_params, base_params = get_optimizer_params(model)
        param_groups = [{'params': base_params}, 
                        {'params': scale_params, 'scale': args.scale, 'level': args.level}]

        if 2 ** args.level > model_config.hidden_size:
            logger.info('Using compress level larger than model dimension, will adopt 0 padding')
        optimizer = FOAM(param_groups, lr=args.lr, betas=(args.beta1, args.beta2), \
                               weight_decay=args.weight_decay, res_scale=args.res_scale)
        
    elif args.optimizer.lower() == "adam_mini":
        optimizer = Adam_mini(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
                               dim=model_config.hidden_size, n_heads=model_config.num_attention_heads)
        
    elif args.optimizer.lower() == "muon":
        scale_params, base_params = get_optimizer_params(model)
        logger.info(f"Total params with Muon enabled: {sum(p.numel() for p in scale_params) / 1_000_000:.2f}M")
        
        optimizer = Muon(
            lr=args.lr,
            ns_steps=args.n_steps,
            wd=args.weight_decay,
            muon_params=scale_params,
            adamw_params=base_params,
        )

    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return optimizer