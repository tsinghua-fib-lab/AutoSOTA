from models import (
        HybridNoPEForCausalLM,
        HybridForCausalLM,
        HybridNoPEConfig,
        HybridConfig
    )

def get_model(args, tokenizer):
    if not args.nope:
        config = HybridConfig(use_cache=False, 
            layers=args.layers,
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            d_model=args.hidden_size,
            ssm_cfg={"d_state": args.state_dim},
            vocab_size=len(tokenizer),
        )
    if args.nope:
        config = HybridNoPEConfig(use_cache=False, 
            layers=args.layers,
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            d_model=args.hidden_size,
            ssm_cfg={"d_state": args.state_dim},
            vocab_size=len(tokenizer),
        )

    if not args.nope:
        model = HybridForCausalLM(config)
    if args.nope:
        model = HybridNoPEForCausalLM(config)
        
    return model


