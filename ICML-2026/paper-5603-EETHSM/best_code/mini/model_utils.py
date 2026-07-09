import os
from models import (
        LSTM,
        GPTNeoXAlibiForCausalLM,
        GPTNeoXHardAlibiForCausalLM,
        GPTNeoXNoPEForCausalLM,
        GPTNeoXForCausalLM,
        MambaForCausalLM,
        HybridNoPEForCausalLM,
        HybridForCausalLM,
        HybridNoPEConfig,
        HybridConfig
    )

from transformers import GPTNeoXConfig
from transformers import MambaConfig
from transformers import AutoModelForCausalLM

def get_model(args, tokenizer):
    if args.model in ["T_nope","T_rope","T_alibi"]:
        config = GPTNeoXConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            vocab_size=len(tokenizer),
        )
    elif args.model == "T_hard_alibi":
        config = GPTNeoXConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            num_masked_heads=args.num_masked_heads,
            vocab_size=len(tokenizer),
        )
    elif args.model == "mamba":
        config = MambaConfig(
            hidden_size=args.hidden_size,
            d_model=args.hidden_size,
            n_layer=args.layers,
            ssm_cfg={"d_state": args.state_dim},
            vocab_size=len(tokenizer),
        )
    elif args.model == "hybrid":
        config = HybridConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            d_model=args.hidden_size,
            n_layer=args.layers,
            ssm_cfg={"d_state": args.state_dim},
            vocab_size=len(tokenizer),
        )
    elif args.model == "hybrid_nope":
        config = HybridNoPEConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            d_model=args.hidden_size,
            n_layer=args.layers,
            ssm_cfg={"d_state": args.state_dim},
            vocab_size=len(tokenizer),
        )


    
    if args.model=="T_rope":
        model = GPTNeoXForCausalLM(config)
    elif args.model=="T_nope":
        model = GPTNeoXNoPEForCausalLM(config)
    elif args.model=="T_alibi":
        model = GPTNeoXAlibiForCausalLM(config)
    elif args.model=="T_hard_alibi":
        model = GPTNeoXHardAlibiForCausalLM(config)
    elif args.model=="mamba":
        model = MambaForCausalLM(config)
    elif args.model=="lstm":
        model = LSTM(
            embedding_dim=args.hidden_size,
            vocab_size=len(tokenizer),
            num_layers=args.layers,
            dropout_rate=0.65
        )
    elif args.model=="hybrid":
        model = HybridForCausalLM(config)
    elif args.model=="hybrid_nope":
        model = HybridNoPEForCausalLM(config)

    if args.model == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(args.pretrain_model)
        
    return model


