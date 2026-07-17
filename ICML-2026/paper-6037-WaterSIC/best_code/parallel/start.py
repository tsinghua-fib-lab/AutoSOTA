import torch
import os
import sys
import json
from datetime import timedelta
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.args import ModelArgs
from parallel.model import Transformer
from llama.tokenizer import Tokenizer
from llama2.tokenizer import Tokenizer as Tokenizer2


def start(ckpt_dir, is_llama_2, qconfig, override_params={}, device="cuda"):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))

    # Ensure model_parallel_size is defined regardless of whether MP is already initialized.
    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)
    else:
        # If already initialized (e.g., start() called twice in same process),
        # fall back to WORLD_SIZE (which torchrun sets) or 1.
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
        # Keep stderr open so tracebacks from non-rank-0 are visible
        # sys.stderr = open(os.devnull, "w")

    torch.manual_seed(58)

    checkpoints = list(sorted(filter(lambda x: x.endswith(".pth"), os.listdir(ckpt_dir))))
    assert len(checkpoints) == model_parallel_size

    ckpt_path = os.path.join(ckpt_dir, checkpoints[get_model_parallel_rank()])
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    with open(os.path.join(ckpt_dir, "params.json")) as f:
        params = json.loads(f.read())
    for k, v in override_params.items():
        params[k] = v

    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.model")
    tokenizer_cls = Tokenizer2 if is_llama_2 else Tokenizer
    tokenizer = tokenizer_cls(tokenizer_path)

    model_args = ModelArgs(**params)
    print(f"[model] rope_theta={model_args.rope_theta}, use_scaled_rope={model_args.use_scaled_rope}, max_seq_len={model_args.max_seq_len}")
    torch.set_default_dtype(torch.bfloat16)

    if is_llama_2:
        state_dict.pop("rope.freqs")
        model_args.vocab_size = tokenizer.n_words

    model = Transformer(model_args, qconfig).to(device)
    model.load_state_dict(state_dict)

    return model, tokenizer
