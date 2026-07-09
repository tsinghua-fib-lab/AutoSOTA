import logging, sys
from pathlib import Path

import torch
import transformers
from accelerate import dispatch_model
from functools import partial


src_path = Path(__file__).parents[1].joinpath("src")
assert src_path.exists(), f"Path does not exist: {src_path}"
sys.path.append(src_path.as_posix())


from qera.quantize.quantizers import get_quantizer
from qera.utils import create_device_map

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    qera_dtype = getattr(torch, "float32")
    device_map = "auto-balanced"

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=qera_dtype, _attn_implementation="eager"
    )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_device_map(model, device_map=device_map)
    logger.info(f"Device map: {device_map}")
    model = dispatch_model(model, device_map)
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    ori_k_proj = model.model.layers[0].self_attn.k_proj.weight.clone()

    normal_float_config = {"num_bits": 4, "block_size": 64}
    mxint_config = {"width": 4, "block_size": 32, "block_axis": -1}

    w_mxint_quantizer = partial(get_quantizer("mxint"), **mxint_config)
    w_normalfloat_quantizer = partial(
        get_quantizer("normalfloat"), **normal_float_config
    )

    mxint_result = w_mxint_quantizer(ori_k_proj)
    normalfloat_result = w_normalfloat_quantizer(ori_k_proj)

    print(f"Original weight: {ori_k_proj}")
    print(f"MXInt quantized weight: {mxint_result}")
    print(f"NormalFloat quantized weight: {normalfloat_result}")

    # quantization error for both quantizers
    error_mxint = torch.abs(ori_k_proj - w_mxint_quantizer(normalfloat_result))
    error_normalfloat = torch.abs(ori_k_proj - w_normalfloat_quantizer(mxint_result))

    print(f"MXInt quantization error: {error_mxint.mean()}")
    print(f"NormalFloat quantization error: {error_normalfloat.mean()}")
