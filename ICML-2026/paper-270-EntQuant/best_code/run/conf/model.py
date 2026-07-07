from typing import Any

from hydra_zen import store, ZenField
from omegaconf import MISSING
from transformers import AutoTokenizer, GenerationConfig, LlamaTokenizer

from entquant.model.tokenizer import load_tokenizer
from run.hydra_zen import fbuilds, just, make_config

model_cfg = make_config(
    identifier=ZenField(str, MISSING),
    base_model_name_or_path=ZenField(str, MISSING),
    ctx_length=ZenField(int, 2048),
    device_map=ZenField(str, "cuda"),
    dtype=ZenField(str, "bfloat16"),
    model_cls=ZenField(Any, None),
    model_kwargs=ZenField(dict[str, Any], {}),
    generation_config=ZenField(Any, None),
    tokenizer_cls=ZenField(Any, just(AutoTokenizer)),
    tokenizer_kwargs=ZenField(dict[str, Any], {"use_fast": True, "padding_side": "left"}),
    hydra_convert="all",
)

# -------- Default Test Model --------

default = model_cfg(
    identifier="default",
    base_model_name_or_path="Qwen/Qwen3-0.6B",
)
store(default, group="cfg/model", name="default")

# -------- LLaMA 1 --------
# NOTE: jeffwan repos only provide pytorch_model.bin (no safetensors).
# Run scripts/convert_to_safetensors.py once to convert them in the HF cache.

LLAMA1_KWARGS = dict(
    tokenizer_cls=just(LlamaTokenizer),
    tokenizer_kwargs={"padding_side": "left"},
)
llama1_7b = model_cfg(
    identifier="llama1_7b",
    base_model_name_or_path="jeffwan/llama-7b-hf",
    **LLAMA1_KWARGS,
)
llama1_13b = model_cfg(
    identifier="llama1_13b",
    base_model_name_or_path="jeffwan/llama-13b-hf",
    **LLAMA1_KWARGS,
)
llama1_30b = model_cfg(
    identifier="llama1_30b",
    base_model_name_or_path="jeffwan/llama-30b-hf",
    **LLAMA1_KWARGS,
)
store(llama1_7b, group="cfg/model", name="llama1_7b")
store(llama1_13b, group="cfg/model", name="llama1_13b")
store(llama1_30b, group="cfg/model", name="llama1_30b")

# -------- LLaMA 2 --------

llama2_7b = model_cfg(
    identifier="llama2_7b",
    base_model_name_or_path="/models/Llama-2-7b-hf",
)
llama2_7b_instr = model_cfg(
    identifier="llama2_7b_instr",
    base_model_name_or_path="NousResearch/Llama-2-7b-chat-hf",
)
llama2_13b = model_cfg(
    identifier="llama2_13b",
    base_model_name_or_path="NousResearch/Llama-2-13b-hf",
)
llama2_13b_instr = model_cfg(
    identifier="llama2_13b_instr",
    base_model_name_or_path="NousResearch/Llama-2-13b-chat-hf",
)
llama2_70b = model_cfg(
    identifier="llama2_70b",
    base_model_name_or_path="NousResearch/Llama-2-70b-hf",
)
llama2_70b_instr = model_cfg(
    identifier="llama2_70b_instr",
    base_model_name_or_path="NousResearch/Llama-2-70b-chat-hf",
)
store(llama2_7b, group="cfg/model", name="llama2_7b")
store(llama2_7b_instr, group="cfg/model", name="llama2_7b_instr")
store(llama2_13b, group="cfg/model", name="llama2_13b")
store(llama2_13b_instr, group="cfg/model", name="llama2_13b_instr")
store(llama2_70b, group="cfg/model", name="llama2_70b")
store(llama2_70b_instr, group="cfg/model", name="llama2_70b_instr")

# -------- LLaMA 3.1 --------

LLAMA31_KWARGS = dict(ctx_length=4096)
llama31_8b = model_cfg(
    identifier="llama31_8b",
    base_model_name_or_path="meta-llama/Llama-3.1-8B",
    **LLAMA31_KWARGS,
)
llama31_8b_instr = model_cfg(
    identifier="llama31_8b_instr",
    base_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    **LLAMA31_KWARGS,
)
llama31_70b = model_cfg(
    identifier="llama31_70b",
    base_model_name_or_path="meta-llama/Llama-3.1-70B",
    **LLAMA31_KWARGS,
)
llama31_70b_instr = model_cfg(
    identifier="llama31_70b_instr",
    base_model_name_or_path="meta-llama/Llama-3.1-70B-Instruct",
    **LLAMA31_KWARGS,
)
llama31_405b = model_cfg(
    identifier="llama31_405b",
    base_model_name_or_path="meta-llama/Llama-3.1-405B",
    **LLAMA31_KWARGS,
)
llama31_405b_instr = model_cfg(
    identifier="llama31_405b_instr",
    base_model_name_or_path="meta-llama/Llama-3.1-405B-Instruct",
    **LLAMA31_KWARGS,
)
store(llama31_8b, group="cfg/model", name="llama31_8b")
store(llama31_8b_instr, group="cfg/model", name="llama31_8b_instr")
store(llama31_70b, group="cfg/model", name="llama31_70b")
store(llama31_70b_instr, group="cfg/model", name="llama31_70b_instr")
store(llama31_405b, group="cfg/model", name="llama31_405b")
store(llama31_405b_instr, group="cfg/model", name="llama31_405b_instr")

# -------- LLaMA 3.3 --------

llama33_70b_instr = model_cfg(
    identifier="llama33_70b_instr",
    base_model_name_or_path="meta-llama/Llama-3.3-70B-Instruct",
    ctx_length=4096,
)
store(llama33_70b_instr, group="cfg/model", name="llama33_70b_instr")

# -------- Qwen 3 --------

qwen3_8b = model_cfg(
    identifier="qwen3_8b",
    base_model_name_or_path="Qwen/Qwen3-8B",
)
qwen3_14b = model_cfg(
    identifier="qwen3_14b",
    base_model_name_or_path="Qwen/Qwen3-14B",
)
qwen3_32b = model_cfg(
    identifier="qwen3_32b",
    base_model_name_or_path="Qwen/Qwen3-32B",
)
qwen3_30b_a3b_instr = model_cfg(
    identifier="qwen3_30b_a3b_instr",
    base_model_name_or_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
)
qwen3_235b_a22b_instr = model_cfg(
    identifier="qwen3_235b_a22b_instr",
    base_model_name_or_path="Qwen/Qwen3-235B-A22B-Instruct-2507",
)
store(qwen3_8b, group="cfg/model", name="qwen3_8b")
store(qwen3_14b, group="cfg/model", name="qwen3_14b")
store(qwen3_32b, group="cfg/model", name="qwen3_32b")
store(qwen3_30b_a3b_instr, group="cfg/model", name="qwen3_30b_a3b_instr")
store(qwen3_235b_a22b_instr, group="cfg/model", name="qwen3_235b_a22b_instr")

# -------- DeepSeek R1 Distill --------

deepseek_r1_qwen_32b = model_cfg(
    identifier="deepseek_r1_qwen_32b",
    base_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ctx_length=4096,
)
deepseek_r1_llama_70b = model_cfg(
    identifier="deepseek_r1_llama_70b",
    base_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ctx_length=4096,
)
store(deepseek_r1_qwen_32b, group="cfg/model", name="deepseek_r1_qwen_32b")
store(deepseek_r1_llama_70b, group="cfg/model", name="deepseek_r1_llama_70b")

# -------- OLMo 3.1 --------

olmo31_32b_instr = model_cfg(
    identifier="olmo31_32b_instr",
    base_model_name_or_path="allenai/Olmo-3.1-32B-Instruct",
    ctx_length=4096,
)
store(olmo31_32b_instr, group="cfg/model", name="olmo31_32b_instr")

# -------- Apertus (Swiss AI) --------

apertus_70b_instr = model_cfg(
    identifier="apertus_70b_instr",
    base_model_name_or_path="swiss-ai/Apertus-70B-Instruct-2509",
    ctx_length=4096,
    generation_config=fbuilds(
        GenerationConfig,
        bos_token_id=1,
        eos_token_id=2,
        do_sample=True,
        # recommended settings from https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509
        temperature=0.8,
        top_p=0.9,
        max_length=65536,
    ),
)
store(apertus_70b_instr, group="cfg/model", name="apertus_70b_instr")

# -------- Mistral Large --------

mistral_large_instr_2411 = model_cfg(
    identifier="mistral_large_instr_2411",
    base_model_name_or_path="mistralai/Mistral-Large-Instruct-2411",
    ctx_length=4096,
)
store(
    mistral_large_instr_2411,
    group="cfg/model",
    name="mistral_large_instr_2411",
)

# -------- Tokenizer --------

store(
    fbuilds(
        load_tokenizer,
        model_name_or_path="${cfg.model.base_model_name_or_path}",
        tokenizer_cls="${cfg.model.tokenizer_cls}",
        tokenizer_kwargs="${cfg.model.tokenizer_kwargs}",
    ),
    group="cfg/tokenizer",
    name="default",
)
