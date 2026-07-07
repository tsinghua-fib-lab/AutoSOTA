import torch
from hydra_zen import store, ZenField
from transformers import BitsAndBytesConfig, HqqConfig

from run.hydra_zen import make_config, pbuilds

int8_config = pbuilds(BitsAndBytesConfig, load_in_8bit=True)
nf4_config = pbuilds(
    BitsAndBytesConfig,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
hqq_config = pbuilds(
    HqqConfig,
    nbits=4,
    group_size=64,
)

store(int8_config, group="cfg/quantization/config", name="bnb_int8")
store(nf4_config, group="cfg/quantization/config", name="bnb_nf4")
store(hqq_config, group="cfg/quantization/config", name="hqq")

quantization = make_config(
    config=ZenField(BitsAndBytesConfig | HqqConfig),
    modules_to_exclude=ZenField(list[str], default=["lm_head"]),
    # crucial because otherwise type union hints in BitsAndBytesConfig | HqqConfig
    # can make hydra fail - omegaconf/hydra treat them as containers
    hydra_convert="all",
)

store(quantization, group="cfg/quantization", name="default")
