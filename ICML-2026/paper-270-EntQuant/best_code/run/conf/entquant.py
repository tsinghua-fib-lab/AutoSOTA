from hydra_zen import store, ZenField
from hydra_zen.typing import Builds
from omegaconf import MISSING

from entquant.compression.backends import nvCOMPBackend
from entquant.quantization.optimizer import SymmetricEntropyOptimizer, WrappedAbsmaxOptimizer
from run.hydra_zen import fbuilds, make_config

# -------- Entropy optimizer --------

absmax_optimizer = fbuilds(WrappedAbsmaxOptimizer)
entquant_optimizer_4bit = fbuilds(
    SymmetricEntropyOptimizer,
    lr=1.0,
    reg_param=3.9,
    device_compute="cuda",
)
entquant_optimizer_3bit = entquant_optimizer_4bit(reg_param=14.5)
entquant_optimizer_2bit = entquant_optimizer_4bit(reg_param=58.0, lr=0.25)

store(absmax_optimizer, group="cfg/entquant/optimizer", name="absmax")
store(entquant_optimizer_4bit, group="cfg/entquant/optimizer", name="default")
store(entquant_optimizer_4bit, group="cfg/entquant/optimizer", name="symmetric_4bit")  # same as default
store(entquant_optimizer_3bit, group="cfg/entquant/optimizer", name="symmetric_3bit")
store(entquant_optimizer_2bit, group="cfg/entquant/optimizer", name="symmetric_2bit")

# -------- EntQuant config --------
# Flat config matching EntQuantModel.from_pretrained parameters.

entquant_config = make_config(
    weight_qtype=ZenField(str, "qfloat8"),
    activation_qtype=ZenField(str | None, None),
    quantize=ZenField(bool, True),
    compress=ZenField(bool, True),
    include=[
        "*mlp*.up_proj*",
        "*mlp*.down_proj*",
        "*mlp*.gate_proj*",
        "*self_attn*.q_proj*",
        "*self_attn*.k_proj*",
        "*self_attn*.v_proj*",
        "*self_attn*.o_proj*",
    ],
    exclude=ZenField(list[str] | None, None),
    optimizer=MISSING,
    optimizer_fallback=absmax_optimizer,
    backend=ZenField(Builds[nvCOMPBackend] | None, fbuilds(nvCOMPBackend)),
    block_pattern=ZenField(str, "model.layers.*"),
    hydra_convert="all",  # crucial because otherwise type union hints make hydra fail
)

# Named variants
store(
    entquant_config,
    group="cfg/entquant",
    name="default",
)
store(  # same as default
    entquant_config,
    group="cfg/entquant",
    name="fp8",
)
store(
    entquant_config(weight_qtype="qint8"),
    group="cfg/entquant",
    name="int8",
)
store(
    entquant_config(quantize=True, compress=False, backend=None),
    group="cfg/entquant",
    name="convert_fp8",
)
store(
    entquant_config(quantize=True, compress=False, backend=None, weight_qtype="qint8"),
    group="cfg/entquant",
    name="convert_int8",
)
store(
    entquant_config(quantize=False, compress=True),
    group="cfg/entquant",
    name="load_fp8",
)
store(
    entquant_config(quantize=False, compress=True, weight_qtype="qint8"),
    group="cfg/entquant",
    name="load_int8",
)
store(
    entquant_config(quantize=True, compress=False, backend=None),
    group="cfg/entquant",
    name="quantize_fp8",
)
store(
    entquant_config(quantize=True, compress=False, backend=None, weight_qtype="qint8"),
    group="cfg/entquant",
    name="quantize_int8",
)
