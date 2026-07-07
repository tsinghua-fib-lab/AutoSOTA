from hydra_zen import store

from entquant.super_weights.super_weights import SuperWeightsConfig
from run.hydra_zen import fbuilds

super_weights_config = fbuilds(
    SuperWeightsConfig,
    tokenizer="${cfg.tokenizer}",
    include="*mlp*.down_proj*",
    spike_threshold=50.0,
    top_k=25,
)
super_weights_config_low = super_weights_config(spike_threshold=10.0)
super_weights_config_off = super_weights_config(spike_threshold=float("inf"))

store(super_weights_config, group="cfg/super_weights", name="default")
store(super_weights_config_low, group="cfg/super_weights", name="low")
store(super_weights_config_off, group="cfg/super_weights", name="off")
