from hydra_zen import store

from run.hydra_zen import make_config

# =============================================================================
# Hydra Experiment Configs
# =============================================================================

experiment_store = store(group="experiment", package="_global_")
exec_workflow_config = store.get_entry(group=None, name="exec_workflow")["node"]

# ------ Base Model Experiment Config ------

base_model_hydra_defaults = [
    {"override /workflow": "default"},
    {"override /cfg": "default"},
    {"override /run": "default"},
    {"override /cfg/eval": "none"},
    "_self_",
]

experiment_store(
    make_config(
        run=dict(identifier="${cfg.model.identifier}__base_model"),
        hydra_defaults=base_model_hydra_defaults,
        bases=(exec_workflow_config,),
    ),
    name="base_model",
)

# ------ EntQuant Experiment Config ------

entquant_hydra_defaults = [
    {"override /workflow": "entquant"},
    {"override /cfg": "entquant"},
    {"override /run": "default"},
    {"override /cfg/eval": "none"},
    "_self_",
]

experiment_store(
    make_config(
        run=dict(
            identifier="${cfg.model.identifier}__entquant_int8"
            "__rp${cfg.entquant.optimizer.reg_param}"
            "_lr${cfg.entquant.optimizer.lr}"
            "__sw_${cfg.super_weights.spike_threshold}"
        ),
        hydra_defaults=entquant_hydra_defaults
        + [
            {"override /cfg/entquant": "int8"},
            {"override /cfg/entquant/optimizer": "symmetric_4bit"},
        ],
        bases=(exec_workflow_config,),
    ),
    name="entquant_int8",
)

experiment_store(
    make_config(
        run=dict(
            identifier="${cfg.model.identifier}__entquant_fp8"
            "_rp${cfg.entquant.optimizer.reg_param}"
            "_lr${cfg.entquant.optimizer.lr}"
            "__sw_${cfg.super_weights.spike_threshold}"
        ),
        hydra_defaults=entquant_hydra_defaults
        + [
            {"override /cfg/entquant": "fp8"},
            {"override /cfg/entquant/optimizer": "symmetric_4bit"},
        ],
        bases=(exec_workflow_config,),
    ),
    name="entquant_fp8",
)

experiment_store(
    make_config(
        run=dict(identifier="${cfg.model.identifier}__int8"),
        hydra_defaults=entquant_hydra_defaults
        + [
            {"override /cfg/entquant": "quantize_int8"},
            {"override /cfg/entquant/optimizer": "absmax"},
        ],
        bases=(exec_workflow_config,),
    ),
    name="int8",
)

experiment_store(
    make_config(
        run=dict(identifier="${cfg.model.identifier}__fp8"),
        hydra_defaults=entquant_hydra_defaults
        + [
            {"override /cfg/entquant": "quantize_fp8"},
            {"override /cfg/entquant/optimizer": "absmax"},
        ],
        bases=(exec_workflow_config,),
    ),
    name="fp8",
)

# ------ BnB NF4 Experiment Config ------

bnb_hydra_defaults = [
    {"override /workflow": "quantization"},
    {"override /cfg": "quantization"},
    {"override /run": "default"},
    {"override /cfg/eval": "none"},
    {"override /cfg/quantization/config": "bnb_nf4"},
    "_self_",
]

experiment_store(
    make_config(
        run=dict(identifier="${cfg.model.identifier}__bnb_nf4"),
        hydra_defaults=bnb_hydra_defaults,
        bases=(exec_workflow_config,),
    ),
    name="bnb_nf4",
)

experiment_store(
    make_config(
        run=dict(identifier="${cfg.model.identifier}__bnb_nf4__sw"),
        hydra_defaults=[
            {"override /cfg/super_weights": "default"},
        ]
        + bnb_hydra_defaults,
        bases=(exec_workflow_config,),
    ),
    name="bnb_nf4_sw",
)

# ------ HQQ Experiment Config ------

hqq_hydra_defaults = [
    {"override /workflow": "quantization"},
    {"override /cfg": "quantization"},
    {"override /run": "default"},
    {"override /cfg/eval": "none"},
    {"override /cfg/quantization/config": "hqq"},
    "_self_",
]

experiment_store(
    make_config(
        run=dict(
            identifier="${cfg.model.identifier}__hqq"
            "_${cfg.quantization.config.nbits}bit"
            "_g${cfg.quantization.config.group_size}"
        ),
        hydra_defaults=hqq_hydra_defaults,
        bases=(exec_workflow_config,),
    ),
    name="hqq",
)

experiment_store(
    make_config(
        run=dict(
            identifier="${cfg.model.identifier}__hqq"
            "_${cfg.quantization.config.nbits}bit"
            "_g${cfg.quantization.config.group_size}__sw"
        ),
        hydra_defaults=[
            {"override /cfg/super_weights": "default"},
        ]
        + hqq_hydra_defaults,
        bases=(exec_workflow_config,),
    ),
    name="hqq_sw",
)
