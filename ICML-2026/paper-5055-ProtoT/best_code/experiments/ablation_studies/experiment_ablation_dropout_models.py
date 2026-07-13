# experiment_ablation_dropout_models.py
import copy

from easytuna import run_experiment


common_hyp_config = {
    # --- Optimisation hyp-s ---
    "BATCH": 32,
    "SEED": {
        "type": "seed",
        "seeds": [234, 124, 325],  # list of seeds to run
        "parallel": True,  # run all seeds in parallel or sequentially
    },
    # --- Architecture/data constants ---
    "DATASET": "FineWeb",
    "TOKENIZER": "bpe",
    "TOKENIZER_PATH": "tok/fineweb_bpe_16000.json",
    "VOCAB_SIZE": 16000,
    "TIE_HEAD": True,  # tie input and output embeddings
    "EPOCHS1": 10,
    "USE_LR_SCHEDULER": True,  # use a lr scheduler with warmup and cosine decay
    "USE_COMPILE": True,  # model-specific overrides below
    "PROFILE_FLOPS": True,  # profile flops during training: needed for constrained optimization
    "DISABLE_TQDM": True,  # disable tqdm for cleaner logging
    "DISABLE_GENERATION": True,  # disable text generation after training
}


model_sweeps = [
    {
        "model_tag": "protot",
        "base_config": {"MODEL": "PrototypeAttn"},
        "lr_list": [1.0e-3, 2.0e-3, 3.0e-3],
        "variants": [
            ("default_dropout_0p1", {"DROPOUT": 0.1}),
            ("no_dropout", {"DROPOUT": 0.0}),
        ],
    },
    {
        "model_tag": "llama",
        "base_config": {"MODEL": "llama", "HEADS": 4},
        "lr_list": [0.8e-3, 1.6e-3, 3.2e-3],
        "variants": [
            ("default_dropout_0p1", {"DROPOUT": 0.1, "ATTN_DROPOUT": 0.1}),
            ("no_self_attn_dropout", {"DROPOUT": 0.1, "ATTN_DROPOUT": 0.0}),
            ("no_dropout", {"DROPOUT": 0.0, "ATTN_DROPOUT": 0.0}),
        ],
    },
    {
        "model_tag": "mamba",
        "base_config": {"MODEL": "mamba", "USE_COMPILE": False},
        "lr_list": [1.9e-3, 3.8e-3, 7.6e-3],
        "variants": [
            ("default_dropout_0p1", {"DROPOUT": 0.1}),
            ("no_dropout", {"DROPOUT": 0.0}),
        ],
    },
    {
        "model_tag": "deltanet",
        "base_config": {"MODEL": "deltanet", "HEADS": 4, "USE_COMPILE": False},
        "lr_list": [3.4e-3, 6.8e-3, 13.6e-3],
        "variants": [
            ("default_dropout_0p1", {"DROPOUT": 0.1}),
            ("no_dropout", {"DROPOUT": 0.0}),
        ],
    },
]


def main():
    """
    Reproduce dropout ablations across ProtoT, LLaMA, Mamba, and DeltaNet.
    Each setting is run as a deterministic LR sweep with fixed LR candidates.
    """
    exper_config = {
        "exper_id": "ablation_dropout_models",
        "resume_if_exists": False,
        "train_model_script": "run_clm.py",
        "sampler_name": "cBO",
        "metric_name": "final_val_ppl",
        "optim_direction": "minimize",
        "constraints": {
            "flops_per_example": {
                "max_value": 100e9,  # effectively unconstrained
            }
        },
        "n_trials": 1,
        "n_parallel_trials": 1,
        "n_startup_trials": 0,
        "verbose": True,
    }

    for model_spec in model_sweeps:
        model_tag = model_spec["model_tag"]
        lr_list = model_spec["lr_list"]
        base_config = copy.deepcopy(common_hyp_config)
        base_config.update(model_spec["base_config"])

        for variant_name, variant_overrides in model_spec["variants"]:
            for lr in lr_list:
                hyp_config = copy.deepcopy(base_config)
                hyp_config.update(variant_overrides)
                hyp_config["LR1"] = lr

                lr_tag = str(lr).replace(".", "p")
                study_id = f"{model_tag}_{variant_name}_lr_{lr_tag}"
                print(f"Running study: {study_id}")

                run_experiment(
                    study_id=study_id,
                    hyp_config=hyp_config,
                    **exper_config,
                )

    print("All sequential experiments completed.")


if __name__ == "__main__":
    main()
