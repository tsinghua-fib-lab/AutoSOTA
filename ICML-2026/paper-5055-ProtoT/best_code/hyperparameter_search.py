# experiment.py
import copy
from easytuna import run_experiment

ProtoAttn_hyp_config = {
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'PrototypeAttn',
}
llama_hyp_config = {
    'HEADS': 4,
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'llama',
}
mamba_hyp_config = {
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'mamba',
}
deltanet_hyp_config = {
    'HEADS': 4,
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'deltanet',
}
common_hyp_config = {
    # --- Optimisation hyp-s ---
    'LR1': {
        'type': float, 'init': 1e-3, 'interval': [3e-05, 0.03], 'log': True
    },
    'BATCH': {
        'type': int, 'init': 32, 'val_list': [32, 64, 128]
    },
    'SEED': {
        'type': 'seed',
        'seeds': [234, 124, 325],  # average each trial over 3 seeds
        'parallel': True,  # run all seeds in parallel or sequentially
    },
    # --- Architecture hyp-s ---
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'DATASET': 'FineWeb',
    'TOKENIZER': 'bpe',
    'TOKENIZER_PATH': 'tok/fineweb_bpe_16000.json',
    'VOCAB_SIZE': 16000,
    'TIE_HEAD': True, # tie input and output embeddings
    'EPOCHS1': 10,  
    'USE_LR_SCHEDULER': True,  # use a lr scheduler with warmup and cosine decay
    'USE_COMPILE': True,  # don't use compile for short runs
    'PROFILE_FLOPS': True,  # profile flops during training: needed for constrained optimization
    'DISABLE_TQDM': True,  # disable tqdm for cleaner logging
    'DISABLE_GENERATION': True  # disable text generation after training
}
ProtoAttn_hyp_config.update(common_hyp_config)
llama_hyp_config.update(common_hyp_config)
mamba_hyp_config.update(common_hyp_config)
deltanet_hyp_config.update(common_hyp_config)

def main():
    """
    PrototypeAttention experiment on FineWeb
    """
    exper_config = {
        'exper_id': 'hyp_search_FineWeb_lr_scheduler',
        'resume_if_exists': False,
        'train_model_script': "run_clm.py",  # script to run for each trial
        'sampler_name': 'cBO',  # cBO for optimal updates or cTPE for more exploration
        'metric_name': 'final_val_ppl',
        'optim_direction': 'minimize',
        'constraints': {
            'flops_per_example': {
                'max_value': 100e9,  # unlimited flops: just searching over bs and lr
            }
        },  # constraints on flops
        'n_trials': 50,
        'n_parallel_trials': 1,
        'n_startup_trials': 15,  # Number of random trials to run before the main optimization
        'verbose': True,
    }

    # Run model studies sequentially
    runs = [
        ('prototype_attn', ProtoAttn_hyp_config),
        ('llama', llama_hyp_config),
        ('mamba', mamba_hyp_config),
        ('deltanet', deltanet_hyp_config),
    ]

    for study_id, hyp_config in runs:
        print(f"Starting {study_id} hyperparameter search...")
        run_experiment(
            study_id=study_id,
            hyp_config=copy.deepcopy(hyp_config),
            **exper_config,
        )
        print(f"Completed {study_id} hyperparameter search.")

    print("All hyperparameter searches completed.")

if __name__ == "__main__":
    main()
