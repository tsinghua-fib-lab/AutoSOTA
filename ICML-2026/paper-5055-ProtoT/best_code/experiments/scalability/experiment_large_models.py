# experiment.py
import copy
from math import sqrt
from easytuna import run_experiment

ProtoAttn_hyp_config = {
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'PrototypeAttn',
    'SEED': {
        'type': 'seed',
        'seeds': [124, 325],  # list of seeds to run
        'parallel': False,  # run all seeds in parallel or sequentially
    },
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
    'LR1': 7e-4,  # scale lr to match the 2x larger effective batch size from context
    'BATCH': 32,
    #'SEED': 234,
    # --- Architecture hyp-s ---
    'BOTTLENECK': 512,
    'LAYERS': 12,
    'SEQ_LEN': 512,
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'DATASET': 'FineWeb',
    'TRAIN_SIZE': 338_695,
    'TOKENIZER': 'bpe',
    'TOKENIZER_PATH': 'tok/fineweb_bpe_16000.json',
    'VOCAB_SIZE': 16000,
    'TIE_HEAD': True, # tie input and output embeddings
    'EPOCHS1': 10,  
    'USE_LR_SCHEDULER': True,  # use a lr scheduler with warmup and cosine decay
    'USE_COMPILE': True,  # don't use compile for short runs
    'PROFILE_FLOPS': True,  # profile flops during training: needed for constrained optimization
    'DISABLE_TQDM': True  # disable tqdm for cleaner logging
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
        'exper_id': 'LargeModels_FineWeb',  # Unique name for the experiment
        'resume_if_exists': False,
        'train_model_script': "run_clm.py",  # script to run for each trial
        'sampler_name': 'cBO',  # cBO for optimal updates or cTPE for more exploration
        'metric_name': 'final_val_ppl',
        'optim_direction': 'minimize',
        'constraints': {
            'flops_per_example': {
                'max_value': 1e100,  # unlimited flops
            }
        },  # constraints on flops
        'n_trials': 1,
        'n_parallel_trials': 1,
        'n_startup_trials': 25,  # Number of random trials to run before the main optimization
        'verbose': True,
    }
    # Run all studies sequentially. Each large model can occupy most of the GPU
    # memory, so avoid launching multiple EasyTuna workers at once.
    ProtoAttn_hyp_config['LR1'] = 2.0e-3 #* sqrt(2)  # from hyp search with scheduler, scaled for 512 seq len
    llama_hyp_config['LR1'] = 1.7e-3 #* sqrt(2)      # from hyp search with scheduler, scaled for 512 seq len
    mamba_hyp_config['LR1'] = 2.3e-3 #3.8e-3 is best, but unstable #* sqrt(2)      # [not confirmed yet] from hyp search with scheduler, scaled for 512 seq len
    deltanet_hyp_config['LR1'] = 6.8e-3 * sqrt(2)  #from hyp search with scheduler

    runs = [
        ('prototype_attn_more_seeds', ProtoAttn_hyp_config),
        ('llama_scheduler_scaled', llama_hyp_config),
        ('mamba_scheduler_2.3e-3_lr', mamba_hyp_config),
        ('deltanet_scheduler_scaled', deltanet_hyp_config),
    ]

    for study_id, hyp_config in runs:
        print(f"Starting {study_id}...")
        run_experiment(
            study_id=study_id,
            hyp_config=copy.deepcopy(hyp_config),
            **exper_config,
        )
        print(f"Completed {study_id}.")

    print("All sequential experiments completed.")

if __name__ == "__main__":
    main()
