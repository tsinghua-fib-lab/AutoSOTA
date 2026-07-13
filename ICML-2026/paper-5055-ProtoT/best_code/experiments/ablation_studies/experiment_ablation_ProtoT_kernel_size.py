# experiment.py
import copy
from easytuna import run_experiment

ProtoAttn_hyp_config = {
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'PrototypeAttn',
}
llama_hyp_config = {
    'HEADS': {
        'type': int, 'init': 8, 'val_list': [2, 8]
    },
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'llama',
}
mamba_hyp_config = {
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'mamba',
}
deltanet_hyp_config = {
    'HEADS': 8,
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'deltanet',
}
common_hyp_config = {
    # --- Optimisation hyp-s ---
    'BATCH': 32,
    'SEED': {
        'type': 'seed',
        'seeds': [234, 124, 325],  # list of seeds to run
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
        'n_trials': 1,
        'n_parallel_trials': 1,
        'n_startup_trials': 0,  # Number of random trials to run before the main optimization
        'verbose': True,
    }

    # Deterministic sweep: run each (kernel, lr) exactly once, sequentially.
    kernel_list = [4, 5, 6, 7]
    lr_list = [1.0e-3, 2.0e-3, 3.0e-3]
    for k in kernel_list:
        for lr in lr_list:
            hyp_config = copy.deepcopy(ProtoAttn_hyp_config)
            hyp_config['PROTO_KERNEL_SIZE'] = k
            hyp_config['LR1'] = lr

            lr_tag = str(lr).replace('.', 'p')
            study_id = f'prototype_attn_kernel_size_{k}_lr_{lr_tag}_fixed_wd'
            print(f"Running study: {study_id}")

            run_experiment(
                study_id=study_id,
                hyp_config=hyp_config,
                **exper_config,
            )

    print("All sequential experiments completed.")

if __name__ == "__main__":
    main()
