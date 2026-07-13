import copy

from easytuna import run_experiment


common_hyp_config = {
    # --- Optimisation hyp-s ---
    'BATCH': 32,
    'SEED': {
        'type': 'seed',
        'seeds': [234, 124, 325],  # list of seeds to run
        'parallel': False,  # run all seeds sequentially
    },
    # --- Architecture hyp-s ---
    'BOTTLENECK': 256,
    'LAYERS': 6,
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'DATASET': 'FineWeb',
    'TRAIN_SIZE': 18_000,
    'TOKENIZER': 'bpe',
    'TOKENIZER_PATH': 'tok/fineweb_bpe_16000.json',
    'VOCAB_SIZE': 16000,
    'TIE_HEAD': True,  # tie input and output embeddings
    'EPOCHS1': 10,
    'USE_COMPILE': True,  # don't use compile for short runs
    'PROFILE_FLOPS': True,  # profile flops during training: needed for constrained optimization
    'DISABLE_TQDM': True,  # disable tqdm for cleaner logging
    'DISABLE_GENERATION': True,  # disable text generation after training
}

model_sweeps = [
    {
        'model_tag': 'deltanet',
        'base_config': {'MODEL': 'deltanet', 'HEADS': 4, 'USE_COMPILE': False},
        'lr_list': [3.4e-3, 6.8e-3, 13.6e-3],
    },
    {
        'model_tag': 'protot',
        'base_config': {'MODEL': 'PrototypeAttn'},
        'lr_list': [1.0e-3, 2.0e-3, 4.0e-3],
    },
    {
        'model_tag': 'llama',
        'base_config': {'MODEL': 'llama', 'HEADS': 4},
        'lr_list': [0.8e-3, 1.6e-3, 3.2e-3],
    },
    {
        'model_tag': 'mamba',
        'base_config': {'MODEL': 'mamba', 'USE_COMPILE': False},
        'lr_list': [1.9e-3, 3.8e-3, 7.6e-3],
    },
]


def main():
    """
    Multi-model context scalability sweep with fixed LR grids and sequential runs.
    """
    exper_config = {
        'exper_id': 'scalability_models_ctx',
        'resume_if_exists': False,
        'train_model_script': 'run_clm.py',
        'sampler_name': 'cBO',
        'metric_name': 'final_val_ppl',
        'optim_direction': 'minimize',
        'constraints': {
            'flops_per_example': {
                'max_value': 100e9,  # effectively unconstrained
            }
        },
        'n_trials': 1,
        'n_parallel_trials': 1,
        'n_startup_trials': 0,
        'verbose': True,
    }

    variants = [
        ('ctx_512', {'SEQ_LEN': 512}),
        ('ctx_1024', {'SEQ_LEN': 1024}),
        ('ctx_2048', {'SEQ_LEN': 2048}),
    ]

    for model_spec in model_sweeps:
        model_tag = model_spec['model_tag']
        lr_list = model_spec['lr_list']
        base_config = copy.deepcopy(common_hyp_config)
        base_config.update(model_spec['base_config'])

        for variant_name, variant_overrides in variants:
            for lr in lr_list:
                hyp_config = copy.deepcopy(base_config)
                hyp_config.update(variant_overrides)
                hyp_config['LR1'] = lr

                lr_tag = str(lr).replace('.', 'p')
                study_id = f'{model_tag}_{variant_name}_lr_{lr_tag}_fixed_wd'
                print(f'Running study: {study_id}')

                run_experiment(
                    study_id=study_id,
                    hyp_config=hyp_config,
                    **exper_config,
                )

    print('All sequential experiments completed.')


if __name__ == '__main__':
    main()
