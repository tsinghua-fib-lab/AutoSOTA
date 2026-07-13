import copy
from easytuna import run_experiment

ProtoAttn_hyp_config = {
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'PrototypeAttn',
    'LAYERS': 6,
    'BOTTLENECK': 256,
}
common_hyp_config = {
    # --- Optimisation hyp-s ---
    'BATCH': 32,
    'SEED': {
        'type': 'seed',
        'seeds': [234, 124, 325],  # list of seeds to run
        'parallel': False,  # run all seeds in parallel or sequentially
    },
    # --- Architecture hyp-s ---
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'DATASET': 'FineWeb',
    'TOKENIZER': 'bpe',
    'TOKENIZER_PATH': 'tok/fineweb_bpe_16000.json',
    'VOCAB_SIZE': 16000,
    'TIE_HEAD': True,  # tie input and output embeddings
    'EPOCHS1': 10,
    'USE_LR_SCHEDULER': True,  # use a lr scheduler with warmup and cosine decay
    'USE_COMPILE': True,  # don't use compile for short runs
    'PROFILE_FLOPS': True,  # profile flops during training: needed for constrained optimization
    'DISABLE_TQDM': True,  # disable tqdm for cleaner logging
    'DISABLE_GENERATION': True,  # disable text generation after training
}
ProtoAttn_hyp_config.update(common_hyp_config)


def main():
    """
    ProtoAttn scalability sweep over h/l/ctx variants with sequential LR sweeps.
    """
    exper_config = {
        'exper_id': 'scalability_ProtoAttn_h_l_ctx',
        'resume_if_exists': False,
        'train_model_script': 'run_clm.py',
        'sampler_name': 'cBO',  # cBO for optimal updates or cTPE for more exploration
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
        ('2x_hdim_256seq', {'BOTTLENECK': 512, 'SEQ_LEN': 256}),
        ('2x_hdim_512seq', {'BOTTLENECK': 512, 'SEQ_LEN': 512}),
        ('2x_hdim_1024seq', {'BOTTLENECK': 512, 'SEQ_LEN': 1024}),
        ('2x_hdim_2048seq', {'BOTTLENECK': 512, 'SEQ_LEN': 2048}),
        ('2x_layers_256seq', {'LAYERS': 12, 'SEQ_LEN': 256}),
        ('2x_layers_512seq', {'LAYERS': 12, 'SEQ_LEN': 512}),
        ('2x_layers_1024seq', {'LAYERS': 12, 'SEQ_LEN': 1024}),
        ('2x_layers_2048seq', {'LAYERS': 12, 'SEQ_LEN': 2048}),
    ]
    lr_list = [1.0e-3, 2.0e-3, 3.0e-3]

    for variant_name, variant_overrides in variants:
        for lr in lr_list:
            hyp_config = copy.deepcopy(ProtoAttn_hyp_config)
            hyp_config.update(variant_overrides)
            hyp_config['LR1'] = lr

            lr_tag = str(lr).replace('.', 'p')
            study_id = f'prototype_attn_{variant_name}_lr_{lr_tag}_fixed_wd'
            print(f'Running study: {study_id}')

            run_experiment(
                study_id=study_id,
                hyp_config=hyp_config,
                **exper_config,
            )

    print('All sequential experiments completed.')


if __name__ == '__main__':
    main()
