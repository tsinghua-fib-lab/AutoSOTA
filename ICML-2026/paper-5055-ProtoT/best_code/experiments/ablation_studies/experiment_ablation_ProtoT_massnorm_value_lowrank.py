# experiment_ablation_ProtoT_massnorm_value_lowrank.py
import copy

from easytuna import run_experiment

ProtoAttn_hyp_config = {
    # --- Constants (passed through verbatim; no dict wrapper) ---
    'MODEL': 'PrototypeAttn',
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
    ProtoT ablation: run three variants and sweep LR sequentially:
      1) default,
      2) mass norm disabled,
      3) value low-rank disabled (with mass norm enabled).
    """
    exper_config = {
        'exper_id': 'ablation_ProtoT_massnorm_value_lowrank',
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
        ('default', {}),
        ('no_mass_norm', {'PROTO_DISABLE_MASS_NORM': True}),
        ('no_value_lowrank', {'PROTO_DISABLE_VALUE_LOWRANK': True}),
    ]
    lr_list = [1.0e-3, 2.0e-3, 3.0e-3]

    for variant_name, variant_overrides in variants:
        for lr in lr_list:
            hyp_config = copy.deepcopy(ProtoAttn_hyp_config)
            hyp_config.update(variant_overrides)
            hyp_config['LR1'] = lr

            lr_tag = str(lr).replace('.', 'p')
            study_id = f'prototype_attn_ablation_{variant_name}_lr_{lr_tag}_fixed_wd'
            print(f'Running study: {study_id}')

            run_experiment(
                study_id=study_id,
                hyp_config=hyp_config,
                **exper_config,
            )

    print('All sequential experiments completed.')


if __name__ == '__main__':
    main()
