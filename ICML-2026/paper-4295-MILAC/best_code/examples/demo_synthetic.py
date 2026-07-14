# -*- coding: utf-8 -*-
"""
MILCCI Demo: Synthetic Data
============================
Generates a synthetic multi-axis tensor, runs MILCCI decomposition,
evaluates reconstruction quality, and produces plots.

Usage:
    python examples/demo_synthetic.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')

import milcci
from milcci import plotting


def main():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'demo_output')
    os.makedirs(save_path, exist_ok=True)
    print('Saving figures to: %s' % os.path.abspath(save_path))

    print('=' * 60)
    print('MILCCI Synthetic Data Demo')
    print('=' * 60)

    # ------------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------------
    print('\n--- Generating synthetic data ---')
    synth = milcci.generate_synthetic_data(
        N=30,           # neurons
        T=80,           # time bins
        n_ensembles_each=[2, 2],  # 2 ensembles per axis
        axis_values=[[0, 1, 2], [0, 1]],  # 3 values on axis 0, 2 on axis 1
        noise_std=0.2,
        seed=42,
    )
    Y = synth['Y']
    labels = synth['labels']
    numbers2tuples = synth['numbers2tuples']
    N, T, M = Y.shape
    n_ensembles = sum(synth['n_ensembles_each'])

    print('Data shape: N=%d, T=%d, M=%d' % (N, T, M))
    print('Ensembles: %d total (%s per axis)' % (n_ensembles, str(synth['n_ensembles_each'])))
    print('Class names: %s' % str(synth['class_names']))
    print('Label tuples: %s' % str(synth['labels_tuples']))

    # ------------------------------------------------------------------
    # 2. Run MILCCI
    # ------------------------------------------------------------------
    print('\n--- Running MILCCI ---')
    result = milcci.fit(
        data=Y,
        labels=labels,
        numbers2tuples=numbers2tuples,
        n_ensembles=n_ensembles,
        n_ensembles_each=synth['n_ensembles_each'],
        nu=[0.01] * n_ensembles,
        lambda_similarity=100,
        factor_A=5,
        decor_A=2,
        num_repeats=15,
        cont_axis_list=[],      # both axes are discrete here
        split_A=True,
        another_update_for_A=False,
        params_init_A={'ensemble_positive': False},
        verbose=True,
        seed=42,
    )

    Phi = result['Phi']
    A = result['A']
    A_full = result['A_full']
    labels_unique_order = result['params']['labels_unique_order']

    print('Result shapes:')
    print('  Phi:    %s' % str(Phi.shape))
    print('  A:      %s' % str(A.shape))
    print('  A_full: %s' % str(A_full.shape))

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    print('\n--- Evaluation ---')

    r2_global = milcci.global_r2(Y, A_full, Phi)
    print('Global R^2:          %.4f' % r2_global)

    rho = milcci.reconstruction_correlation(Y, A_full, Phi)
    print('Reconstruction rho:  %.4f' % rho)

    r2_per_trial = milcci.per_trial_r2(Y, A_full, Phi)
    print('Per-trial R^2: mean=%.4f, std=%.4f, min=%.4f, max=%.4f'
          % (r2_per_trial.mean(), r2_per_trial.std(),
             r2_per_trial.min(), r2_per_trial.max()))

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    print('\n--- Generating plots ---')

    # A heatmaps
    plotting.plot_A_heatmaps(
        A, labels_unique_order, numbers2tuples,
        class_names=synth['class_names'],
        n_ensembles_each=synth['n_ensembles_each'],
        save_path=save_path, figname_prefix='A_heatmaps',
    )

    # A similarity matrix
    plotting.plot_A_similarity_matrix(
        A, labels_unique_order, numbers2tuples,
        n_ensembles_each=synth['n_ensembles_each'],
        class_names=synth['class_names'],
        save_path=save_path, figname_prefix='A_similarity',
    )

    # Phi traces
    plotting.plot_phi_traces(
        Phi, labels, labels_unique_order, numbers2tuples,
        n_ensembles_each=synth['n_ensembles_each'],
        class_names=synth['class_names'],
        save_path=save_path, figname_prefix='Phi_traces',
    )

    # Reconstruction comparison
    plotting.plot_reconstruction(
        Y, A_full, Phi,
        trial_indices=[0, 2, 4],
        save_path=save_path, figname_prefix='reconstruction',
    )

    # R^2 bar plot
    plotting.plot_r2_per_trial(
        r2_per_trial, labels=labels, numbers2tuples=numbers2tuples,
        save_path=save_path, figname_prefix='r2_per_trial',
    )

    # Ground-truth comparison
    plotting.plot_ground_truth_comparison(
        A, synth['A_true'], Phi, synth['Phi_true'],
        labels_unique_order,
        save_path=save_path, figname_prefix='gt_comparison',
    )

    # Summary dashboard
    plotting.plot_summary(
        Y, result, numbers2tuples, labels,
        class_names=synth['class_names'],
        n_ensembles_each=synth['n_ensembles_each'],
        save_path=save_path, figname_prefix='summary',
    )

    # ------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------
    print('\n--- Sanity checks ---')
    assert Phi.shape == (T, n_ensembles, M), 'Phi shape mismatch: %s' % str(Phi.shape)
    assert A_full.shape == (N, n_ensembles, M), 'A_full shape mismatch: %s' % str(A_full.shape)
    assert np.all(np.isfinite(Phi)), 'Phi has non-finite values'
    assert np.all(np.isfinite(A_full)), 'A_full has non-finite values'
    assert r2_global > -1.0, 'R^2 is unreasonably low: %.4f' % r2_global

    # axis structure: conditions sharing axis_0 should have identical A[:,:2]
    for val in synth['axis_values'][0]:
        matching = [i for i, lab in enumerate(labels_unique_order)
                    if numbers2tuples[lab][0] == val]
        if len(matching) > 1:
            A_block_0 = A[:, :2, matching[0]]
            A_block_1 = A[:, :2, matching[1]]
            corr = np.corrcoef(A_block_0.flatten(), A_block_1.flatten())[0, 1]
            print('  axis_0 value=%s: A[:,:2] corr between conds %d,%d = %.3f'
                  % (str(val), matching[0], matching[1], corr))
            assert corr > 0.99, (
                'axis structure violated: corr=%.3f for axis_0 value=%s' % (corr, str(val))
            )

    print('\nAll sanity checks passed.')
    print('=' * 60)
    print('Demo complete. Figures saved to: %s' % os.path.abspath(save_path))


if __name__ == '__main__':
    main()
