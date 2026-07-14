# -*- coding: utf-8 -*-
"""
Tests for MILCCI.

Run with:  python -m pytest tests/ -v
  or:      python tests/test_milcci.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import milcci
from milcci.utils import (check_empty_list, make_labels_unique_order,
                          find_indices_in_list, check_if_labels_batches,
                          lists2list, spec_corr, identity)
from milcci.solvers import solve_regularized, solve_nnls_multi, solve_ls_for_phi
from milcci.core import reconstruct, _make_A_full


def test_utils():
    print('test_utils...', end=' ')
    assert check_empty_list([]) is True
    assert check_empty_list([1]) is False
    assert check_empty_list(np.array([])) is False

    labels = [2, 2, 0, 0, 1, 1]
    unique = make_labels_unique_order(labels)
    assert list(unique) == [2, 0, 1], 'got %s' % str(unique)

    assert find_indices_in_list([10, 20, 10], 10) == [0, 2]
    assert lists2list([[1, 2], [3]]) == [1, 2, 3]

    assert check_if_labels_batches([0, 0, 1, 1, 2]) is True
    assert check_if_labels_batches([0, 1, 0]) is False

    v = np.array([1.0, 2.0, 3.0])
    assert abs(spec_corr(v, v) - 1.0) < 1e-10, 'self-corr should be 1'
    assert identity(5) == 5
    print('PASSED')


def test_solvers():
    print('test_solvers...', end=' ')
    rng = np.random.RandomState(0)
    A = rng.randn(10, 3)
    x_true = rng.randn(3, 1)
    b = A @ x_true

    x_hat = solve_regularized(A, b, solver='inv')
    assert x_hat.shape == (3, 1), 'shape %s' % str(x_hat.shape)
    err = np.mean((x_hat - x_true) ** 2)
    assert err < 1e-10, 'LS error too large: %.2e' % err

    # NNLS
    A_pos = np.abs(rng.randn(10, 3))
    x_pos = np.abs(rng.randn(3))
    b_pos = A_pos @ x_pos
    x_nnls = solve_nnls_multi(A_pos, b_pos)
    err_nnls = np.mean((x_nnls - x_pos) ** 2)
    assert err_nnls < 1e-6, 'NNLS error too large: %.2e' % err_nnls

    # solve_ls_for_phi
    data = rng.randn(5, 20)
    A_small = rng.randn(5, 3)
    phi = solve_ls_for_phi(data, A_small, lambda_l2=0.01)
    assert phi.shape == (3, 20), 'phi shape %s' % str(phi.shape)
    print('PASSED')


def test_synthetic_generation():
    print('test_synthetic_generation...', end=' ')
    synth = milcci.generate_synthetic_data(
        N=20, T=40, n_ensembles_each=[2, 2],
        axis_values=[[0, 1], [0, 1]],
        noise_std=0.1, seed=0, trials_per_condition=1,
    )
    Y = synth['Y']
    n_unique = 4  # 2 x 2
    M = n_unique * 1
    assert Y.shape == (20, 40, M), 'Y shape %s' % str(Y.shape)
    assert len(synth['labels']) == M, 'labels length %d' % len(synth['labels'])
    assert len(synth['numbers2tuples']) == n_unique
    assert np.all(np.isfinite(Y)), 'Y has non-finite values'

    # check ground-truth reco
    A_true = synth['A_true']
    Phi_true = synth['Phi_true']
    for m in range(M):
        reco = A_true[:, :, m] @ Phi_true[:, :, m].T
        r2 = 1 - np.sum((Y[:, :, m] - reco) ** 2) / (np.sum((Y[:, :, m] - Y[:, :, m].mean()) ** 2) + 1e-18)
        assert r2 > 0.5, 'ground truth R^2 for trial %d is %.3f (too low)' % (m, r2)
    print('PASSED')


def test_synthetic_gp_structure():
    """Check that GP traces have within-axis correlation."""
    print('test_synthetic_gp_structure...', end=' ')
    synth = milcci.generate_synthetic_data(
        N=20, T=60, n_ensembles_each=[2, 2],
        axis_values=[[0, 1], [0, 1]],
        noise_std=0.1, seed=42, trials_per_condition=4,
        gp_sigma=0.2,
    )
    Phi = synth['Phi_true']
    labels_arr = np.array(synth['labels_tuples'])

    # axis 0, ensemble 0: trials sharing axis_0 value should be correlated
    for val in [0, 1]:
        matching = np.where(labels_arr[:, 0] == val)[0]
        corrs = []
        for i in range(len(matching)):
            for j in range(i + 1, len(matching)):
                c = np.corrcoef(Phi[:, 0, matching[i]], Phi[:, 0, matching[j]])[0, 1]
                corrs.append(c)
        mean_corr = np.mean(corrs)
        assert mean_corr > 0.5, (
            'axis_0 val=%d ens_0: within-group Phi corr %.3f is too low' % (val, mean_corr)
        )
    print('PASSED')


def test_fit_discrete():
    print('test_fit_discrete...', end=' ')
    synth = milcci.generate_synthetic_data(
        N=25, T=50, n_ensembles_each=[2, 2],
        axis_values=[[0, 1], [0, 1]],
        noise_std=0.15, seed=42, trials_per_condition=3,
    )
    M = len(synth['labels'])
    result = milcci.fit(
        data=synth['Y'], labels=synth['labels'],
        numbers2tuples=synth['numbers2tuples'],
        n_ensembles=4, n_ensembles_each=[2, 2],
        nu=[0.01] * 4, lambda_similarity=500,
        factor_A=5, decor_A=5, num_repeats=3,
        split_A=True, seed=42,
    )
    Phi = result['Phi']
    A_full = result['A_full']
    assert Phi.shape == (50, 4, M), 'Phi shape %s' % str(Phi.shape)
    assert A_full.shape == (25, 4, M), 'A_full shape %s' % str(A_full.shape)
    assert np.all(np.isfinite(Phi)), 'Phi non-finite'
    assert np.all(np.isfinite(A_full)), 'A_full non-finite'

    r2 = milcci.global_r2(synth['Y'], A_full, Phi)
    assert r2 > 0, 'R^2 = %.4f is too low' % r2

    # check axis structure
    labels_unique = result['params']['labels_unique_order']
    n2t = synth['numbers2tuples']
    for val in [0, 1]:
        matching = [i for i, lab in enumerate(labels_unique) if n2t[lab][0] == val]
        if len(matching) > 1:
            A0 = result['A'][:, :2, matching[0]]
            A1 = result['A'][:, :2, matching[1]]
            corr = np.corrcoef(A0.flatten(), A1.flatten())[0, 1]
            assert corr > 0.99, (
                'axis-0 ensembles corr=%.3f for same axis-0 value' % corr
            )
    print('PASSED (R^2=%.3f)' % r2)


def test_fit_continuous():
    print('test_fit_continuous...', end=' ')
    synth = milcci.generate_synthetic_data(
        N=20, T=40, n_ensembles_each=[2, 2],
        axis_values=[[0, 1, 2, 3], [0, 1]],
        noise_std=0.15, seed=99, trials_per_condition=2,
    )
    result = milcci.fit(
        data=synth['Y'], labels=synth['labels'],
        numbers2tuples=synth['numbers2tuples'],
        n_ensembles=4, n_ensembles_each=[2, 2],
        nu=[0.01] * 4, lambda_similarity=500,
        cont_axis_list=[0],
        params_basis_pattern={'wind_size': 2, 'weight_min': 0.005, 'weight_max': 0.01, 'one_or_two_sides': 2},
        split_A=True, seed=42,
    )
    assert np.all(np.isfinite(result['Phi'])), 'Phi non-finite'
    assert np.all(np.isfinite(result['A_full'])), 'A non-finite'
    r2 = milcci.global_r2(synth['Y'], result['A_full'], result['Phi'])
    assert r2 > -1.0, 'R^2 = %.4f too low' % r2
    print('PASSED (R^2=%.3f)' % r2)


def test_fit_dynamic_prior():
    print('test_fit_dynamic_prior...', end=' ')
    synth = milcci.generate_synthetic_data(
        N=15, T=30, n_ensembles_each=[2, 2],
        axis_values=[[0, 1], [0, 1]],
        noise_std=0.2, seed=7, trials_per_condition=2,
    )
    result = milcci.fit(
        data=synth['Y'], labels=synth['labels'],
        numbers2tuples=synth['numbers2tuples'],
        n_ensembles=4, n_ensembles_each=[2, 2],
        nu=[0.01] * 4, lambda_similarity=500,
        split_A=True, style_infer_phi='dynamic_prior', seed=42,
    )
    assert np.all(np.isfinite(result['Phi'])), 'Phi non-finite'
    r2 = milcci.global_r2(synth['Y'], result['A_full'], result['Phi'])
    assert r2 > -1.0, 'R^2 = %.4f' % r2
    print('PASSED (R^2=%.3f)' % r2)


def test_reconstruct():
    print('test_reconstruct...', end=' ')
    rng = np.random.RandomState(0)
    N, P, T, M = 10, 3, 20, 5
    A_full = rng.randn(N, P, M)
    Phi = rng.randn(T, P, M)
    Y_hat = reconstruct(A_full, Phi)
    assert Y_hat.shape == (N, T, M), 'shape %s' % str(Y_hat.shape)
    expected = A_full[:, :, 0] @ Phi[:, :, 0].T
    assert np.allclose(Y_hat[:, :, 0], expected), 'reco mismatch'
    print('PASSED')


def test_make_A_full():
    print('test_make_A_full...', end=' ')
    rng = np.random.RandomState(0)
    A_ind = rng.randn(10, 3, 4)
    labels = [0, 0, 1, 2, 3, 3, 2, 1]
    labels_unique = make_labels_unique_order(labels)
    A_full = _make_A_full(A_ind, labels, labels_unique)
    assert A_full.shape == (10, 3, 8), 'shape %s' % str(A_full.shape)
    assert np.allclose(A_full[:, :, 0], A_full[:, :, 1]), 'same-label A mismatch'
    assert not np.allclose(A_full[:, :, 0], A_full[:, :, 2]), 'different-label A should differ'
    print('PASSED')


def test_evaluation():
    print('test_evaluation...', end=' ')
    rng = np.random.RandomState(0)
    N, T, M, P = 10, 20, 5, 3
    A_full = np.abs(rng.randn(N, P, M))
    Phi = np.abs(rng.randn(T, P, M))
    Y = reconstruct(A_full, Phi)
    r2_vec = milcci.per_trial_r2(Y, A_full, Phi)
    assert np.all(r2_vec > 0.999), 'perfect reco should give R^2~1, got %s' % str(r2_vec)
    r2 = milcci.global_r2(Y, A_full, Phi)
    assert r2 > 0.999, 'global R^2 = %.4f' % r2
    rho = milcci.reconstruction_correlation(Y, A_full, Phi)
    assert rho > 0.999, 'rho = %.4f' % rho
    print('PASSED')


if __name__ == '__main__':
    test_utils()
    test_solvers()
    test_synthetic_generation()
    test_synthetic_gp_structure()
    test_reconstruct()
    test_make_A_full()
    test_evaluation()
    test_fit_discrete()
    test_fit_continuous()
    test_fit_dynamic_prior()
    print('\n=== ALL TESTS PASSED ===')
