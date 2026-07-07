import torch
import numpy as np

from utils.vtrace import compute_v_trace

def _ground_truth_calculation(discounts, log_rhos, rewards, values,
                              values_t_plus_1, clip_rho_threshold,
                              clip_pg_rho_threshold):
    """Calculates the ground truth for V-trace in Python/Numpy."""
    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)
    cs = np.minimum(rhos, clip_rho_threshold)
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)
    clipped_pg_rhos = rhos
    if clip_pg_rho_threshold:
        clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

    # This is a very inefficient way to calculate the V-trace ground truth.
    # We calculate it this way because it is close to the mathematical notation of
    # V-trace.
    # v_s = V(x_s)
    #       + \sum^{T-1}_{t=s} \gamma^{t-s}
    #         * \prod_{i=s}^{t-1} c_i
    #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
    # Note that when we take the product over c_i, we write `s:t` as the notation
    # of the paper is inclusive of the `t-1`, but Python is exclusive.
    # Also note that np.prod([]) == 1.

    for s in range(seq_len):
        v_s = np.copy(values[s])  # Very important copy.
        for t in range(s, seq_len):
            v_s += (
              np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t],
                                                        axis=0) * clipped_rhos[t] *
              (rewards[t] + discounts[t] * values_t_plus_1[t] - values[t]))
        vs.append(v_s)
    vs = np.stack(vs, axis=0)
    # pg_advantages = (
    #     clipped_pg_rhos * (rewards + discounts * np.concatenate(
    #         [vs[1:], values_t_plus_1[None, -1]], axis=0) - values))
    pg_advantages = (
        (rewards + discounts * np.concatenate(
            [vs[1:], values_t_plus_1[None, -1]], axis=0) - values))

    return vs, pg_advantages

def test_vtrace():
    done = np.zeros((5, 256))
    discounts = np.ones((5, 256)) * 0.9
    log_rhos = np.random.randn(5, 256)
    rewards = np.random.randn(5, 256) + 5.0
    values = np.random.random((5, 256))
    values_t_plus_1 = np.hstack( (values[:, :-1], np.random.random((5, 1))) )

    i = 2
    true_vs, true_adv = _ground_truth_calculation(discounts[i], log_rhos[i], rewards[i], values[i], values_t_plus_1[i], 2.0, 2.0)

    rewards = torch.from_numpy(rewards)
    values = torch.from_numpy(values)
    values_t_plus_1 = torch.from_numpy(values_t_plus_1)
    done = torch.from_numpy(done)
    rhos = torch.from_numpy(np.exp(log_rhos))
    vs, adv = compute_v_trace(rewards, values, values_t_plus_1, done, rhos, 0.9, 256, 2.0)
    vs = vs[i].numpy()
    adv = adv[i].numpy()
    assert np.allclose(true_vs, vs)
    assert np.allclose(true_adv, adv)