from basicts.metrics.numpy_metrics import (
    masked_coverage,
    masked_pi_width,
    masked_pi_width_median,
    masked_winkler_score,
)
from basicts.utils.data_utils import find_close
from tsl.utils.casting import torch_to_numpy

def evaluate_multiquantile_predictions(q_hat, r_true, mask, alphas, target_qs ,loader_name='eval'):


    results = dict()
    for target_alpha in alphas:
        idx_low = find_close(target_alpha / 2, target_qs)
        idx_high = find_close(1 - target_alpha / 2, target_qs)
        pi_interval = [q_hat[idx_low], q_hat[idx_high]]

        results[f'{loader_name}_coverage_at_{(1 - target_alpha) * 100}'] = masked_coverage(torch_to_numpy(pi_interval),
                                                                               torch_to_numpy(r_true),
                                                                               torch_to_numpy(mask))
        # compute pi width
        results[f'{loader_name}_pi_width_at_{(1 - target_alpha) * 100}'] = masked_pi_width(torch_to_numpy(pi_interval),
                                                                               torch_to_numpy(r_true),
                                                                               torch_to_numpy(mask))
        results[f'{loader_name}_pi_width_median_at_{(1 - target_alpha) * 100}'] = masked_pi_width_median(torch_to_numpy(pi_interval),
                                                                                          torch_to_numpy(r_true),
                                                                                          torch_to_numpy(mask))
        # compute winkler score
        results[f'{loader_name}_winkler_at_{(1 - target_alpha) * 100}'] = masked_winkler_score(torch_to_numpy(pi_interval),
                                                                                   torch_to_numpy(r_true),
                                                                                   torch_to_numpy(mask),
                                                                                   alpha=target_alpha)
    return results
