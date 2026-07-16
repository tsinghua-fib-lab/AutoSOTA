# =================================================================
# paths.py
# Description: Single source of truth for attack result/input file paths
# =================================================================

import os


def watermarked_response_path(input_dir, algorithm):
    """Path to a watermark algorithm's watermarked-text input file."""
    return os.path.join(input_dir, f"{algorithm}_response.json")


def attack_result_path(
    save_dir,
    attack_algorithms,
    algorithm,
    model_name=None,
    *,
    num_data=None,
    beta=None,
    percentile=None,
):
    """Full path to an attack's per-(model, algorithm) result JSON.

    Single source of truth shared by generation (write) and evaluation (read)
    so the two can never diverge.
    """
    if attack_algorithms == "BIRA":
        filename = f"BIRA_beta_{beta}_percentile_{percentile}_num_data_{num_data}.json"
        return os.path.join(save_dir, attack_algorithms, model_name, algorithm, filename)

    if attack_algorithms == "vanilla_paraphrasing":
        filename = f"vanilla_paraphrasing_num_data_{num_data}.json"
        return os.path.join(save_dir, attack_algorithms, model_name, algorithm, filename)

    if attack_algorithms in ("dipper-1", "dipper-2"):
        filename = f"dipper_paraphraser_num_data_{num_data}.json"
        return os.path.join(save_dir, attack_algorithms, algorithm, filename)

    if attack_algorithms == "SIRA":
        filename = f"SIRA_num_data_{num_data}.json"
        return os.path.join(save_dir, attack_algorithms, model_name, algorithm, filename)

    raise NotImplementedError(f"Unknown attack_algorithms: {attack_algorithms}")
