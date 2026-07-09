"""
Code for reproducing medical QA experiments in Prinster, et al., (2026) "Conformal Policy Control"

Code is partially adapted from https://github.com/jjcherian/conformal-safety
Data is copied from that repo
"""

import numpy as np
import pandas as pd

import pickle
import json
import os

from tqdm import tqdm

import argparse

## For LTT
# from confseq import betting

from typing import List

from utils import (
    get_frac_true_claims_retained,
    get_taus_grid_from_data,
    hb_p_value,
    remove_specific_leading_chars,
    split_dataset,
)


"""Loss function for subclaim factuality False Discovery Rate (FDR)"""


def loss_factuality_fdr(
    claim_scores: List[np.ndarray],  ## Float point scores
    annotations: List[np.ndarray],  ## Boolean annotations
    tau: float,
    epsilon: float = None,  ## Set to None in FDR experiments
    min_loss: int = 0,
    soft: bool = False,  ## ALGO-3: Use soft sigmoid thresholding
    beta: float = 50.0,  ## ALGO-3: Sigmoid temperature for soft thresholding
):
    if soft:
        ## ALGO-3: Soft sigmoid-based thresholding for smooth risk surface
        diff = claim_scores - tau
        weights = 1.0 / (1.0 + np.exp(-beta * diff))  ## sigmoid(beta * (score - tau))
        total_weight = np.sum(weights)
        if total_weight > 0:
            weighted_false = np.sum((~annotations).astype(float) * weights)
            if epsilon is None:
                return weighted_false / total_weight
            else:
                return int((weighted_false / total_weight) > epsilon)
        else:
            return min_loss
    else:
        annotations_included = annotations[claim_scores >= tau]
        if epsilon is None:
            ## Default condition
            return (
                np.mean(~annotations_included)
                if len(annotations_included) > 0
                else min_loss
            )
        else:
            return (
                int(np.mean(~annotations_included) > epsilon)
                if len(annotations_included) > 0
                else min_loss
            )


"""Runs risk control for method_name and loss_name"""


def run_risk_control(
    claim_scores: List[np.ndarray],
    annotations: List[np.ndarray],
    taus_to_search,
    alpha,
    epsilon=None,
    method_name="gcrc",  ## "gcrc", "monotized_losses_crc", "standard_crc", "ltt"
    small_num_adjust=1e-10,
    n_grid=200,  ## Number of threshold to search, ## 1000
    B=1,  ## Maximum loss
    loss_name="loss_factuality_fdr",
    test_point_loss=0.0,  ## CODE-4: Loss assigned to the conservative test point (default 0 = use B)
    soft=False,  ## ALGO-3: Use soft sigmoid thresholding
    beta=50.0,  ## ALGO-3: Sigmoid temperature
):

    if len(claim_scores) != len(annotations):
        raise Exception(
            f"len(claim_scores) = {len(claim_scores)} != {len(annotations)} = len(annotations)"
        )

    n = len(claim_scores)

    taus_to_search = (
        np.unique(taus_to_search)[::-1] + small_num_adjust
    )  ## Sort descending (safest to most aggressive)
    num_taus_unique = len(taus_to_search)
    k = max(int(num_taus_unique / n_grid), 1)

    taus_to_search = taus_to_search[::k]

    risk_prev = 0.0
    tau_prev = 1.0 + small_num_adjust

    if taus_to_search[0] < 1.0:
        ## Make sure to include safe tau value
        taus_to_search = np.concatenate(([1.0], taus_to_search))

    if method_name == "standard_crc":
        ## Standard CRC implemented by searching from most-aggressive (smallest) hyperparameter to safest (largest)
        taus_to_search = taus_to_search[::-1]

    losses = np.zeros(n + 1)
    # CODE-4: Configurable test point loss. When test_point_loss=0 (default), uses B (conservative).
    # Set to alpha, alpha/2, or None to reduce conservatism.
    losses[n] = B if test_point_loss == 0.0 else test_point_loss  ## Conservative loss for test point
    risks = [risk_prev]

    for t, tau in enumerate((taus_to_search)):
        for i in range(n):
            if (
                method_name in ["gcrc", "standard_crc", "ltt"]
                or method_name == "monotized_losses_crc"
                and t == 0
            ):
                losses[i] = eval(loss_name)(
                    claim_scores[i], annotations[i], tau, epsilon=epsilon,
                    soft=soft, beta=beta
                )

            elif method_name == "monotized_losses_crc" and t > 0:
                ## If running monotized-losses CRC (Corollary 1 in Angelopoulos et al. (2024), loss is maximum seen so far for that point)
                losses[i] = max(
                    losses[i],
                    eval(loss_name)(
                        claim_scores[i], annotations[i], tau, epsilon=epsilon,
                        soft=soft, beta=beta
                    ),
                )

            else:
                raise Exception(f"method_name '{method_name}' not recognized!")

        if method_name == "ltt":
            ## For LTT, no conservative "test point loss"
            risk_curr = losses[:n].mean()
            p_val_curr = hb_p_value(risk_curr, n, alpha)

        else:
            risk_curr = losses.mean()

        risks.append(risk_curr)

        if method_name in ["gcrc", "monotized_losses_crc"] and risk_curr > alpha:
            ## Stopping condition for GCRC and monotized-losses CRC methods, return most recent tau prior to current
            break

        elif method_name == "ltt" and p_val_curr > alpha:
            ## Stopping condition for LTT is based on calculated p-value
            break

        elif method_name == "standard_crc" and risk_curr <= alpha:
            ## Stopping condition for standard CRC: return first identified safe hyperparam
            return tau, losses.mean(), risks

        else:
            risk_prev = risk_curr
            tau_prev = tau

    return tau_prev, risk_prev, risks


"""Runs one risk control trial for method_name"""


def run_rc_trial(
    x_arr,  ## List where each entry is an array of sub-claim scores for a response
    y_arr,  ## List where each entry is an array of sub-claim "oracle scores" or annotations
    z_arr,  ## List of features for the prompt and response used for conditional calibration (all ones for marginal cp)
    rng,
    method_name,
    alpha,
    epsilon,
    loss_name="loss_factuality_fdr",
    cal_frac=0.7,
    test_point_loss=0.0,  ## CODE-4: Loss for conservative test point (default 0 -> use B)
    soft=False,  ## ALGO-3: Soft sigmoid
    beta=50.0,  ## ALGO-3: Sigmoid temperature
):

    data_calib, data_test, idx_calib, idx_test = split_dataset(
        (x_arr, y_arr), rng, train_frac=cal_frac
    )  ## here "train_frac" is actually for cal

    taus_to_search = get_taus_grid_from_data(data_calib[0])

    ## Get selected threshold and risk for method_name
    threshold, _, risks = run_risk_control(
        *data_calib,
        taus_to_search=taus_to_search,
        alpha=alpha,
        method_name=method_name,
        loss_name=loss_name,
        n_grid=n_grid,
        B=B,
        test_point_loss=test_point_loss,
        soft=soft,
        beta=beta,
    )
    constraint_violations = []
    claim_perc = []

    for i, j in enumerate(idx_test):
        loss = eval(loss_name)(
            data_test[0][i], data_test[1][i], tau=threshold, epsilon=epsilon
        )

        # valid_inds.append(threshold >= scores_test[i])
        constraint_violations.append(loss)

        claim_perc.append(
            get_frac_true_claims_retained(
                [data_test[0][i]], [data_test[1][i]], [threshold]
            )[0]
        )

    # valid_inds = np.asarray(valid_inds).flatten()
    constraint_violations = np.asarray(constraint_violations).flatten()
    claim_perc = np.asarray(claim_perc).flatten()

    return np.mean(constraint_violations), np.mean(claim_perc), risks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multistep FCS experiments.")

    parser.add_argument(
        "--score_names",
        nargs="+",
        help="Names of subclaim scores to run expts for, eg: logprobs frequency selfeval",
        required=True,
    )
    parser.add_argument(
        "--method_names",
        nargs="+",
        help="Names of methods scores to run expts for, eg: monotized_losses_crc ltt gcrc",
        required=True,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/drewprinster/conformal-safety/data",
        help="Path to data directory.",
    )
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials")
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.005,
        help="Minimum alpha to run experiments on",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.1,
        help="Maximum alpha to run experiments on",
    )
    parser.add_argument(
        "--alpha_inc", type=float, default=0.005, help="Increment in alphas grid"
    )
    parser.add_argument(
        "--cal_frac",
        type=float,
        default=0.7,
        help="Fraction of labeled data used for calibration (vs training propper)",
    )
    parser.add_argument(
        "--n_grid",
        type=int,
        default=200,
        help="Number of lambdas in grid to search over",
    )
    parser.add_argument(
        "--B", type=float, default=1.0, help="Upper bound on loss functions"
    )
    parser.add_argument(
        "--soft", action="store_true", default=False,
        help="ALGO-3: Use soft sigmoid thresholding in FDR loss"
    )
    parser.add_argument(
        "--beta", type=float, default=50.0,
        help="ALGO-3: Sigmoid temperature for soft thresholding"
    )
    parser.add_argument(
        "--blend_weights", type=float, nargs=3, default=None,
        metavar=("W_SELF", "W_FREQ", "W_LOGPROB"),
        help="ALGO-2: Weights for score blending (selfevals, frequency, logprobs). Default: 0.5 0.3 0.2"
    )
    parser.add_argument(
        "--test_point_loss",
        type=float,
        default=None,
        help="Loss for conservative test point (default None = use B). Set to 0 to remove, alpha for less conservative.",
    )

    method_names = ["monotized_losses_crc", "ltt", "gcrc"]

    args = parser.parse_args()
    score_names = [str(s) for s in args.score_names]
    method_names = [str(m) for m in args.method_names]
    data_path = args.data_path
    n_trials = args.n_trials
    alpha_min = args.alpha_min
    alpha_max = args.alpha_max
    alpha_inc = args.alpha_inc
    cal_frac = args.cal_frac
    n_grid = args.n_grid
    B = args.B
    test_point_loss = args.test_point_loss if args.test_point_loss is not None else 0.0  ## CODE-4: 0.0 means use B
    soft = args.soft  ## ALGO-3
    beta = args.beta  ## ALGO-3

    epsilon = None  # 0.1

    # src_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'src')
    # sys.path.append(src_dir)

    ## Loading datasets
    orig_datasets = {}
    suffix = ".jsonl"
    dataset_dir = f"{data_path}/MedLFQAv2"  # "/Users/cherian/Projects/conformal-safety/data/MedLFQAv2"
    for path in os.listdir(dataset_dir):
        dataset_name = path[: -len(suffix)]
        if not path.startswith("."):
            with open(os.path.join(dataset_dir, path), "r") as fp:
                orig_datasets[dataset_name] = [
                    json.loads(line) for line in fp.readlines()
                ]

    ## Dictionary where can search using quesiton and get name of original dataset it came from
    dataset_lookup = {}
    for name, data in orig_datasets.items():
        for dat in data:
            dataset_lookup[dat["Question"]] = name

    ## Getting scores
    # data_path = "/home/drewprinster/conformal-safety/data" #"/Users/cherian/Projects/conformal-safety/data"
    dataset_path = os.path.join(data_path, "medlfqa_dataset.pkl")
    freq_path = os.path.join(data_path, "medlfqa_frequencies.npz")
    logprob_path = os.path.join(data_path, "medlfqa_logprobs.npz")
    selfeval_path = os.path.join(data_path, "medlfqa_selfevals.npz")

    with open(dataset_path, "rb") as fp:
        dataset = pickle.load(fp)

    ## HARD CODED FIX FOR REDUNDANT PROMPT...which has atomic_facts assigned to the wrong redundancy
    dataset[1132]["atomic_facts"] = dataset[1048]["atomic_facts"]

    frequencies = np.load(freq_path)
    logprobs = np.load(logprob_path)
    selfevals = np.load(selfeval_path)

    drop_prompts = []
    for k in frequencies:
        if frequencies[k].ndim != 1:
            drop_prompts.append(k)
        elif np.allclose(selfevals[k], -1):
            drop_prompts.append(k)
        elif k not in logprobs:
            drop_prompts.append(k)
        elif remove_specific_leading_chars(k).strip() not in dataset_lookup:
            drop_prompts.append(k)

    # drop and match ordering of dataset
    dataset = [
        dat for dat in dataset if dat["prompt"] not in drop_prompts
    ]  ## List where each entry is a dictionary with 'prompt', 'response', and 'atomic_facts'
    full_dataset = dataset
    prompts_to_keep = [
        dat["prompt"] for dat in dataset
    ]  ## List where each entry is the full prompt
    names_to_keep = [
        p.split("about")[-1].strip()[:-1] for p in prompts_to_keep
    ]  ## List where each entry is an abbreviated prompt for a name

    ## Lists where each entry in the list is an array of scores for subclaims
    frequencies_arr = [frequencies[p] for p in prompts_to_keep]  ## Frequency scoring
    selfevals_arr = [selfevals[p] for p in prompts_to_keep]  ## Self-evaluation scoring
    logprobs_arr = [logprobs[p] for p in prompts_to_keep]  ## Log-probability scoring
    annotations_arr = [
        np.asarray([af["is_supported"] for af in dat["atomic_facts"]])
        for dat in dataset
    ]  ## Oracle (annotation) scoring
    ordinal_arr = [
        np.arange(len(f)) for f in frequencies_arr
    ]  ## Ordinal scoring (baseline)

    print(
        f"dataset lengths: dataset {len(dataset)}, annotations {len(annotations_arr)}, frequencies {len(frequencies_arr)}, selfevals {len(selfevals_arr)}, logprobs {len(logprobs_arr)}"
    )

    # get prompt-level features

    # names of datasets
    dataset_arr = [
        dataset_lookup[remove_specific_leading_chars(p).strip()]
        for p in prompts_to_keep
    ]  ## List of original dataset name for each question
    dataset_dummies = pd.get_dummies(
        dataset_arr
    )  ## Pandas dataframe with dummies indicating dataset origin name for each question
    dataset_names = [
        name[:-3] if name.endswith("_qa") else name for name in dataset_dummies.columns
    ]  ## Unique list of (five) dataset names

    # length of response
    response_len_arr = [len(dat["response"]) for dat in dataset]
    response_len_arr = np.asarray(response_len_arr).reshape(
        -1, 1
    )  ## Array where each entry is the string length of the response to each question

    # length of prompt
    prompt_len_arr = [
        len(remove_specific_leading_chars(dat["prompt"]).strip()) for dat in dataset
    ]
    prompt_len_arr = np.asarray(prompt_len_arr).reshape(
        -1, 1
    )  ## Array where each entry is the string length of the prompt/question

    # mean (exponentiated) logprob
    logprobs_mean_arr = [np.mean(arr) for arr in logprobs_arr]
    logprobs_mean_arr = np.asarray(
        logprobs_mean_arr
    ).reshape(
        -1, 1
    )  ## Array where each entry is the mean log-probability (over subclaims) for each response

    # std (exponentiated) logprob
    logprobs_std_arr = [np.std(arr) for arr in logprobs_arr]
    logprobs_std_arr = np.asarray(
        logprobs_std_arr
    ).reshape(
        -1, 1
    )  ## Array where each entry is the std log-probability (over subclaims) for each response

    z_ones = np.ones(
        (len(frequencies_arr), 1)
    )  ## Array of ones equal in length to dataset
    z_arr = np.concatenate(
        (z_ones, response_len_arr, prompt_len_arr, logprobs_mean_arr, logprobs_std_arr),
        axis=1,
    )
    z_dummies = dataset_dummies.to_numpy()

    ## Array of features computed using prompt and response (X_i) in paper
    ## Cols: (0) ones, (1) response len, (2) prompt len, (3) mean log-prob over subclaims, (4) std log-prob over subclaims
    ## (5, 6, 7, 8) indicators for 4 of 5 datasets (last one is indicated by first 4 being false)
    z_arr_dummies = np.concatenate((z_arr, z_dummies[:, :-1]), axis=1)

    # print(len(dataset_arr), len(response_len_arr), len(prompt_len_arr), len(logprobs_mean_arr), len(logprobs_std_arr))

    rng = np.random.default_rng(seed=0)
    alphas = np.arange(alpha_min, alpha_max + alpha_inc, alpha_inc)

    print(f"Running with alphas : {alphas}")

    risk_dict = {}
    claims_dict = {}

    # score_names = ["logprobs", "frequency", "selfevals"] #"frequency", "selfevals",
    # ALGO-2: Compute blended score combining selfevals, frequency, and logprobs
    # Normalize each score type to [0,1] then combine with learned/guided weights
    def normalize_scores(score_arr):
        all_scores = np.concatenate(score_arr)
        s_min, s_max = np.min(all_scores), np.max(all_scores)
        if s_max > s_min:
            return [(s - s_min) / (s_max - s_min) for s in score_arr]
        return score_arr

    selfevals_norm = normalize_scores(selfevals_arr)
    freq_norm = normalize_scores(frequencies_arr)
    logprobs_norm = normalize_scores(logprobs_arr)

    # ALGO-2: Default weights (can be overridden via --blend_weights)
    blend_w = getattr(args, "blend_weights", None)
    if blend_w is None:
        w_self, w_freq, w_logp = 0.5, 0.3, 0.2  # default: selfeval-dominant
    else:
        w_self, w_freq, w_logp = blend_w

    blended_arr = [
        w_self * s + w_freq * f + w_logp * l
        for s, f, l in zip(selfevals_norm, freq_norm, logprobs_norm)
    ]

    score_arr_dict = {
        "logprobs": logprobs_arr,
        "frequency": frequencies_arr,
        "selfevals": selfevals_arr,
        "blended": blended_arr,  # ALGO-2: Combined score
    }
    # method_names = ["monotized_losses_crc", "ltt", "gcrc"] # "standard_crc", "gcrc", "monotized_losses_crc", "monotized_losses_crc", "ltt",
    loss_name = "loss_factuality_fdr"  # "loss_factuality_fdr"

    for s_i, score_name in enumerate(score_names):
        subclaim_scores_arr = score_arr_dict[score_name]

        print(f"Running experiments for {score_name} scoring...")
        for method_name in method_names:  # "gcrc", "monotized_losses_crc",
            print(f"\n{method_name}")
            risk_dict[method_name] = pd.DataFrame(
                np.c_[alphas, np.zeros(len(alphas)), np.zeros(len(alphas))],
                columns=["alphas", "risk_mean", "risk_std"],
            )
            claims_dict[method_name] = pd.DataFrame(
                np.c_[alphas, np.zeros(len(alphas)), np.zeros(len(alphas))],
                columns=["alphas", "claims_mean", "claims_std"],
            )

            for a, alpha in enumerate(alphas):
                risks = np.zeros(n_trials)
                claims = np.zeros(n_trials)

                # CODE-2: Increased jitter magnitude (1e-6) for effective tie-breaking.
                # Add deterministic tie-breaker by subclaim index when scores are equal within eps.
                subclaim_scores_arr_jitter = []
                for subclaim_scores in subclaim_scores_arr:
                    scores_jittered = subclaim_scores + rng.uniform(low=0, high=1e-6, size=subclaim_scores.shape)
                    # Deterministic tie-breaking: add tiny index-based offset for equal scores
                    eps = 1e-12
                    tie_break = np.arange(len(scores_jittered)) * 1e-14
                    scores_jittered = scores_jittered + tie_break
                    scores_jittered = np.minimum(1.0 - 1e-12, scores_jittered)
                    subclaim_scores_arr_jitter.append(scores_jittered)

                rng = np.random.default_rng(seed=1)

                print(f"alpha : {alpha}")
                for trial in tqdm(range(n_trials)):
                    risk, claim_perc, _ = run_rc_trial(
                        subclaim_scores_arr_jitter,
                        annotations_arr,
                        z_arr_dummies,
                        rng,
                        method_name=method_name,
                        alpha=alpha,
                        epsilon=epsilon,
                        loss_name=loss_name,
                        cal_frac=cal_frac,
                        test_point_loss=test_point_loss,
                        soft=soft,
                        beta=beta,
                    )
                    # print(f"constraint_violations : {constraint_violations}")
                    # print(f"claim_perc : {claim_perc}")
                    risks[trial] = risk
                    claims[trial] = claim_perc

                risk_dict[method_name].iloc[a, 1] = np.mean(risks)
                risk_dict[method_name].iloc[a, 2] = np.std(risks)
                claims_dict[method_name].iloc[a, 1] = np.mean(claims)
                claims_dict[method_name].iloc[a, 2] = np.std(claims)

        method_name_map = {
            "gcrc": "GCRC (proposed)",
            "monotized_losses_crc": "Monotized-losses CRC \n(Mohri & Hashimoto, 2024; \nAngelopoulos et al., 2024)",
            "standard_crc": "standard CRC (Angelopoulos, et al., 2024)",
            "ltt": "LTT (Angelopoulos, et al., 2025)",
        }

        # colors_dict = {'gcrc' : BLUE, 'monotized_losses_crc' : 'C1', 'ltt' : 'C2', 'standard_crc' : 'C6'}
        # markers_dict = {'gcrc' : 'o', 'monotized_losses_crc' : 's', 'ltt' : '^', 'standard_crc' : 'X'}

        results_df = pd.DataFrame()
        results_df["alphas"] = alphas

        for metric in ["risk", "claims"]:
            metric_dict = eval(f"{metric}_dict")
            for m, method_name in enumerate(method_names):
                alphas = metric_dict[method_name]["alphas"]
                metric_mean = metric_dict[method_name][f"{metric}_mean"]
                metric_std = metric_dict[method_name][f"{metric}_std"]
                results_df[f"{method_name}_{metric}_mean"] = metric_mean
                results_df[f"{method_name}_{metric}_std"] = metric_std

        results_df.to_csv(
            f"{loss_name}Loss_{score_name}Scoring_{n_trials}trials_{n_grid}ngrid.csv"
        )
