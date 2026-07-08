import itertools
import json
import os
import time
from datetime import datetime
from statistics import mean, stdev

from joblib import Parallel, delayed
from tqdm import tqdm

from src.envs.gridworld import Gridworld
from src.mixed_delayed_rr import Mixed_Delayed_RR
from src.performative_pepg import PePG
from src.performative_prediction import Performative_Prediction
from src.utils import *


def generate_data(params):
    print("Begin generating performative prediction data\n")
    start = time.time()

    compare_algorithms = params.get("compare_algorithms", False)
    multi_agent = params.get("multi_agent", True)

    use_pepg = params.get("use_pepg", False)
    use_mdrr = params.get("use_mdrr", False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_filename = f"data/outputs_{timestamp}.json"

    os.makedirs("data", exist_ok=True)

    if compare_algorithms:
        print(f"Running algorithm comparison in {'multi-agent'} environment...")

        outputs = _run_multi_agent_comparison(params)

        data_filename = f"data/outputs_comparison_{timestamp}.json"
        with open(data_filename, "w") as f:
            json.dump(outputs, f, indent=4)

        with open("data/latest_output.txt", "w") as f:
            f.write(data_filename)

    else:
        if use_mdrr:
            print("Running MDRR in multi-agent environment...")
            outputs = _run_multi_agent_mdrr(params)
        elif use_pepg:
            print("Running PePG in multi-agent environment...")
            outputs = _run_multi_agent_pepg(params)
        else:
            print("Running traditional PG in multi-agent environment...")
            outputs = _run_multi_agent_pg(params)

        data_filename = f"data/outputs_{timestamp}.json"
        with open(data_filename, "w") as f:
            json.dump(outputs, f, indent=4)

        with open("data/latest_output.txt", "w") as f:
            f.write(data_filename)

    end = time.time()
    print(f"Time elapsed: {end - start}")
    print(f"Data saved to: {data_filename}")
    print("Finish generating data\n")

    return


def _run_multi_agent_comparison(params):
    print("Running multi-agent PG experiments...")
    pg_results = _run_multi_agent_pg(params)

    print("Running multi-agent PePG experiments...")
    pepg_results = _run_multi_agent_pepg(params)

    print("Running multi-agent MDRR experiments...")
    mdrr_results = _run_multi_agent_mdrr(params)

    return {"pg": pg_results, "pepg": pepg_results, "mdrr": mdrr_results}


def _run_multi_agent_mdrr(params):
    return _generate_data_multiagent(params, use_mdrr=True)


def _run_multi_agent_pepg(params):
    return _generate_data_multiagent(params, use_pepg=True)


def _run_multi_agent_pg(params):
    return _generate_data_multiagent(params, use_pepg=False)


def _generate_data_multiagent(params, use_pepg=False, use_mdrr=False):
    gradient = params["gradient"]
    sampling = params["sampling"]
    n_jobs = params["n_jobs"]
    eps = params["eps"]
    fbeta = params["fbeta"]
    betas = params["betas"]
    fgamma = params["fgamma"]
    gammas = params["gammas"]
    num_followers = params["num_followers"]
    max_iterations = params["max_iterations"]
    flamda = params["flamda"]
    lamdas = params["lamdas"]
    freg = params["freg"]
    regs = params["regs"]
    feta = params["feta"]
    etas = params["etas"]
    seeds = params["seeds"]
    fn_sample = params["fn_sample"]
    n_samples = params["n_samples"]
    policy_gradient = params["policy_gradient"]
    nus = params["nus"]
    unregularized_obj = params["unregularized_obj"]
    lagrangian = params["lagrangian"]
    N = params["N"]
    delta = params["delta"]
    B = params["B"]
    warmup = params["warmup"]
    v = params.get("v", [1.1])  # default v for MDRR

    configs = []
    if (
        not gradient
        and not sampling
        and not policy_gradient
        and not use_mdrr
        and not unregularized_obj
    ):
        # iterate lamdas
        for lamda in lamdas:
            configs.append(
                {"beta": fbeta, "lamda": lamda, "gamma": fgamma, "reg": freg}
            )
        # iterate betas
        for beta in betas:
            configs.append(
                {"beta": beta, "lamda": flamda, "gamma": fgamma, "reg": freg}
            )
        # iterate regs
        for reg in regs:
            configs.append(
                {"beta": fbeta, "lamda": flamda, "gamma": fgamma, "reg": reg}
            )
        # iterate gammas
        for gamma in gammas:
            configs.append(
                {"beta": fbeta, "lamda": flamda, "gamma": gamma, "reg": freg}
            )
        # iterate gammas and lamdas
        for lamda, gamma in itertools.product(lamdas, gammas):
            configs.append({"beta": fbeta, "lamda": lamda, "gamma": gamma, "reg": freg})
    if gradient:
        # iterate etas
        assert freg == "L2"
        for eta in etas:
            if sampling:
                configs.append(
                    {
                        "beta": fbeta,
                        "lamda": flamda,
                        "gamma": fgamma,
                        "reg": freg,
                        "eta": eta,
                        "n_sample": fn_sample,
                    }
                )
            else:
                configs.append(
                    {
                        "beta": fbeta,
                        "lamda": flamda,
                        "gamma": fgamma,
                        "reg": freg,
                        "eta": eta,
                    }
                )
    if sampling:
        # iterate n_samples
        for n_sample in n_samples:
            if gradient:
                configs.append(
                    {
                        "beta": fbeta,
                        "lamda": flamda,
                        "gamma": fgamma,
                        "reg": freg,
                        "eta": feta,
                        "n_sample": n_sample,
                    }
                )
            else:
                configs.append(
                    {
                        "beta": fbeta,
                        "lamda": flamda,
                        "gamma": fgamma,
                        "reg": freg,
                        "n_sample": n_sample,
                    }
                )

    if use_mdrr:
        # iterate nus and v values for MDRR
        for w in warmup:
            for eta in etas:
                for nu in nus:
                    for v_val in v:
                        configs.append(
                            {
                                "beta": fbeta,
                                "lamda": flamda,
                                "gamma": fgamma,
                                "reg": freg,
                                "eta": eta,
                                "nu": nu,
                                "warmup": w,
                                "v": v_val,
                            }
                        )
    elif policy_gradient or use_pepg:
        # iterate nus
        for w in warmup:
            for eta in etas:
                for nu in nus:
                    configs.append(
                        {
                            "beta": fbeta,
                            "lamda": flamda,
                            "gamma": fgamma,
                            "reg": freg,
                            "eta": eta,
                            "nu": nu,
                            "warmup": w,
                        }
                    )

    if unregularized_obj:
        # iterate lamdas
        for lamda in lamdas:
            configs.append(
                {"beta": fbeta, "lamda": lamda, "gamma": fgamma, "reg": freg}
            )

    configs = [dict(tup) for tup in set(tuple(d.items()) for d in configs)]

    outputs = []
    configs = sorted(configs, key=lambda d: d.get("n_sample", 0))
    for config in configs:
        output = {k: v for k, v in config.items()}
        algo_name = (
            "Multi-Agent MDRR"
            if use_mdrr
            else ("Multi-Agent PePG" if use_pepg else "Multi-Agent PG")
        )
        with tqdm_joblib(
            tqdm(
                desc=f"Executing {algo_name} for n_sample={config.get('n_sample', 'N/A')}",
                total=len(seeds),
            )
        ) as progress_bar:
            tmp_output = Parallel(n_jobs=min(n_jobs, len(seeds)))(
                delayed(execute_performative_prediction_multiagent)(
                    config,
                    eps,
                    num_followers,
                    max_iterations,
                    gradient,
                    sampling,
                    policy_gradient,
                    unregularized_obj,
                    lagrangian,
                    N,
                    delta,
                    B,
                    use_pepg,
                    use_mdrr,
                    seed,
                )
                for seed in seeds
            )
        d_diffs = [tmp_output[seed]["d_diff"] for seed in seeds]
        output["d_diff_mean"] = list(map(mean, zip(*d_diffs)))
        output["d_diff_std"] = list(map(stdev, zip(*d_diffs)))
        if gradient:
            sub_gaps = [tmp_output[seed]["sub_gap"] for seed in seeds]
            output["sub_gap_mean"] = list(map(mean, zip(*sub_gaps)))
            output["sub_gap_std"] = list(map(stdev, zip(*sub_gaps)))
        outputs.append(output)

        if any(
            "v_values" in tmp_output[i] and tmp_output[i]["v_values"]
            for i in range(len(seeds))
        ):
            v_values_list = [
                tmp_output[i]["v_values"]
                for i in range(len(seeds))
                if "v_values" in tmp_output[i] and tmp_output[i]["v_values"]
            ]
            if v_values_list:
                output["v_values_mean"] = list(map(mean, zip(*v_values_list)))
                output["v_values_std"] = list(map(stdev, zip(*v_values_list)))

    return outputs


def execute_performative_prediction_multiagent(
    config,
    eps,
    num_followers,
    max_iterations,
    gradient,
    sampling,
    policy_gradient,
    unregularized_obj,
    lagrangian,
    N,
    delta,
    B,
    use_pepg=False,
    use_mdrr=False,
    seed=1,
):
    beta = config["beta"]
    lamda = config["lamda"]
    gamma = config["gamma"]
    reg = config["reg"]
    warmup = config.get("warmup", 0)
    v_val = config.get("v", 1.1)

    if gradient or policy_gradient:
        eta = config["eta"]
    else:
        eta = None
    if policy_gradient:
        nu = config["nu"]
    else:
        nu = None
    if sampling and not lagrangian:
        n_sample = config.get("n_sample")
    elif sampling:
        n_sample = config.get("n_sample", 0) // 2
    else:
        n_sample = None

    env = Gridworld(beta, eps, gamma, num_followers, sampling, n_sample, seed)

    if use_mdrr:
        algorithm = Mixed_Delayed_RR(
            env,
            max_iterations,
            lamda,
            reg,
            gradient,
            eta,
            sampling,
            n_sample,
            policy_gradient,
            nu,
            unregularized_obj,
            lagrangian,
            N,
            delta,
            B,
            warmup=warmup,
            v=v_val,
            seed=seed,
        )
    elif use_pepg:
        algorithm = PePG(
            env,
            max_iterations,
            lamda,
            reg,
            gradient,
            eta,
            sampling,
            n_sample,
            policy_gradient,
            nu,
            unregularized_obj,
            lagrangian,
            N,
            delta,
            B,
            warmup=warmup,
            seed=seed,
        )
    else:
        # Use traditional PG
        algorithm = Performative_Prediction(
            env,
            max_iterations,
            lamda,
            reg,
            gradient,
            eta,
            sampling,
            n_sample,
            policy_gradient,
            nu,
            unregularized_obj,
            lagrangian,
            N,
            delta,
            B,
        )

    output = {k: v for k, v in config.items()}
    algorithm.execute()
    output["d_diff"] = algorithm.d_diff

    if use_mdrr:
        output["algorithm"] = "MDRR"
    elif use_pepg:
        output["algorithm"] = "PePG"
    else:
        output["algorithm"] = "PG"

    if gradient or policy_gradient or unregularized_obj:
        output["sub_gap"] = algorithm.sub_gap
    if hasattr(algorithm, "v_values"):
        output["v_values"] = algorithm.v_values

    vis = env._get_env_vis()
    if use_mdrr:
        algo_prefix = "mdrr"
    elif use_pepg:
        algo_prefix = "pepg"
    else:
        algo_prefix = "pg"

    config_name = (
        f"multi_{algo_prefix}_" + f"beta={beta}_lambda={lamda}_gamma={gamma}_reg={reg}"
    )
    if gradient:
        config_name += f"eta={eta}"
    if use_mdrr:
        config_name += f"_v={v_val}"
    if sampling:
        config_name += f"n_sample={n_sample}"
    # Always include seed to prevent file conflicts in parallel execution
    config_name += f"_seed={seed}"
    with open(f"limiting_envs/{config_name}.json", "w") as f:
        json.dump(vis, f, indent=4)

    return output
