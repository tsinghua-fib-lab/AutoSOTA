import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import fsspec
import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from runner import runner


def _build_game_and_players(cfg):
    game, nash_eq = instantiate(cfg.game[cfg.game.game_name])
    players = []
    strategy_spaces = game.strategy_classes()
    for i in range(game.num_players()):
        player_cfg = getattr(cfg, f"player{i}" if cfg.asymmetric_players else "player0")
        players.append(instantiate(player_cfg, strategy_spaces[i]))
    return game, nash_eq, players


def _game_param_str(cfg):
    # Exclude hydra control keys like _target_ from the formatted run name.
    params = {k: v for k, v in cfg.game[cfg.game.game_name].items() if not k.startswith("_")}
    return "_".join([f"{key}{value}" for key, value in params.items()])


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    start_time = time.time()
    np.random.seed(cfg.seed)

    game, _nash_eq, players = _build_game_and_players(cfg)

    alg_name = (
        ",".join([players[i].name() for i in range(game.num_players())])
        if cfg.asymmetric_players
        else players[0].name()
    )
    save_path = "{}/{}/{}".format(
        cfg.output_dir,
        cfg.game.game_name + "_" + _game_param_str(cfg),
        alg_name,
    )

    # run experiments
    print("==========Run experiment==========")
    max_workers = int(mp.cpu_count()) - 1
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        arguments = [[trial_id, cfg, save_path, np.random.randint(0, 2**32)] for trial_id in range(0, cfg.n_trials)]
        pool.map(run_experiment, *tuple(zip(*arguments, strict=True)))

    nash_conv_dfs = []
    for trial_id in range(cfg.n_trials):
        nash_conv_file = f"{save_path}/results_{trial_id}.csv"
        nash_conv_df = pd.read_csv(nash_conv_file, index_col=0)
        nash_conv_dfs.append(nash_conv_df)
    if len(nash_conv_dfs) > 1:
        nash_conv_df = pd.concat(nash_conv_dfs, axis=1)
        save_path_nash_conv_last_iterate = f"{save_path}/nash_conv_last_iterate.csv"
        with fsspec.open(save_path_nash_conv_last_iterate, "w") as f:
            nash_conv_df["nash_conv_last_iterate"].to_csv(f)
        save_path_player0_individual_nash_conv = f"{save_path}/player0_individual_nash_conv.csv"
        with fsspec.open(save_path_player0_individual_nash_conv, "w") as f:
            nash_conv_df["player0_individual_nash_conv"].to_csv(f)
        print(nash_conv_df)
        save_path_player1_individual_nash_conv = f"{save_path}/player1_individual_nash_conv.csv"
        with fsspec.open(save_path_player1_individual_nash_conv, "w") as f:
            nash_conv_df["player1_individual_nash_conv"].to_csv(f)
        save_path_nash_conv_time_average = f"{save_path}/nash_conv_time_average.csv"
        with fsspec.open(save_path_nash_conv_time_average, "w") as f:
            nash_conv_df["nash_conv_time_average"].to_csv(f)
    elapsed_time = time.time() - start_time
    print(save_path, f" elapsed_time:{elapsed_time}" + "[sec]")


def run_experiment(trial_id, cfg, save_path, seed):
    print(f"==========Start trial {trial_id}==========")
    np.random.seed(seed)

    game, _nash_eq, players = _build_game_and_players(cfg)

    log = runner.run(
        game,
        cfg.num_iters,
        players,
        cfg.stop_early,
        cfg.print_interval,
        cfg.log_interval,
    )

    # save log
    df = log.to_dataframe()
    df = df.set_index("t")
    print(f"==========Finish trial {trial_id}==========")
    save_file_path = f"{save_path}/results_{trial_id}.csv"
    with fsspec.open(save_file_path, "w") as f:
        df.to_csv(f)

    for i in range(game.num_players()):
        strategy = players[i].strategy
        np.savetxt(f"{save_path}/strategy{i}_{trial_id}.csv", strategy)
    return df


if __name__ == "__main__":
    main()
