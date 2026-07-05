from games.matrix_game import ZeroSumMatrixGame

from runner.logger import Logger


def run(
    game,
    num_iters,
    players,
    stop_early=False,
    print_interval=100000,
    log_interval=10000,
):
    n_players = game.num_players()

    log = Logger()
    for t in range(num_iters):
        for i in range(n_players):
            strategies = [player.strategy.copy() for player in players]
            gradient = game.full_feedback(strategies)
            players[i].add_gradient(gradient[i])
            players[i].calc_strategy()

        if t % print_interval == 0 or t == num_iters - 1:
            print_info(t, game, players)

        if stop_early and isinstance(game, ZeroSumMatrixGame):
            regularized_gap_last_iterate = players[0].regularized_gap(
                game.payoff @ players[1].strategy, players[0].strategy
            ) + players[1].regularized_gap(-game.payoff.T @ players[0].strategy, players[1].strategy)
            if regularized_gap_last_iterate < 1e-13:
                print_info(t, game, players)
                record_log(log, t, game, players)
                break

        # record log
        if t < log_interval or t % log_interval == 0 or t == num_iters - 1:
            record_log(log, t, game, players)

    return log


def record_log(log, t, game, players):
    n_players = len(players)
    strategies = [player.strategy for player in players]
    if isinstance(game, ZeroSumMatrixGame):
        nash_conv_last_iterate = game.nash_conv(strategies)
        nash_conv_time_average = game.nash_conv([players[i].average_strategy for i in range(len(players))])
        individual_nash_convs = game.individual_nash_convs(strategies)
        individual_regrets = game.individual_regrets(strategies)
    else:
        policy = players[0].policy
        nash_conv_last_iterate = game.nash_conv(policy)
        nash_conv_time_average = None
        individual_nash_convs = game.individual_nash_convs(policy)
        individual_regrets = game.individual_regrets(players[0].policy)
    log["t"].append(t)
    log["nash_conv_last_iterate"].append(nash_conv_last_iterate)
    log["nash_conv_time_average"].append(nash_conv_time_average)
    for i in range(n_players):
        log[f"player{i}_individual_regret"].append(individual_regrets[i])
        log[f"player{i}_individual_nash_conv"].append(individual_nash_convs[i])


def print_info(t, game, players):
    print(f"---Run Iteration {t}---")
    strategies = [player.strategy for player in players]
    if isinstance(game, ZeroSumMatrixGame):
        nash_conv_last_iterate = game.nash_conv(strategies)
        regularized_gap_last_iterate = players[0].regularized_gap(
            game.payoff @ players[1].strategy, players[0].strategy
        ) + players[1].regularized_gap(-game.payoff.T @ players[0].strategy, players[1].strategy)
        nash_conv_time_average = game.nash_conv([players[i].average_strategy for i in range(len(players))])
        individual_nash_convs = game.individual_nash_convs(strategies)
        individual_regrets = game.individual_regrets(strategies)
    else:
        policy = players[0].policy
        nash_conv_last_iterate = game.nash_conv(policy)
        nash_conv_time_average = None
        regularized_gap_last_iterate = None
        individual_nash_convs = game.individual_nash_convs(policy)
        individual_regrets = game.individual_regrets(players[0].policy)
    print(f"Nash conv of last-iterate strategies : {nash_conv_last_iterate}")
    print(f"Regularized gap of last-iterate strategies : {regularized_gap_last_iterate}")
    print(f"Nash conv of average strategies : {nash_conv_time_average}")
    print(f"Individual Nash convs: {individual_nash_convs}")
    print(f"Individual Regrets: {individual_regrets}")
    print(strategies)
