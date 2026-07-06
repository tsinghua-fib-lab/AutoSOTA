import nashpy
import numpy as np
import pyspiel
import timeout_decorator
from interface.strategy import ProbabilitySimplexStrategySpace
from open_spiel.python.algorithms import lp_solver

from games.base_game import BaseGame


def compute_game_value(payoff_matrix):
    """
    ゼロサムゲームのゲーム値を計算する関数。

    Args:
        payoff_matrix (list of list or numpy.ndarray): プレイヤー1の利得行列。

    Returns:
        float: ゲームの値（プレイヤー1の期待利得）。
    """
    # プレイヤー2の利得行列はゼロサムゲームでは - プレイヤー1の利得行列
    neg_payoff_matrix = -1 * payoff_matrix

    # ゲームの定義
    game = pyspiel.create_matrix_game(
        payoff_matrix,  # プレイヤー1の利得行列
        neg_payoff_matrix,  # プレイヤー2の利得行列
    )

    # ナッシュ均衡を計算し、ゲームの値を取得
    st1, st2, _, _ = lp_solver.solve_zero_sum_matrix_game(game)
    st1_np = np.array(st1).flatten()
    st2_np = np.array(st2).flatten()
    game_value_1 = payoff_matrix @ st2_np @ st1_np
    game_value_2 = neg_payoff_matrix.T @ st1_np @ st2_np
    return [game_value_1, game_value_2]


class ZeroSumMatrixGame(BaseGame):
    def __init__(self, payoff, game_value=None):
        self.payoff = payoff
        if game_value is not None:
            self.game_values = [game_value, -game_value]
        else:
            try:
                # self.game_values = self.calc_game_value(payoff)
                self.game_values = compute_game_value(payoff)
                print("Game value:", self.game_values)
            except Exception:
                print("Calculating game_values is time out.")
                self.game_values = []

    @timeout_decorator.timeout(5)
    def calc_game_value(self, payoff):
        game = nashpy.Game(payoff, payoff * -1)
        equilibrias = game.support_enumeration()
        game_values = []
        for eq in equilibrias:
            game_values.append(payoff @ eq[1] @ eq[0])
            game_values.append(-payoff.T @ eq[0] @ eq[1])
            break
        return game_values

    def num_players(self):
        return 2

    def strategy_classes(self):
        return [
            ProbabilitySimplexStrategySpace(self.payoff.shape[player_id]) for player_id in range(self.num_players())
        ]

    def num_actions(self, player_id):
        return self.payoff.shape[player_id]

    def full_feedback(self, strategies):
        return [self.payoff @ strategies[1], -self.payoff.T @ strategies[0]]

    def nash_conv(self, strategies):
        return max(self.payoff @ strategies[1]) + max(-self.payoff.T @ strategies[0])

    def individual_regrets(self, strategies):
        return [
            max(self.payoff @ strategies[1]) - (self.payoff @ strategies[1]) @ strategies[0],
            max(-self.payoff.T @ strategies[0]) - (-self.payoff.T @ strategies[0]) @ strategies[1],
        ]

    def individual_nash_convs(self, strategies):
        return (
            [
                max(-self.payoff.T @ strategies[0]) - self.game_values[1],
                max(self.payoff @ strategies[1]) - self.game_values[0],
            ]
            if self.game_values
            else [None, None]
        )


def biased_rps():
    payoff = np.array([[0, -1, 3], [1, 0, -1], [-3, 1, 0]], dtype=np.float64)
    nash_eq = np.array([[1 / 5, 3 / 5, 1 / 5], [1 / 5, 3 / 5, 1 / 5]])
    game_value = nash_eq[0] @ payoff @ nash_eq[1]
    return ZeroSumMatrixGame(payoff, game_value), nash_eq


def biased_matching_pennies():
    payoff = np.array([[-1 / 3, 2 / 3], [2 / 3, -1]], dtype=np.float64)
    nash_eq = np.array([[5 / 8, 3 / 8], [5 / 8, 3 / 8]])
    game_value = nash_eq[0] @ payoff @ nash_eq[1]
    return ZeroSumMatrixGame(payoff, game_value), nash_eq


def m_ne():
    payoff = -np.array(
        [
            [0, -1, 1, 0, 0],
            [1, 0, -1, 0, 0],
            [-1, 1, 0, 0, 0],
            [-1, 1, 0, 2, -1],
            [-1, 1, 0, -1, 2],
        ],
        dtype=np.float64,
    )
    nash_eq = np.array([[1 / 3, 1 / 3, 1 / 3, 0, 0], [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]])
    game_value = nash_eq[0] @ payoff @ nash_eq[1]
    return ZeroSumMatrixGame(payoff, game_value), nash_eq
