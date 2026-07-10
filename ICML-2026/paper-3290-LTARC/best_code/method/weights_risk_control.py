import numpy as np
import pandas as pd


class Weights:
    """Base class for computing weighted objectives and constraints with conformal risk control."""

    def __init__(self, Z_0, Z_1, alpha):
        self.Z_0 = Z_0
        self.Z_1 = Z_1
        self.alpha = alpha

    def get_obj_and_constr(self, a_policy):
        Z = np.where(a_policy == 0, self.Z_0, self.Z_1)
        obj = np.mean(Z)

        prop = np.mean(a_policy)
        if prop == 0:
            return obj, 1, len(Z)

        Z = np.where(a_policy == 0, 0, Z) / prop
        constr = np.mean(Z)

        if np.mean(Z) == 0:
            constr = 1

        return obj, constr, len(Z)

    def get_b(self, a_policy):
        Z_max = 500

        if np.mean(a_policy) == 0:
            return Z_max, 0

        prop = np.mean(a_policy)
        Z = np.where(a_policy == 0, 0, self.Z_1) / prop

        Z_alpha = self.get_conformal_weight(Z, Z_max)

        return Z_alpha, Z[-1]

    def get_conformal_weight(self, Z, Z_max):
        n = len(Z)
        Z = np.sort(Z)
        ind = int(np.ceil((1 - self.alpha)*(n+1)))-1
        if ind >= n:
            return Z_max
        return Z[ind]


class WeightsDecision(Weights):
    """Weights for observational data using sensitivity model with parameter gamma."""

    def __init__(self, df, x_names, gamma, get_p_a_x, alpha):

        self.df = df.copy().reset_index(drop=True)
        self.x_names = x_names
        assert "y" in df.columns, "No 'y' column in df"
        assert "a" in df.columns, "No 'a' column in df"
        for x in x_names:
            assert x in df.columns, "No {} column in df".format(x)

        self.gamma = gamma

        df = self.add_weights(get_p_a_x, gamma)

        super().__init__(df['Z_0'], df['Z_1'], alpha)

    def add_weights(self, get_p_a_x, gamma):
        def get_weight(p_a_x, a, gamma_i):
            weight = 1 + gamma_i * (1 / p_a_x - 1)
            return np.where(self.df['a'] == a, weight, 0)

        def get_weights(a):
            p_a_x = get_p_a_x(self.df[self.x_names], a)

            weight_low = get_weight(p_a_x, a, 1 / gamma)
            weight_high = get_weight(p_a_x, a, gamma)

            weight = np.where(self.df['y'] == 1, weight_high, weight_low)
            return weight
        df = pd.DataFrame()
        for a in [0, 1]:
            df[f'weight_{a}'] = get_weights(a)
            df[f'Z_{a}'] = df[f'weight_{a}'] * self.df['y']

        return df


class WeightsSampling(Weights):
    """Weights for data with selection bias, combining treatment and sampling sensitivity."""

    def __init__(self, df, x_names, gamma_a, gamma_s, alpha, get_p_s1, get_p_a_x, get_p_s_x):

        self.df = df.copy().reset_index(drop=True)
        self.x_names = x_names
        assert "y" in df.columns, "No 'y' column in df"
        assert "a" in df.columns, "No 'a' column in df"
        for x in x_names:
            assert x in df.columns, "No {} column in df".format(x)

        self.gamma_a = gamma_a
        self.gamma_s = gamma_s

        self.p_s1 = get_p_s1()

        df = self.add_weights(get_p_a_x, get_p_s_x, gamma_a, gamma_s)

        super().__init__(df['Z_0'], df['Z_1'], alpha)

    def add_weights(self, get_p_a_x, get_p_s_x, gamma_a, gamma_s):
        def get_weight(p_a_x, p_s1_x, a, gamma_a_i, gamma_s_i):
            weight = gamma_s_i * ((1 - p_s1_x) / p_s1_x * self.p_s1 / (1 - self.p_s1)) * (
                    1 + gamma_a_i * (1 / p_a_x - 1))
            return np.where(self.df['a'] == a, weight, 0)

        def get_weights(a):
            p_a_x = get_p_a_x(self.df[self.x_names], a)
            p_s1_x = get_p_s_x(self.df[self.x_names], 1)

            weight_low = get_weight(p_a_x, p_s1_x, a, 1 / gamma_a, 1 / gamma_s)
            weight_high = get_weight(p_a_x, p_s1_x, a, gamma_a, gamma_s)

            weight = np.where(self.df['y'] == 1, weight_high, weight_low)
            return weight

        df = pd.DataFrame()
        for a in [0, 1]:
            df[f'weight_{a}'] = get_weights(a)
            df[f'Z_{a}'] =df[f'weight_{a}'] * self.df['y']

        return df['Z_0'], df['Z_1']
