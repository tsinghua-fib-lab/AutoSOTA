
import numpy as np
import pandas as pd
import logging

from abc import abstractmethod
from itertools import chain
from itertools import permutations
from itertools import product


class BuildPolicyHelper:
    def __init__(self):
        self.policy_temp = []
        self.active_criterias_temp = []
        self.loss_temp = []
        self.criteria_temp = []
        self.max_criteria_temp = []

    def add(self, best_policy, active_criterias_i, loss_min, criteria_i, max_criteria_i):
        self.policy_temp.append(best_policy)
        self.active_criterias_temp.append(active_criterias_i.copy())
        self.loss_temp.append(loss_min)
        self.criteria_temp.append(criteria_i)
        self.max_criteria_temp.append(max_criteria_i)

    def options_left(self):
        return len(self.policy_temp) > 0

    def get_first(self):
        policy_i = self.policy_temp[0]
        active_criterias_i = self.active_criterias_temp[0]
        loss_i = self.loss_temp[0]
        criteria_i = self.criteria_temp[0]
        max_criteria_i = self.max_criteria_temp[0]
        return policy_i, active_criterias_i, loss_i, criteria_i, max_criteria_i

    def remove_first(self):
        self.policy_temp.pop(0)
        self.active_criterias_temp.pop(0)
        self.loss_temp.pop(0)
        self.criteria_temp.pop(0)
        self.max_criteria_temp.pop(0)


class FinalPolicyHelper:
    def __init__(self, criteria_dict, constraint_value):
        self.policy_final_list = []
        self.criteria_dict = criteria_dict
        self.constraint_value = constraint_value

    def add(self, policy_i, loss_i, constraint_i, max_constraint_i):
        self.policy_final_list.append((policy_i, loss_i, constraint_i, max_constraint_i))

    def logg_all_for_debugging(self):
        logging.debug('-----')
        for policy_i, loss_i, constraint_i, max_constraint_i in self.policy_final_list:
            logging.debug('Potential final policy:')
            for key, exit_d in policy_i[:-1]:
                feature, bins, labels, name = self.criteria_dict[key]
                logging.debug('{}, exit: {}'.format(name, exit_d))
            logging.debug('Reward: {}'.format(loss_i))
            logging.debug('Constraint: {}'.format(constraint_i))
            logging.debug('Max constraint: {}'.format(max_constraint_i))
            logging.debug('------')

    def get_best_policy(self):
        best_loss = 10000
        best_policy = []
        for policy_i, loss_i, constraint_i, max_constraint_i in self.policy_final_list:
            if loss_i < best_loss and constraint_i < self.constraint_value:
                best_loss = loss_i
                best_policy = policy_i
                constraint = constraint_i
                max_constraint = max_constraint_i
        return best_policy, best_loss, constraint, max_constraint


class FftPolicyBase:
    """Base class for building Fast-and-Frugal Tree (FFT) policies with constraints."""

    def __init__(self, df, weighter, constraint_value, features, n_bins):
        assert "a" in df.columns, "No 'a' column in df"
        for x in features:
            assert x in df.columns, "No {} column in df".format(x)

        self.features = features
        self.weighter = weighter
        self.constraint_value = constraint_value
        self.a = df['a'].unique().tolist()
        self.loss_max = 1000
        self.df = df

        self.decision_dict = {}

        self.all_test_criterias, self.criteria_dict = self.create_test_criterias(
            df, n_bins)

        self.cut_dict = self.create_cut_dict(df)

    @abstractmethod
    def find_best_loss(self, policy, active_criterias, loss, max_n_layers, lookahead):
        pass

    def create_test_criterias(self, df, n_bins):
        """
        Creates a list of all criterias to be tested.
        Continuous values are discretized.
        """

        df_features = df[self.features]
        active_criterias = []
        criteria_dict = {}
        idx = 0
        for (feature, data) in df_features.items():
            if data.dtype.name == 'bool':
                test_criteria = (feature, [True], [True],
                                 '{} = {}'.format(feature, True))
                test_criteria_reverse = (feature, [True], [False],
                                         '{} != {}'.format(feature, True))
                criteria_dict[idx] = test_criteria
                criteria_dict[idx + 1] = test_criteria_reverse

                active_criterias.append((idx, idx + 1))

                idx += 2
            elif data.dtype.name == 'category':
                value_list = df[feature].unique().tolist()
                for value in value_list:
                    test_criteria = (feature, [value], [True],
                                     '{} = {}'.format(feature, value))
                    test_criteria_reverse = (feature, [value], [False],
                                             '{} != {}'.format(feature, value))
                    criteria_dict[idx] = test_criteria
                    criteria_dict[idx + 1] = test_criteria_reverse

                    active_criterias.append((idx, idx + 1))

                    idx += 2
            else:
                # Feature
                test_criterias = np.linspace(0, 100, n_bins + 1, False)[1:]

                p_all = np.percentile(data, test_criterias)
                p_all = np.unique(p_all)

                min_i = np.min(data)
                max_i = np.max(data)
                for p in reversed(p_all):
                    if p != min_i and p != max_i:
                        # Test <=
                        test_criteria = (feature, [min_i, p, max_i], [True, False],
                                         '{} <= {}'.format(feature, p))
                        test_criteria_reverse = (feature, [min_i, p, max_i], [False, True],
                                                 '{} > {}'.format(feature, p))
                        criteria_dict[idx] = test_criteria
                        criteria_dict[idx + 1] = test_criteria_reverse

                        active_criterias.append((idx, idx + 1))

                        idx += 2

        # Logging
        for key in criteria_dict:
            logging.debug('Key: {}, criteria: {}'.format(key, criteria_dict[key][3]))

        return active_criterias, criteria_dict

    def train_policy(self, max_n_layers, lookahead):
        """
        Main training loop of the tree

        Want to build:
        key = 1
        d_exit = 0
        policy = [(key, d_exit)]

        key2 = 4
        d_exit2 = 0
        policy += [(key2, d_exit2)]

        The key is used to get the actual criteria
        feature, bins, labels, name = self.criterias_dict[key]

        where (for example):
        feature = 'x0'
        bins = [min, 50, max]
        labels = [False, True]
        name = 'x0 > 50'

        or:
        feature = 'x1'
        bins = [min, 50, max]
        labels = [True, False]
        name = 'x1 <= 50'

        """

        # Building decision list
        policy_builder = BuildPolicyHelper()

        # Decision lists to choose between
        policy_final = FinalPolicyHelper(self.criteria_dict, self.constraint_value)

        best_loss, loss_second, d_best = self.get_loss_no_tree()
        policy_builder.add([d_best], self.all_test_criterias, best_loss, 1, loss_second)
        policy_final.add([d_best], best_loss, loss_second, loss_second)

        while policy_builder.options_left():
            policy_i, active_criterias_i, loss_i, criteria_i, max_criteria_i = policy_builder.get_first()
            policy_builder.remove_first()

            # Build layer
            logging.debug('------')
            logging.debug('Best loss no split: {}'.format(loss_i))

            loss_min, loss_constraint, best_policy, alternative_policy, active_criterias_i, alternative_criterias_i, is_final = self.find_best_loss(
                policy_i, active_criterias_i, loss_i, criteria_i, max_n_layers, lookahead)

            max_loss_constraint = max_criteria_i if loss_constraint < max_criteria_i else loss_constraint

            if loss_min == self.loss_max or loss_min > loss_i or not best_policy:
                logging.debug('No further')
            elif is_final or len(best_policy) - 1 >= max_n_layers:
                logging.debug('Is final')
            else:
                logging.debug('Continue')
                policy_builder.add(best_policy, active_criterias_i, loss_min, loss_constraint, max_loss_constraint)
                if len(alternative_policy) > 0:
                    logging.debug('Add alternative')
                    policy_builder.add(alternative_policy, alternative_criterias_i, loss_min, loss_constraint, max_loss_constraint)

            policy_final.add(best_policy, loss_min, loss_constraint, max_loss_constraint)

        policy_final.logg_all_for_debugging()
        policy_best, best_loss, constraint, max_loss_constraint = policy_final.get_best_policy()

        # Log result
        logging.debug('Final policy:')
        self.log_policy(policy_best)
        logging.debug('Reward: {}'.format(best_loss))
        logging.debug('Constraint: {}'.format(constraint))
        logging.debug('Max constraint: {}'.format(max_loss_constraint))

        return policy_best, best_loss, constraint, max_loss_constraint

    def get_loss_no_tree(self):
        best_loss = 10000
        # Find treat none policy
        for d in [0, 1]:
            d_vec = d * np.ones(len(self.df), dtype=int)
            loss, loss_second_i, __ = self.weighter.get_obj_and_constr(d_vec)
            if d == 0:
                loss_second_i = 0

            if loss < best_loss and loss_second_i < self.constraint_value:
                best_loss = loss
                loss_second = loss_second_i
                d_best = d

        logging.debug('No tree, best d = {}, loss = {}'.format(d_best, best_loss))

        return best_loss, loss_second, d_best

    def create_cut_dict(self, df):
        cut_dict = {}
        for key in self.criteria_dict:
            feature, bins, labels, name = self.criteria_dict[key]
            if len(bins) == 3:
                df_filter = pd.cut(x=df[feature], bins=bins, include_lowest=True, labels=labels).to_numpy(dtype=bool)
                cut_dict[key] = df_filter
            else:
                cut_dict[key] = (df[feature] == bins[0]).to_numpy(dtype=bool)
        return cut_dict

    def log_policy(self, policy):
        for key, exit_d in policy[:-1]:
            feature, bins, labels, name = self.criteria_dict[key]
            logging.info('{}, exit: {}'.format(name, exit_d))
        logging.info(policy[-1])

    @staticmethod
    def get_decisions_fixed(df, policy, criteria_dict):
        d = policy[-1] * np.ones(len(df), dtype=int)
        df_filter_exit = np.full((len(df)), False, dtype=bool)
        for key, exit_d in policy[:-1]:
            feature, bins, labels, __ = criteria_dict[key]
            if len(bins) == 3:
                df_filter = pd.cut(x=df[feature], bins=bins, include_lowest=True, labels=labels).to_numpy(dtype=bool)
            else:
                df_filter = (df[feature] == bins[0]).to_numpy(dtype=bool)
            df_filter_new_exit = np.logical_and(~df_filter_exit, ~df_filter)
            d[df_filter_new_exit] = exit_d
            df_filter_exit = np.logical_or(df_filter_new_exit, df_filter_exit)
        return d, df_filter_exit

    @staticmethod
    def get_decisions_building(cut_dict, d_policy, df_filter_exit, key_criteria, decisions_list):
        d_list = [d_policy.copy() for __ in decisions_list]

        df_filter = cut_dict[key_criteria]
        df_filter_new_exit = np.logical_and(~df_filter_exit, ~df_filter)
        if np.mean(df_filter_new_exit) < 0.01:
            is_same = True
        else:
            is_same = np.array_equal(df_filter_new_exit, df_filter_exit)
        for decision, d_i in zip(decisions_list, d_list):
            d_i[df_filter_new_exit] = decision[0]
        df_filter_exit = np.logical_or(df_filter_new_exit, df_filter_exit)
        if np.mean(df_filter_exit) >0.99:
            is_same = True
        for decision, d_i in zip(decisions_list, d_list):
            d_i[~df_filter_exit] = decision[-1]
        return d_list, df_filter_exit, is_same

    @staticmethod
    def get_key_list(fixed_policy, skeleton_key, skeleton_dexit_list):
        all_policies = []
        for skeleton_dexit in skeleton_dexit_list:
            policy_list = fixed_policy.copy()
            for key, d in zip(skeleton_key, skeleton_dexit):
                policy_list.append((key, d))
            key = tuple(policy_list)
            all_policies.append(key)
        if len(all_policies) == 0 and len(fixed_policy) > 0:
            key = tuple(fixed_policy)
            all_policies.append(key)
        elif len(all_policies) == 0:
            all_policies.append(())
        return all_policies

    def update_active_criterias(self, active_criterias, new_criteria):
        new_feature, new_bins, new_labels, new_name = new_criteria
        to_pop = []
        for key, key_reverse in active_criterias:
            feature, bins, labels, name = self.criteria_dict[key]
            if new_feature == feature and len(new_bins) == 3:
                if (bins[1] >= new_bins[1]) == new_labels[0]:
                    to_pop.append((key, key_reverse))
                elif bins[1] == new_bins[1]:
                    to_pop.append((key, key_reverse))
            elif len(new_bins) == 1 and new_feature == feature:  # bins[0] == new_bins[0]:
                to_pop.append((key, key_reverse))
        new_active_criterias = active_criterias.copy()
        for i in to_pop[::-1]:
            new_active_criterias.remove(i)
        return new_active_criterias

    @staticmethod
    def get_n_lookahead(policy, max_n_layers, lookahead):
        n_layers = len(policy) - 1
        empty_layers = max_n_layers - n_layers
        if lookahead < empty_layers:
            return lookahead
        else:
            return empty_layers

    def get_possible_skeletons(self, n_lookahead, active_criterias):
        criterias_list = chain(*active_criterias)
        all_skeletons = list(permutations(criterias_list, n_lookahead))
        possible_skeletons = []
        for a_skeleton in all_skeletons:
            is_ok = True
            check = []
            for criteria_key in a_skeleton:
                feature, bins, labels, name = self.criteria_dict[criteria_key]
                if len(bins) == 3:
                    test_criteria = (feature, bins[1], labels[0])
                    for feature_i, value_i, criteria_i in check:
                        if feature_i == feature:
                            value = test_criteria[1]
                            if (value >= value_i) == criteria_i:
                                is_ok = False
                                break
                            if value == value_i:
                                is_ok = False
                                break
                    if is_ok:
                        check.append(test_criteria)
                elif len(bins) == 1:
                    test_criteria = (feature, bins[0], labels[0])
                    for feature_i, value_i, criteria_i in check:
                        if feature_i == feature:
                            value = test_criteria[1]
                            if value == value_i:
                                is_ok = False
                                break
                    if is_ok:
                        check.append(test_criteria)
                else:
                    break
            if is_ok:
                possible_skeletons.append(list(a_skeleton))
        return possible_skeletons

    def get_possible_decisions(self, n_lookahead):
        # Removes adjacent decisions that are equal
        all_decisions = list(product(self.a, repeat=n_lookahead))
        possible_decisions = []
        for decision_i in all_decisions:
            a_decision = list(decision_i)
            if len(a_decision) >= 2 and a_decision[-1] != a_decision[-2]:
                possible_decisions += [a_decision]
        return possible_decisions

    def get_decisions_policy(self, policy, df):
        d_policy, __ = self.get_decisions_fixed(df, policy, self.criteria_dict)
        return d_policy

    def get_first_x0_split(self, policy):
        for key, exit_d in policy[:-1]:
            feature, bins, labels, name = self.criteria_dict[key]
            if feature == 'x0':
                return bins[1]
        return policy[-1]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Create FFT")
        parser.add_argument('--n_bins', type=int, default=200,
                            help='number of bins each continuous value is discretized into (default: %(default)s)')
        parser.add_argument('--lookahead', type=int, nargs='+', default=[0, 1],
                            help='number of steps to lookahead when building fft (default: %(default)s)')
        return parent_parser
