import logging
import numpy as np

from method.create_fft_base import FftPolicyBase


class FftPolicyConstraint(FftPolicyBase):
    """FFT policy builder that optimizes an objective subject to a constraint."""

    def __init__(self, df, weighter, constraint_value, features, n_bins):
        super().__init__(df, weighter, constraint_value, features, n_bins)

        self.weighter = weighter
        self.get_obj_constraint = weighter.get_obj_and_constr

        self.constraint_value = constraint_value


    def find_best_loss(self, policy, active_criterias, loss, constraint, max_n_layers, lookahead):
        n_lookahead = self.get_n_lookahead(policy, max_n_layers, lookahead)
        # Removed lookahead asymmetry penalty (Idea 09)
        # if lookahead != 1:
                # loss = loss + 0.001
        loss_before = loss
        alternative_policy = []
        new_active_criterias = []
        active_alternative_criterias = []
        best_policy = policy.copy()
        loss_constraint = constraint

        for n_i in range(1, n_lookahead + 1):
            skeleton_criteria_list = self.get_possible_skeletons(n_i, active_criterias)
            skeleton_dexit_before_list = self.get_possible_decisions(n_i)
            skeleton_dexit_list = self.get_possible_decisions(2)

            for skeleton_criterias in skeleton_criteria_list:
                skeleton_criteria_before = skeleton_criterias.copy()
                del skeleton_criteria_before[-1:]
                new_criteria = skeleton_criterias[-1]

                key_list = self.get_key_list(policy[:-1], skeleton_criteria_before, skeleton_dexit_before_list)

                for key in key_list:
                    if not key:
                        df_exit = np.full((len(self.df)), False, dtype=bool)
                        d = np.ones(len(self.df), dtype=int)

                    else:
                        d, df_exit = self.decision_dict.get(key)

                    # Get decisons
                    d_policy_list, df_exit, is_same = self.get_decisions_building(self.cut_dict, d, df_exit,
                                                                                  new_criteria, skeleton_dexit_list)

                    if is_same and n_i == 1:
                        active_criterias = active_criterias.copy()
                        if new_criteria % 2 == 0:
                            to_remove = (new_criteria, new_criteria + 1)
                        else:
                            to_remove = (new_criteria - 1, new_criteria)
                        if to_remove in active_criterias:
                            active_criterias.remove(to_remove)

                    # Get loss
                    for d_policy, d_exit in zip(d_policy_list, skeleton_dexit_list):
                        # Save result to reuse later
                        if not key:
                            new_policy = [(new_criteria, d_exit[0])]
                            new_key = tuple(new_policy)
                        else:
                            new_policy = list(key)
                            new_policy.append((new_criteria, d_exit[0]))
                            new_key = tuple(new_policy)

                        self.decision_dict[new_key] = (d_policy, df_exit)
                        if not is_same:
                            loss_i, loss_second, __ = self.get_obj_constraint(d_policy)
                            n = len(new_policy)
                            loss_i = loss_i + 0.005 * n

                            if loss_i < loss and loss_second < self.constraint_value:
                                loss = loss_i
                                logging.debug('Loss for the constraint {}:'.format(loss_second))
                                best_policy = new_policy.copy()
                                loss_constraint = loss_second

                                # Log if in debug mode
                                logging.debug("Loss: {}".format(loss))
                                for key_i, exit_d_i in best_policy:
                                    feature, bins, labels, name = self.criteria_dict[key_i]
                                    logging.debug('{}, exit: {}'.format(name, exit_d_i))
                                best_policy.append(d_exit[-1])
                                logging.debug("Best policy: {}".format(best_policy))

                                if lookahead == 1:
                                    alternative_policy = new_policy.copy()
                                    alternative_policy.pop()
                                    new_alt_criteria = new_criteria + 1 - 2 * (new_criteria % 2)
                                    alternative_policy.append((new_alt_criteria, d_exit[-1]))
                                    alternative_policy.append(d_exit[-2])

        is_final = True
        new_policy = policy.copy()
        new_policy.pop()

        if loss < loss_before and loss_constraint < self.constraint_value:
            if n_lookahead + len(policy) - 1 != max_n_layers:
                is_final = False
                new_layer = len(policy) - 1
                new_policy.append(best_policy[new_layer])
                new_policy.append(best_policy[-1])

                key, d_exit = best_policy[new_layer]
                criteria = self.criteria_dict[key]
                new_active_criterias = self.update_active_criterias(active_criterias, criteria)

                if lookahead == 1:
                    key, d_exit = alternative_policy[new_layer]
                    criteria = self.criteria_dict[key]
                    active_alternative_criterias = self.update_active_criterias(active_criterias, criteria)

            else:
                new_policy = best_policy
        else:
            new_policy = best_policy

        return loss, loss_constraint, new_policy, alternative_policy, new_active_criterias, active_alternative_criterias, is_final
