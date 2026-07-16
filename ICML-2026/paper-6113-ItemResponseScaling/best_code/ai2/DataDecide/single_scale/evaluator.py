import numpy as np
from scipy.stats import spearmanr, pearsonr

# TODO: Implement inputs with std

def get_transformed_values_compute_and_seeds(df, group=None, model=None, metric=None, compute_limit=None, compute_bin_width=None) -> dict:
    """
    Get transformed values from dataframe. For each seed, get the last value by C.
    Return: dict[(seed, value) pairs]
    """
    conditions = []
    if model:
        conditions.append(df['model'] == model)
    if group:
        conditions.append(df['group'] == group)
    if metric:
        conditions.append(df['metric'] == metric)
    if compute_limit:
        conditions.append(df['compute_latest'] <= compute_limit)

    # Combine all conditions
    if conditions:
        df = df.loc[np.logical_and.reduce(conditions)]

    df = df.sort_values(by='compute_latest', ascending=True)

    # Group by seed
    # For each, pick the transformed value closest to compute mark
    seed_latest_values = df.groupby('seed').last()

    assert len(seed_latest_values['compute_latest'].unique()) == 1, "compute_latest values are not the same for all seeds"
    compute_latest = seed_latest_values['compute_latest'].iloc[0]

    sorted_seed_latest_values = dict(sorted(seed_latest_values['value'].to_dict().items()))

    if compute_limit is not None and compute_bin_width is not None and compute_limit != float('inf'):
        # Check all seeds are within compute_bin_width of the compute_limit
        for seed, row in seed_latest_values.iterrows():
            if abs(row['compute_latest'] - compute_limit) > compute_bin_width:
                # print(f"Seed {seed} is not within {compute_bin_width} of compute_limit {compute_limit}, with compute_latest {row['compute_latest']}")
                return None, compute_latest

    #TODO check if all seeds are available
    return sorted_seed_latest_values, compute_latest


class BaseWinnerDecider:
    """
    Base class for deciding winner.
    Returns {
        "mix1_better": bool,
        "mix2_better": bool,
        "abstain": bool
    }
    mix1_better/mix2_better (2-class) and abstain (3-class) can both be True.
    """
    def _decide_abstain(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement get_abstain_decision")

    def _get_single_decision(self, mean1, mean2):
        """
        Decide which mix is better based on mean values alone.
        """
        if mean1 > mean2:
            return "mix1_better"
        elif mean2 > mean1:
            return "mix2_better"
        else:
            # corner case abstain condition
            return "abstain"

    def __call__(self, seed_results1: dict, seed_results2: dict):
        """
        Common logic for determining the winner.
        """
        # Find common seeds
        common_seeds = set(seed_results1.keys()) & set(seed_results2.keys())
        if not common_seeds:
            raise ValueError(f"No common seeds found between data mixes:\n{seed_results1}\n{seed_results2}")
        
        assert common_seeds == set(seed_results1.keys()) and common_seeds == set(seed_results2.keys()), f"not all seeds are available, {common_seeds}, {seed_results1.keys()}, {seed_results2.keys()}"

        decision = {"mix1_better": False, "mix2_better": False, "abstain": False}

        # Collect decisions for all seeds
        decisions = [
            self._get_single_decision(seed_results1[seed], seed_results2[seed])
            for seed in common_seeds
        ]

        # Count votes
        mix1_votes = decisions.count("mix1_better")
        mix2_votes = decisions.count("mix2_better")
        total_seeds = len(common_seeds)

        # Assign mix better decisions
        if mix1_votes > total_seeds / 2:
            decision["mix1_better"] = True
        elif mix2_votes > total_seeds / 2:
            decision["mix2_better"] = True

        # Assign abstain condition
        if self._decide_abstain(mix1_votes, mix2_votes, total_seeds, seed_results1, seed_results2):
            decision["abstain"] = True

        return decision

class SingleValueWinnerDecider(BaseWinnerDecider):
    """
    Winner decider for single value comparisons. TODO this is always false intentionally?
    """
    def _decide_abstain(self, mix1_votes, mix2_votes, total_seeds, seed_results1, seed_results2):
        mean1 = seed_results1[next(iter(seed_results1))]
        mean2 = seed_results2[next(iter(seed_results2))]
        std1, std2 = None, None

        if std1 is None or std2 is None:
            return False
        return (mean1 - std1 <= mean2 + std2) and (mean1 + std1 >= mean2 - std2)


class MultiSeedWinnerDecider(BaseWinnerDecider):
    """
    Winner decider for multi-seed comparisons. Abstains if there is any disagreement between seeds.
    """
    def _decide_abstain(self, mix1_votes, mix2_votes, total_seeds, seed_results1, seed_results2):
        return mix1_votes < total_seeds and mix2_votes < total_seeds


class Evaluator:
    """
    Compute metrics that evaluate how well target predictions worked.
    """
    def __init__(self, config, multi_seed=False):
        self.config = config
        self.multi_seed = multi_seed
        self.raw_results = []
        if self.multi_seed:
            self.get_winner_fn = MultiSeedWinnerDecider()
        else:
            self.get_winner_fn = SingleValueWinnerDecider()

    def calculate_metrics(self, results: list[dict]) -> dict:
        """
        Input is list of (seed, transformed) values:
        eg.
        [{
            'mix1': A,
            'mix2': B,
            'primary_values1': {seed1: val1, seed2: val2, seed3: val3},
            'primary_values2': {seed1: val1, seed2: val2, seed3: val3},
            'metric_values1': {seed1: val1, seed2: val2, seed3: val3},
            'metric_values2': {seed1: val1, seed2: val2, seed3: val3}
        }, ...
        ]

        Calculate requested metrics.
        """
        self.raw_results.extend(results)
        ret = {}

        if self.config.get("binary_accuracy", False):
            ret.update(self._calculate_binary_accuracy())
        if self.config.get("three_way_accuracy", False):
            ret.update(self._calculate_three_way_accuracy())
        if self.config.get("magnitude_correlation", False):
            ret.update(self._calculate_correlation())
        if self.config.get("pearson_correlation", False):
            ret.update(self._calculate_correlation_pearson())
        # if self.config.get("rank_correlation", False):
        #     ret.update(self._calculate_rank_correlation())
        if self.config.get("magnitude_ece", False):
            ret.update(self._calculate_ece())
        if self.config.get("recall_at_k", False):
            ret.update(self._calculate_recall_at_k())
        if self.config.get("weighted_pearson_correlation", False):
            ret.update(self._calculate_weighted_pearson_correlation())
        if self.config.get("NDCG", False):
            ret.update(self._calculate_ndcg())

        if self.config.get("debug", True):
            ret.update(self.calculate_debug())

        self.raw_results = [] # reset
        return ret

    def _calculate_binary_accuracy(self):
        """Calculate binary accuracy. Disregards abstentions."""
        correct, total = 0, 0

        for result in self.raw_results:
            primary_decision = self.get_winner_fn(
                result['primary_values1'],
                result['primary_values2'],
            )
            metric_decision = self.get_winner_fn(
                result['metric_values1'],
                result['metric_values2'],
            )
            if  (metric_decision["mix1_better"] != metric_decision["mix2_better"]) and (
                primary_decision["mix1_better"] == metric_decision["mix1_better"]
                or primary_decision["mix2_better"] == metric_decision["mix2_better"]
            ):
                correct += 1
            total += 1

        binary_accuracy = correct / total if total > 0 else 0
        return {"binary_accuracy": binary_accuracy}

    def _calculate_three_way_accuracy(self):
        """Calculate 3-way accuracy. Counts abstentions as a 3rd class."""
        def get_prediction(decision_dict):
            """Helper to extract prediction in this order"""
            for decision in ["abstain", "mix1_better", "mix2_better"]:
                if decision_dict.get(decision):
                    return decision
            return None

        correct, total = 0, 0

        for result in self.raw_results:
            primary_decision = self.get_winner_fn(
                result['primary_values1'],
                result['primary_values2'],
            )
            metric_decision = self.get_winner_fn(
                result['metric_values1'],
                result['metric_values2'],
            )

            # Determine predictions
            primary_pred = get_prediction(primary_decision)
            metric_pred = get_prediction(metric_decision)
            if primary_pred == metric_pred:
                correct += 1
            total += 1

        three_way_acc = correct / total if total > 0 else 0
        return {"three_way_accuracy": three_way_acc}

    def _calculate_correlation(self):
        """Get Spearman correlation between metric values and primary values"""
        all_primary_values, all_metric_values = {}, {}

        for result in self.raw_results:
            mix1 = result['mix1']
            mix2 = result['mix2']
            assert len(result['primary_values1'].values()) == 1, "primary_values1 should be a dict with only one value"
            assert len(result['primary_values2'].values()) == 1, "primary_values2 should be a dict with only one value"
            if len(result['metric_values1'].values()) != 1 or len(result['metric_values2'].values()) != 1:
                assert len(result['metric_values1'].values()) != 1 and len(result['metric_values2'].values()) != 1, "metric_values should be a dict with only one value"
                result['metric_values1'] = np.mean(list(result['metric_values1'].values()))
                result['metric_values2'] = np.mean(list(result['metric_values2'].values()))
            all_primary_values[mix1] = list(result['primary_values1'].values())[0]
            all_primary_values[mix2] = list(result['primary_values2'].values())[0]
            all_metric_values[mix1] = list(result['metric_values1'].values())[0]
            all_metric_values[mix2] = list(result['metric_values2'].values())[0]
        
        assert all_primary_values.keys() == all_metric_values.keys(), "mixes are not the same"
        
        small_values = []
        large_values = []
        for mix in all_primary_values.keys():
            small_values.append(all_primary_values[mix])
            large_values.append(all_metric_values[mix])
        
        spearman_corr, _ = spearmanr(small_values, large_values)

        return {"magnitude_correlation": spearman_corr}


    def _calculate_correlation_pearson(self):
        """Get Pearson correlation between metric values and primary values"""
        all_primary_values, all_metric_values = {}, {}

        for result in self.raw_results:
            mix1 = result['mix1']
            mix2 = result['mix2']
            assert len(result['primary_values1'].values()) == 1, "primary_values1 should be a dict with only one value"
            assert len(result['primary_values2'].values()) == 1, "primary_values2 should be a dict with only one value"
            if len(result['metric_values1'].values()) != 1 or len(result['metric_values2'].values()) != 1:
                assert len(result['metric_values1'].values()) != 1 and len(result['metric_values2'].values()) != 1, "metric_values should be a dict with only one value"
                result['metric_values1'] = np.mean(list(result['metric_values1'].values()))
                result['metric_values2'] = np.mean(list(result['metric_values2'].values()))
            all_primary_values[mix1] = list(result['primary_values1'].values())[0]
            all_primary_values[mix2] = list(result['primary_values2'].values())[0]
            all_metric_values[mix1] = list(result['metric_values1'].values())[0]
            all_metric_values[mix2] = list(result['metric_values2'].values())[0]
        
        assert all_primary_values.keys() == all_metric_values.keys(), "mixes are not the same"
        
        small_values = []
        large_values = []
        for mix in all_primary_values.keys():
            small_values.append(all_primary_values[mix])
            large_values.append(all_metric_values[mix])
        
        pearson_corr, _ = pearsonr(small_values, large_values)

        return {"pearson_correlation": pearson_corr}
    
    def _calculate_weighted_pearson_correlation(self):
        """Get weighted Pearson correlation between metric values and primary values, where the weights are the primary values"""
        all_primary_values, all_metric_values = {}, {}

        for result in self.raw_results:
            mix1 = result['mix1']
            mix2 = result['mix2']
            assert len(result['primary_values1'].values()) == 1, "primary_values1 should be a dict with only one value"
            assert len(result['primary_values2'].values()) == 1, "primary_values2 should be a dict with only one value"
            if len(result['metric_values1'].values()) != 1 or len(result['metric_values2'].values()) != 1:
                assert len(result['metric_values1'].values()) != 1 and len(result['metric_values2'].values()) != 1, "metric_values should be a dict with only one value"
                result['metric_values1'] = np.mean(list(result['metric_values1'].values()))
                result['metric_values2'] = np.mean(list(result['metric_values2'].values()))
            all_primary_values[mix1] = list(result['primary_values1'].values())[0]
            all_primary_values[mix2] = list(result['primary_values2'].values())[0]
            all_metric_values[mix1] = list(result['metric_values1'].values())[0]
            all_metric_values[mix2] = list(result['metric_values2'].values())[0]
        
        assert all_primary_values.keys() == all_metric_values.keys(), "mixes are not the same"
        
        small_values = []
        large_values = []
        for mix in all_primary_values.keys():
            small_values.append(all_primary_values[mix])
            large_values.append(all_metric_values[mix])
        
        weights = np.array(small_values)
        weights = weights - weights.min()
        weights = weights / weights.max()
        temperature = 0.1
        weights = np.exp(weights / temperature) / np.exp(weights / temperature).sum()
        small_values = np.array(small_values)
        large_values = np.array(large_values)

        weighted_mean_small = np.average(small_values, weights=weights)
        weighted_mean_large = np.average(large_values, weights=weights)

        weighted_cov = np.average((small_values - weighted_mean_small) * (large_values - weighted_mean_large), weights=weights)
        weighted_std_small = np.sqrt(np.average((small_values - weighted_mean_small) ** 2, weights=weights))
        weighted_std_large = np.sqrt(np.average((large_values - weighted_mean_large) ** 2, weights=weights))
                                     
        pearson_corr = weighted_cov / (weighted_std_small * weighted_std_large)

        return {"weighted_pearson_correlation": pearson_corr}



    def _calculate_ece(self, bins=5):
        """Calculate Expected Calibration Error (ECE)"""
        confidences, accuracies = [], []

        for result in self.raw_results:
            metric_values1 = result['metric_values1']
            metric_values2 = result['metric_values2']

            for v1, v2 in zip(metric_values1, metric_values2):
                confidence = abs(v1 - v2)
                primary_decision = self.get_winner_fn(
                    result['primary_values1'],
                    result['primary_values2'],
                )

                if primary_decision["abstain"]:
                    continue

                metric_winner = "mix1_better" if primary_decision["mix1_better"] else "mix2_better"

                # Check if metric winner matches primary decision
                accuracy = 1 if primary_decision[metric_winner] else 0
                confidences.append(confidence)
                accuracies.append(accuracy)

        bin_edges = np.linspace(0, 1, bins + 1)
        ece = 0

        for i in range(bins):
            bin_indices = [j for j, c in enumerate(confidences) if bin_edges[i] <= c < bin_edges[i + 1]]
            if not bin_indices:
                continue

            bin_acc = np.mean([accuracies[j] for j in bin_indices])
            bin_conf = np.mean([confidences[j] for j in bin_indices])
            ece += len(bin_indices) * abs(bin_acc - bin_conf)

        return {"magnitude_ece": ece / len(confidences) if len(confidences) > 0 else 0}

    # TODO: For this to work, we need to store all data ranks at all time
    def _calculate_rank_correlation(self):
        pass

    def _calculate_recall_at_k(self, k=5, n=3):
        """Calculate recall at k top predictions of n target rankings"""
        all_primary_values, all_metric_values = {}, {}

        for result in self.raw_results:
            mix1 = result['mix1']
            mix2 = result['mix2']
            assert len(result['primary_values1'].values()) == 1, "primary_values1 should be a dict with only one value"
            assert len(result['primary_values2'].values()) == 1, "primary_values2 should be a dict with only one value"
            if len(result['metric_values1'].values()) != 1 or len(result['metric_values2'].values()) != 1:
                assert len(result['metric_values1'].values()) != 1 and len(result['metric_values2'].values()) != 1, "metric_values should be a dict with only one value"
                result['metric_values1'] = np.mean(list(result['metric_values1'].values()))
                result['metric_values2'] = np.mean(list(result['metric_values2'].values()))
            all_primary_values[mix1] = list(result['primary_values1'].values())[0]
            all_primary_values[mix2] = list(result['primary_values2'].values())[0]
            all_metric_values[mix1] = list(result['metric_values1'].values())[0]
            all_metric_values[mix2] = list(result['metric_values2'].values())[0]
        
        assert all_primary_values.keys() == all_metric_values.keys(), "mixes are not the same"
        
        small_values = []
        large_values = []
        for mix in all_primary_values.keys():
            small_values.append(all_primary_values[mix])
            large_values.append(all_metric_values[mix])
        
        top_n_indices = np.argsort(large_values)[-n:]
        top_n_mixes = [list(all_primary_values.keys())[i] for i in top_n_indices]

        top_k_indices = np.argsort(small_values)[-k:]
        top_k_mixes = [list(all_primary_values.keys())[i] for i in top_k_indices]

        recall = len(set(top_n_mixes) & set(top_k_mixes)) / n
        return {"recall_at_k": recall}
    
    def _calculate_ndcg(self):
        all_primary_values, all_metric_values = {}, {}

        for result in self.raw_results:
            mix1 = result['mix1']
            mix2 = result['mix2']
            assert len(result['primary_values1'].values()) == 1, "primary_values1 should be a dict with only one value"
            assert len(result['primary_values2'].values()) == 1, "primary_values2 should be a dict with only one value"
            if len(result['metric_values1'].values()) != 1 or len(result['metric_values2'].values()) != 1:
                assert len(result['metric_values1'].values()) != 1 and len(result['metric_values2'].values()) != 1, "metric_values should be a dict with only one value"
                result['metric_values1'] = np.mean(list(result['metric_values1'].values()))
                result['metric_values2'] = np.mean(list(result['metric_values2'].values()))
            all_primary_values[mix1] = list(result['primary_values1'].values())[0]
            all_primary_values[mix2] = list(result['primary_values2'].values())[0]
            all_metric_values[mix1] = list(result['metric_values1'].values())[0]
            all_metric_values[mix2] = list(result['metric_values2'].values())[0]
        
        assert all_primary_values.keys() == all_metric_values.keys(), "mixes are not the same"

        max_metric_value = max(all_primary_values.values())
        min_metric_value = min(all_primary_values.values())
        for mix in all_primary_values.keys():
            all_primary_values[mix] = (all_primary_values[mix] - min_metric_value) / (max_metric_value - min_metric_value)
    
        dcg = 0
        idcg = 0
        pred_mixes_decending = [mix for mix, value in sorted(all_metric_values.items(), key=lambda x: x[1], reverse=True)]
        actual_mixes_decending = [mix for mix, value in sorted(all_primary_values.items(), key=lambda x: x[1], reverse=True)]
        for i, (pred_mix, actual_mix) in enumerate(zip(pred_mixes_decending, actual_mixes_decending)):
            dcg += all_primary_values[pred_mix] / np.log2(i + 2)
            idcg += all_primary_values[actual_mix] / np.log2(i + 2)
        
        ndcg = dcg / idcg
        return {"NDCG": ndcg}


    def calculate_debug(self):
        """Tai: Fake method, help make sure new code still produces same results"""
        correct_count, incorrect_count, abstain_count = 0, 0, 0
        primary_abstain, primarary_total = 0, 0
        mix1_better, mix2_better = 0, 0
        actual_mix1_better, actual_mix2_better = 0, 0
        mix_pairs_incorrect = []
        mix_pairs_correct = []

        for result in self.raw_results:
            primary_decision = self.get_winner_fn(
                result['primary_values1'],
                result['primary_values2'],
            )
            metric_decision = self.get_winner_fn(
                result['metric_values1'],
                result['metric_values2'],
            )

            if primary_decision["abstain"]:
                primary_abstain += 1
            primarary_total += 1

            if (
                primary_decision["mix1_better"] == metric_decision["mix1_better"]
                and primary_decision["mix2_better"] == metric_decision["mix2_better"]            ):
                correct_count += 1
                mix_pairs_correct.append((result['mix1'], result['mix2']))
            else:
                incorrect_count += 1
                mix_pairs_incorrect.append((result['mix1'], result['mix2']))

            if primary_decision["mix1_better"]:
                actual_mix1_better += 1
            if primary_decision["mix2_better"]:
                actual_mix2_better += 1
            
            if metric_decision["mix1_better"]:
                mix1_better += 1
            if metric_decision["mix2_better"]:
                mix2_better += 1
            

            if metric_decision["abstain"]:
                abstain_count += 1

        return {
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "abstain_count": abstain_count,
            "total_count": len(self.raw_results),
            "primary_abstain": primary_abstain,
            "mix1_better": mix1_better,
            "mix2_better": mix2_better,
            "actual_mix1_better": actual_mix1_better,
            "actual_mix2_better": actual_mix2_better,
            "mix_pairs_incorrect": mix_pairs_incorrect,
            "mix_pairs_correct": mix_pairs_correct,

        }
