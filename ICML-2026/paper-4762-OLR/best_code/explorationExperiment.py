# Import necessary libraries for the sliding window multi-armed bandit experiment
import os
import pickle
import numpy as np
import random
import copy
import tqdm  # Progress bar for loops
import matplotlib.pyplot as plt

# Import custom classes and algorithms from local modules
from testSlidingWindowMAB import Bernoulli_Arm, Arm_Reading_Buffer
from testSlidingWindowAlgorithm import bucket_exploration, bucket_regret, sliding_window_ucb, regret_minimization_everlasting, baseline_exploration_not_clear_expiration, baseline_exploration_clear_expiration

class ExperimentResults:
    '''
    The class to store the experiment results
    ----- Parameters -----
    # memory_sizes: sizes of the memory for the exploration algorithm
    # y_mean: mean differences between best arm and algorithm output
    # y_median: median differences between best arm and algorithm output
    # y_max: maximum differences between best arm and algorithm output
    ----- Methods ----
    # None
    '''

    def __init__(self, memory_sizes, y_mean = [], y_median = [],  y_max = []):
        '''
        :param memory_sizes: sizes of the memory for the exploration algorithm
        :param y_mean: mean differences between best arm and algorithm output
        :param y_median: median differences between best arm and algorithm output
        :param y_max: maximum differences between best arm and algorithm output
        '''
        self.memory_sizes = memory_sizes
        self.y_mean = y_mean
        self.y_median = y_median
        self.y_max = y_max

class ExplorationExperiment:
    '''
    The class to run the exploration experiment
    ----- Parameters -----
    # n: number of arms
    # window_size: size of the sliding window
    # memory_sizes: sizes of the memory for the exploration algorithm
    # num_experiments: number of experiments to run for averaging results
    # baseline_expiration_type: type of expiration ('clear' or 'not_clear')
    # data_type: type of data ('synthetic' or 'real_world')
    # algorithm_results: stores the results of the exploration algorithm
    # baseline_results: stores the results of the baseline exploration algorithm
    # delta: confidence parameter for the exploration algorithm
    # arm_setting: setting for the arm reward distributions
    ----- Methods ----
    # run_experiment(): runs the exploration experiment and collects results
    # plot_results(): plots the results of the experiment
    '''

    def __init__(self, n = 1000, window_size = 20, memory_sizes = list(range(2, 18, 2)), num_experiments = 10, baseline_expiration_type = 'clear', data_type = 'synthetic', delta = 0.01, arm_setting = None):
        '''
        :param n: number of arms
        :param window_size: size of the sliding window
        :param memory_sizes: sizes of the memory for the exploration algorithm
        :param num_experiments: number of experiments to run for averaging results
        :param baseline_expiration_type: type of expiration ('clear' or 'not_clear')
        :param delta: confidence parameter for the exploration algorithm
        :param data_type: type of data ('synthetic' or 'real_world')
        :param arm_setting: setting for the arm reward distributions
        '''
        self.n = n
        self.window_size = window_size
        self.memory_sizes = memory_sizes
        self.num_experiments = num_experiments
        self.baseline_expiration_type = baseline_expiration_type
        self.data_type = data_type
        self.algorithm_results = ExperimentResults(memory_sizes)
        self.baseline_results = ExperimentResults(memory_sizes)
        self.delta = delta
        self.arm_setting = arm_setting

    def run_experiment(self):
        if self.data_type == 'synthetic':
            self.run_experiment_synthetic()
        elif self.data_type == 'real_world':
            self.run_experiment_real_world()
        else:
            raise ValueError('The data_type parameter is not correctly setup.')

    def run_experiment_synthetic(self):
        '''
        Runs the exploration experiment and collects results.
        '''
        # Experiment parameters
        num_arms = self.n  # Total number of arms in the experiment
        window_size = self.window_size  # Size of the sliding window for the algorithm

        # Choose the arm reward distribution setting
        # 'clear_cut_setting': Most arms have Gaussian-distributed means, with one high-reward arm (p=0.82)
        # 'mix_in_setting': All arms have uniformly distributed reward means
        if self.arm_setting is not None:
            arm_setting = self.arm_setting
        else:
            arm_setting = 'mix_in_setting'

        # Configure experiment parameters for bucket exploration algorithm
        memory_size_list = self.memory_sizes
        # Confidence parameter for the exploration algorithm
        delta = self.delta

        num_experiments = self.num_experiments  # Number of experiment trials to average results over

        # Initialize result storage arrays
        # Each array stores results for different memory sizes (rows) across num_experiments random seeds (columns)
        # y_mean: stores mean differences between best arm and algorithm output
        # y_median: stores median differences between best arm and algorithm output  
        # y_max: stores maximum differences between best arm and algorithm output
        # y_mean_baseline, y_median_baseline, y_max_baseline: same metrics for baseline exploration algorithm
        y_mean = np.zeros((len(memory_size_list), num_experiments))
        y_median = np.zeros((len(memory_size_list), num_experiments))
        y_max = np.zeros((len(memory_size_list), num_experiments))

        y_mean_baseline = np.zeros((len(memory_size_list), num_experiments))
        y_median_baseline = np.zeros((len(memory_size_list), num_experiments))
        y_max_baseline = np.zeros((len(memory_size_list), num_experiments))

        # Main experiment loop: run num_experiments trials with different random seeds
        for i_seed in tqdm.tqdm(range(num_experiments)):
            # Set random seed for reproducible results
            np.random.seed(i_seed)
            
            # Create arms based on the chosen setting
            if arm_setting == 'clear_cut_setting':
                # Create arms with Gaussian-distributed means, plus one high-reward arm
                arm_list = [Bernoulli_Arm(random_mean='gaussian') for _ in range(num_arms-1)]
                arm_list.append(Bernoulli_Arm(p=0.82))  # Add one clearly superior arm
            elif arm_setting == 'mix_in_setting':
                # Create arms with uniformly distributed reward means
                arm_list = [Bernoulli_Arm(random_mean='uniform') for _ in range(num_arms)]
            else:
                raise ValueError('The arm_setting parameter is not correctly setup.')
            
            # Randomize arm order to avoid position bias
            random.shuffle(arm_list)
            
            # Create streaming buffer that manages arm interactions within sliding window
            streaming_arm_buffer = Arm_Reading_Buffer(arm_set = np.array(arm_list), window_size = window_size)

            result_list = []
            # Test bucket exploration algorithm with different memory sizes
            for memory_size in memory_size_list:
                streaming_arm_buffer.reset()  # Reset buffer state for fair comparison
                result_list.append(bucket_exploration(streaming_arm_buffer, memory_size, delta))

            # Calculate performance metrics: difference between optimal and achieved rewards
            diff = streaming_arm_buffer.best_rewards - np.array(result_list)
            
            # Compute statistical summaries of the performance differences
            mean_diff = np.mean(diff, axis=1)      # Average regret across time
            max_diff = np.max(diff, axis=1)        # Worst-case regret
            median_diff = np.median(diff, axis=1)  # Median regret (robust to outliers)

            # Store results for this seed in the corresponding column
            y_mean[:, i_seed] = mean_diff
            y_median[:, i_seed] = median_diff
            y_max[:, i_seed] = max_diff

            # Baseline exploration algorithm for comparison
            result_list_baseline = []
            for memory_size in memory_size_list:
                streaming_arm_buffer.reset()  # Reset buffer state for fair comparison
                result_list_baseline.append(baseline_exploration_clear_expiration(streaming_arm_buffer, memory_size, delta))

            # Calculate performance metrics: difference between optimal and achieved rewards
            diff = streaming_arm_buffer.best_rewards - np.array(result_list_baseline)
            mean_diff_baseline = np.mean(diff, axis=1)      # Average regret across time
            max_diff_baseline = np.max(diff, axis=1)        # Worst-case regret
            median_diff_baseline = np.median(diff, axis=1)  # Median regret (robust to outliers)

            # Store results for this seed in the corresponding column
            y_mean_baseline[:, i_seed] = mean_diff_baseline
            y_median_baseline[:, i_seed] = median_diff_baseline
            y_max_baseline[:, i_seed] = max_diff_baseline

        algorithm_results = ExperimentResults(memory_size_list, y_mean, y_median, y_max)
        baseline_results = ExperimentResults(memory_size_list, y_mean_baseline, y_median_baseline, y_max_baseline)
        self.algorithm_results = algorithm_results
        self.baseline_results = baseline_results

    def run_experiment_real_world(self):
        '''
        Runs the exploration experiment and collects results.
        '''
        # Experiment parameters
        num_arms = self.n  # Total number of arms in the experiment
        window_size = self.window_size  # Size of the sliding window for the algorithm

        # arm_setting give the p value for the real world data experiment
        if self.arm_setting is None:
            raise ValueError('The arm_setting parameter must be specified for real world data experiments.')
        else:
            arm_setting = self.arm_setting

        # Configure experiment parameters for bucket exploration algorithm
        memory_size_list = self.memory_sizes
        # Confidence parameter for the exploration algorithm
        delta = self.delta

        num_experiments = self.num_experiments  # Number of experiment trials to average results over

        # Initialize result storage arrays
        # Each array stores results for different memory sizes (rows) across num_experiments random seeds (columns)
        # y_mean: stores mean differences between best arm and algorithm output
        # y_median: stores median differences between best arm and algorithm output  
        # y_max: stores maximum differences between best arm and algorithm output
        # y_mean_baseline, y_median_baseline, y_max_baseline: same metrics for baseline exploration algorithm
        y_mean = np.zeros((len(memory_size_list), num_experiments))
        y_median = np.zeros((len(memory_size_list), num_experiments))
        y_max = np.zeros((len(memory_size_list), num_experiments))

        y_mean_baseline = np.zeros((len(memory_size_list), num_experiments))
        y_median_baseline = np.zeros((len(memory_size_list), num_experiments))
        y_max_baseline = np.zeros((len(memory_size_list), num_experiments))

        # Main experiment loop: run num_experiments trials with different random seeds
        for i_seed in tqdm.tqdm(range(num_experiments)):
            # Set random seed for reproducible results
            np.random.seed(i_seed)
            
            # Create arms based on the chosen setting
            arm_list = [Bernoulli_Arm(p = arm_setting[i], random_mean='uniform') for i in range(num_arms)]

            
            # Randomize arm order to avoid position bias
            random.shuffle(arm_list)
            
            # Create streaming buffer that manages arm interactions within sliding window
            streaming_arm_buffer = Arm_Reading_Buffer(arm_set = np.array(arm_list), window_size = window_size)

            result_list = []
            # Test bucket exploration algorithm with different memory sizes
            for memory_size in memory_size_list:
                streaming_arm_buffer.reset()  # Reset buffer state for fair comparison
                result_list.append(bucket_exploration(streaming_arm_buffer, memory_size, delta))

            # Calculate performance metrics: difference between optimal and achieved rewards
            diff = streaming_arm_buffer.best_rewards - np.array(result_list)
            
            # Compute statistical summaries of the performance differences
            mean_diff = np.mean(diff, axis=1)      # Average regret across time
            max_diff = np.max(diff, axis=1)        # Worst-case regret
            median_diff = np.median(diff, axis=1)  # Median regret (robust to outliers)

            # Store results for this seed in the corresponding column
            y_mean[:, i_seed] = mean_diff
            y_median[:, i_seed] = median_diff
            y_max[:, i_seed] = max_diff

            # Baseline exploration algorithm for comparison
            result_list_baseline = []
            for memory_size in memory_size_list:
                streaming_arm_buffer.reset()  # Reset buffer state for fair comparison
                result_list_baseline.append(baseline_exploration_clear_expiration(streaming_arm_buffer, memory_size, delta))

            # Calculate performance metrics: difference between optimal and achieved rewards
            diff = streaming_arm_buffer.best_rewards - np.array(result_list_baseline)
            mean_diff_baseline = np.mean(diff, axis=1)      # Average regret across time
            max_diff_baseline = np.max(diff, axis=1)        # Worst-case regret
            median_diff_baseline = np.median(diff, axis=1)  # Median regret (robust to outliers)

            # Store results for this seed in the corresponding column
            y_mean_baseline[:, i_seed] = mean_diff_baseline
            y_median_baseline[:, i_seed] = median_diff_baseline
            y_max_baseline[:, i_seed] = max_diff_baseline

        algorithm_results = ExperimentResults(memory_size_list, y_mean, y_median, y_max)
        baseline_results = ExperimentResults(memory_size_list, y_mean_baseline, y_median_baseline, y_max_baseline)
        self.algorithm_results = algorithm_results
        self.baseline_results = baseline_results

    def plot_results(self, legend_loc="layout", range_type="range"):
        '''
        Plots the results of the experiment.
        '''
        if range_type == "range":
            self.plot_results_range(legend_loc)
        elif range_type == "error_bar":
            self.plot_results_error_bar(legend_loc)
        else:
            raise ValueError('The range_type parameter is not correctly setup.')

    def plot_results_range(self, legend_loc="layout"):
        '''
        Plots the results of the experiment with ranges.
        '''
        
        memory_size_list = self.algorithm_results.memory_sizes
        y_mean = self.algorithm_results.y_mean
        y_median = self.algorithm_results.y_median
        y_max = self.algorithm_results.y_max
        y_mean_baseline = self.baseline_results.y_mean
        y_median_baseline = self.baseline_results.y_median
        y_max_baseline = self.baseline_results.y_max

        # Analyze results across all seeds to understand algorithm performance and compare it with baseline

        # Compute central tendencies (averages across the 10 random seeds)
        mean_of_mean = [np.mean(yi) for yi in y_mean]     # Average of mean regrets
        mean_of_median = [np.mean(yi) for yi in y_median] # Average of median regrets
        mean_of_max = [np.mean(yi) for yi in y_max]       # Average of maximum regrets

        mean_of_mean_baseline = [np.mean(yi) for yi in y_mean_baseline]     # Average of mean regrets
        mean_of_median_baseline = [np.mean(yi) for yi in y_median_baseline] # Average of median regrets
        mean_of_max_baseline = [np.mean(yi) for yi in y_max_baseline]       # Average of maximum regrets

        # Compute ranges to show variability across seeds
        min_of_mean = [min(yi) for yi in y_mean]     # Best-case mean regret
        max_of_mean = [max(yi) for yi in y_mean]     # Worst-case mean regret

        min_of_mean_baseline = [min(yi) for yi in y_mean_baseline]     # Best-case mean regret (baseline)
        max_of_mean_baseline = [max(yi) for yi in y_mean_baseline]     # Worst-case mean regret (baseline)

        min_of_median = [min(yi) for yi in y_median] # Best-case median regret
        max_of_median = [max(yi) for yi in y_median] # Worst-case median regret

        min_of_median_baseline = [min(yi) for yi in y_median_baseline] # Best-case median regret (baseline)
        max_of_median_baseline = [max(yi) for yi in y_median_baseline] # Worst-case median regret (baseline)

        min_of_max = [min(yi) for yi in y_max]       # Best-case maximum regret
        max_of_max = [max(yi) for yi in y_max]       # Worst-case maximum regret

        min_of_max_baseline = [min(yi) for yi in y_max_baseline]       # Best-case maximum regret (baseline)
        max_of_max_baseline = [max(yi) for yi in y_max_baseline]       # Worst-case maximum regret (baseline)

        # Create comprehensive visualization showing performance vs memory size
        plt.figure(figsize=(10, 6))

        # Plot mean regret with shaded confidence region
        plt.fill_between(memory_size_list, min_of_mean, max_of_mean, color='blue', alpha=0.2, label='Range of mean')
        plt.plot(memory_size_list, mean_of_mean, 'o-', color='blue', label='Mean of mean')

        # Plot median regret with shaded confidence region
        plt.fill_between(memory_size_list, min_of_median, max_of_median, color='orange', alpha=0.2, label='Range of median')
        plt.plot(memory_size_list, mean_of_median, 'o-', color='orange', label='Mean of median')

        # Plot maximum regret with shaded confidence region
        plt.fill_between(memory_size_list, min_of_max, max_of_max, color='red', alpha=0.2, label='Range of max')
        plt.plot(memory_size_list, mean_of_max, 'o-', color='red', label='Mean of max')

        # Plot baseline mean regret with shaded confidence region
        plt.fill_between(memory_size_list, min_of_mean_baseline, max_of_mean_baseline, color='cyan', alpha=0.2, label='Baseline Range of mean')
        plt.plot(memory_size_list, mean_of_mean_baseline, 'o--', color='cyan', label='Baseline Mean of mean')

        # Plot baseline median regret with shaded confidence region
        plt.fill_between(memory_size_list, min_of_median_baseline, max_of_median_baseline, color='tan', alpha=0.2, label='Baseline Range of median')
        plt.plot(memory_size_list, mean_of_median_baseline, 'o--', color='tan', label='Baseline Mean of median')

        # Plot baseline maximum regret with shaded confidence region
        plt.fill_between(memory_size_list, min_of_max_baseline, max_of_max_baseline, color='pink', alpha=0.2, label='Baseline Range of max')
        plt.plot(memory_size_list, mean_of_max_baseline, 'o--', color='pink', label='Baseline Mean of max')

        # Configure plot appearance
        plt.xlabel('Memory size')
        plt.ylabel('Difference between the best arm and the output arm')
        plt.title('Performance of exploration algorithm when n = {}, W = {}'.format(self.n, self.window_size))
        
        if legend_loc == "layout":
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0
            )
            plt.tight_layout()
        elif legend_loc == "upper right":
            plt.legend(loc="upper right")
        else:
            raise ValueError('The legend_loc parameter is not correctly setup.')
        plt.grid()
        plt.show()

    def plot_results_error_bar(self, legend_loc="layout"):
        '''
        Plots the mean results of the experiment.
        '''
        # Create alternative visualization using error bars instead of shaded regions
        memory_size_list = self.algorithm_results.memory_sizes
        y_mean = self.algorithm_results.y_mean
        y_median = self.algorithm_results.y_median
        y_max = self.algorithm_results.y_max
        y_mean_baseline = self.baseline_results.y_mean
        y_median_baseline = self.baseline_results.y_median
        y_max_baseline = self.baseline_results.y_max

        # Compute central tendencies (averages across the 10 random seeds)
        mean_of_mean = [np.mean(yi) for yi in y_mean]     # Average of mean regrets
        mean_of_median = [np.mean(yi) for yi in y_median] # Average of median regrets
        mean_of_max = [np.mean(yi) for yi in y_max]       # Average of maximum regrets

        mean_of_mean_baseline = [np.mean(yi) for yi in y_mean_baseline]     # Average of mean regrets
        mean_of_median_baseline = [np.mean(yi) for yi in y_median_baseline] # Average of median regrets
        mean_of_max_baseline = [np.mean(yi) for yi in y_max_baseline]       # Average of maximum regrets

        # Compute ranges to show variability across seeds
        min_of_mean = [min(yi) for yi in y_mean]     # Best-case mean regret
        max_of_mean = [max(yi) for yi in y_mean]     # Worst-case mean regret

        min_of_mean_baseline = [min(yi) for yi in y_mean_baseline]     # Best-case mean regret (baseline)
        max_of_mean_baseline = [max(yi) for yi in y_mean_baseline]     # Worst-case mean regret (baseline)

        min_of_median = [min(yi) for yi in y_median] # Best-case median regret
        max_of_median = [max(yi) for yi in y_median] # Worst-case median regret

        min_of_median_baseline = [min(yi) for yi in y_median_baseline] # Best-case median regret (baseline)
        max_of_median_baseline = [max(yi) for yi in y_median_baseline] # Worst-case median regret (baseline)

        min_of_max = [min(yi) for yi in y_max]       # Best-case maximum regret
        max_of_max = [max(yi) for yi in y_max]       # Worst-case maximum regret

        min_of_max_baseline = [min(yi) for yi in y_max_baseline]       # Best-case maximum regret (baseline)
        max_of_max_baseline = [max(yi) for yi in y_max_baseline]       # Worst-case maximum regret (baseline)

        # Calculate error bar sizes (range of values across seeds)
        error_of_means = [max_val - min_val for max_val, min_val in zip(max_of_mean, min_of_mean)]
        error_of_medians = [max_val - min_val for max_val, min_val in zip(max_of_median, min_of_median)]
        error_of_maxs = [max_val - min_val for max_val, min_val in zip(max_of_max, min_of_max)]

        # Calculate error bar sizes for baseline
        error_of_means_baseline = [max_val - min_val for max_val, min_val in zip(max_of_mean_baseline, min_of_mean_baseline)]
        error_of_medians_baseline = [max_val - min_val for max_val, min_val in zip(max_of_median_baseline, min_of_median_baseline)]
        error_of_maxs_baseline = [max_val - min_val for max_val, min_val in zip(max_of_max_baseline, min_of_max_baseline)]

        # Create error bar plot showing variability across random seeds
        plt.figure(figsize=(10, 6))

        # Plot mean regret with error bars
        plt.errorbar(memory_size_list, mean_of_mean, yerr=error_of_means, fmt='o-', color='blue', label='Mean of mean')
        # Plot median regret with error bars
        plt.errorbar(memory_size_list, mean_of_median, yerr=error_of_medians, fmt='o-', color='orange', label='Mean of median')
        # Plot maximum regret with error bars
        plt.errorbar(memory_size_list, mean_of_max, yerr=error_of_maxs, fmt='o-', color='red', label='Mean of max')
        # Plot baseline mean regret with error bars
        plt.errorbar(memory_size_list, mean_of_mean_baseline, yerr=error_of_means_baseline, fmt='o--', color='cyan', label='Baseline Mean of mean')
        # Plot baseline median regret with error bars
        plt.errorbar(memory_size_list, mean_of_median_baseline, yerr=error_of_medians_baseline, fmt='o--', color='tan', label='Baseline Mean of median')
        # Plot baseline maximum regret with error bars
        plt.errorbar(memory_size_list, mean_of_max_baseline, yerr=error_of_maxs_baseline, fmt='o--', color='pink', label='Baseline Mean of max')
        # Configure plot appearance
        plt.xlabel('Memory size')
        plt.ylabel('Difference between the best arm and the output arm')
        plt.title('Performance of exploration algorithm when n = {}, W = {}'.format(self.n, self.window_size))
        if legend_loc == "layout":
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0
            )
            plt.tight_layout()
        elif legend_loc == "upper right":
            plt.legend(loc="upper right")
        else:
            raise ValueError('The legend_loc parameter is not correctly setup.')
        plt.grid()
        plt.show()