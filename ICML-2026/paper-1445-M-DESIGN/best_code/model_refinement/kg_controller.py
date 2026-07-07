# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

import random
import numpy as np
from collections import Counter
from scipy.stats import linregress, norm


design_dimensions = {
    'node_classification': ['neigh', 'norm', 'agg', 'comb', 'l_mp', 'stage'],
    'link_prediction': ['decode', 'norm', 'agg', 'comb', 'l_mp', 'stage'],
    'graph_classification': ['decode', 'norm', 'agg', 'comb', 'l_mp', 'stage']
}


def refine_model_with_bayesian(unseen_dataset, task, top_s_benchmarks, knowledge_retrieval, knowledge_estimator, initial_strategy, max_iter=30, window_size=None, response_save_path=None):
    # Initialize variables for storing consistency metrics
    iteration_numbers = []

    # Transform initial similarities to posterior probability
    if any(similarity < 0 for _, similarity in top_s_benchmarks):
        dynamic_similarities = {benchmark: (similarity + 1) / 2 for benchmark, similarity in top_s_benchmarks}
    else:
        dynamic_similarities = {benchmark: similarity for benchmark, similarity in top_s_benchmarks}
    total_similarity = sum(dynamic_similarities.values())
    dynamic_similarities = {benchmark: [similarity / total_similarity] for benchmark, similarity in dynamic_similarities.items()}

    # Initialize variables for Bayesian updating
    bayesian_data = {}
    for benchmark_dataset_name in dynamic_similarities.keys():
        bayesian_data[benchmark_dataset_name] = {
            'observed_pairs': [],       # list of tuples (delta_P_u, delta_P_i)
            'beta': 1.0,                # initial estimate of beta_i
            'sigma2': 0.01              # initial estimate of sigma^2
        }

    # Step 1: Initialize with the best model from the benchmark dataset
    best_models = []
    hyperparameters = design_dimensions[task]
    for benchmark_dataset_name, _ in top_s_benchmarks:
        best_model = knowledge_retrieval.retrieve_top_n_best(benchmark_dataset_name, n=1)
        model_tuple = best_model[0]
        current_model = dict(zip(hyperparameters, model_tuple[:6]))
        best_models.append(current_model)

    most_similar_benchmark_idx = 0
    if initial_strategy == 'best':
        # Select the best model from the benchmark datasets
        current_model = best_models[0]
    elif initial_strategy == 'majority_vote':
        # Majority vote for each hyperparameter
        current_model = {}
        for hp in hyperparameters:
            hp_values = [model[hp] for model in best_models]
            most_common_value, _ = Counter(hp_values).most_common(1)[0]
            current_model[hp] = most_common_value
    elif initial_strategy == 'weighted_average':
        # Update weights based on similarity
        weighted_choices = {hp: Counter() for hp in hyperparameters}
        for i, (benchmark_dataset_name, similarity_score) in enumerate(top_s_benchmarks):
            for hp in hyperparameters:
                weighted_choices[hp][best_models[i][hp]] += similarity_score
        current_model = {hp: weighted_choices[hp].most_common(1)[0][0] for hp in hyperparameters}
    else:
        raise ValueError(f"Initial strategy {initial_strategy} not supported")

    # Evaluate the initial transferred model on the unseen dataset
    current_accuracy = knowledge_retrieval.evaluate_model(unseen_dataset, current_model)
    # Handle failed evaluations (sentinel score -1.0)
    if current_accuracy[0] <= 0.0:
        print(f"WARNING: Initial evaluation failed for {current_model}. Score: {current_accuracy}")
        # Try the first best architecture from the most similar benchmark
        current_model = best_models[0]
        current_accuracy = knowledge_retrieval.evaluate_model(unseen_dataset, current_model)
        if current_accuracy[0] <= 0.0:
            raise RuntimeError(f"Initial architecture evaluation failed even for fallback. Aborting.")
    accuracy_history = [(current_accuracy[0], current_accuracy[1])]
    tried_models = {tuple(current_model.values())}
    initial_proposal = (current_model, current_accuracy[0], current_accuracy[1])
    overall_best_model = (current_model, current_accuracy[0], current_accuracy[1])

    # Step 2: Refinement loop using weighted average
    for i in range(max_iter):
        best_model = None
        modification_dict = {}

        # Extract the most recent similarity values for each benchmark
        # Calculate the total sum of similarity values
        current_similarity = {benchmark: vals[-1] for benchmark, vals in dynamic_similarities.items()}

        # Generate modifications from each benchmark dataset
        for benchmark_idx, (benchmark_dataset_name, _) in enumerate(top_s_benchmarks):
            current_benchmark_accuracy = knowledge_retrieval.retrieve_model(benchmark_dataset_name, current_model)[0]
            modifications = knowledge_retrieval.get_all_one_step_modifications(benchmark_dataset_name, current_model)

            # Calculate the predicted gains for each modification
            if knowledge_estimator is not None and current_similarity[benchmark_dataset_name] < 0.2:
                estimator = knowledge_estimator.estimators[benchmark_dataset_name]
                data_obj = knowledge_estimator.data_cache[benchmark_dataset_name]
                predicted_gains = estimator.predict_benchmark_gains(data_obj, current_model, modifications, knowledge_estimator.device)
            else:
                predicted_gains = None
            last_model = current_model

            # Calculate the predicted gains for each modification
            for mod_idx, (modified_model, new_benchmark_accuracy, _) in enumerate(modifications):
                mod_tuple = tuple(modified_model.values())

                # Obtain the benchmark gain for the current modification
                if predicted_gains is not None:
                    benchmark_gain = predicted_gains[mod_idx]
                else:
                    benchmark_gain = new_benchmark_accuracy - current_benchmark_accuracy

                # Aggregate gains for modification selection
                if mod_tuple not in modification_dict:
                    modification_dict[mod_tuple] = []
                modification_dict[mod_tuple].append((benchmark_gain * current_similarity[benchmark_dataset_name], benchmark_gain, benchmark_idx))

        # Update the index of the most similar benchmark dataset
        most_similar_benchmark_name = max(current_similarity, key=current_similarity.get)
        for idx, (benchmark_dataset_name, _) in enumerate(top_s_benchmarks):
            if benchmark_dataset_name == most_similar_benchmark_name:
                most_similar_benchmark_idx = idx
                break

        # Build a list of modifications with their weighted average gain and gain on most similar benchmark
        modification_gains = []
        for mod_tuple, gains in modification_dict.items():
            #avg_weighted_gain = sum(gain for gain, _, _ in gains) / len(gains)
            weighted_gain = sum(gain for gain, _, _ in gains)

            # Use the gain from the most similar benchmark for tie
            gain_from_most_similar = None
            for _, benchmark_gain, benchmark_idx in gains:
                if benchmark_idx == most_similar_benchmark_idx:
                    gain_from_most_similar = benchmark_gain
                    break
            modification_gains.append((weighted_gain, gain_from_most_similar, mod_tuple))
        
        # Sort mod_list in decreasing order of weighted average gain, then gain_on_similar_benchmark
        modification_gains.sort(key=lambda x: (-x[0], -x[1]))

        # Select modification: epsilon-greedy exploration
        # With probability epsilon, randomly explore instead of exploit
        epsilon = max(0.03, 0.25 * (0.97 ** i))
        if random.random() < epsilon and len(modification_gains) > 1:
            untried = [(wg, gs, mt) for wg, gs, mt in modification_gains if mt not in tried_models]
            if untried:
                idx = random.randint(0, len(untried) - 1)
                best_model = untried[idx][2]
            else:
                best_model = None
        else:
            for weighted_gain, _, mod_tuple in modification_gains:
                if mod_tuple not in tried_models:
                    best_model = mod_tuple
                    break
        
        # If no new modification is found, stop refinement
        if best_model is None:
            print(f"No further new modification found at iteration {i}. Final model for {unseen_dataset}: {current_model}")
            break

        # Update current_model
        current_model = dict(zip(hyperparameters, best_model))
        tried_models.add(best_model)
        new_accuracy = knowledge_retrieval.evaluate_model(unseen_dataset, current_model)
        # Handle failed evaluations (sentinel score -1.0)
        if new_accuracy[0] <= 0.0:
            print(f"WARNING: Evaluation failed for {current_model}. Skipping this architecture.")
            accuracy_history.append((current_accuracy[0], current_accuracy[1]))  # repeat last good score
            continue
        accuracy_history.append((new_accuracy[0], new_accuracy[1]))
        unseen_gain = new_accuracy[0] - current_accuracy[0]

        if new_accuracy[0] > overall_best_model[1]:
            overall_best_model = (current_model, new_accuracy[0], new_accuracy[1])
        current_accuracy = new_accuracy

        # Store the observed pair for Bayesian updating
        finetune_datasets = []
        benchmark_gains = modification_dict[best_model]
        for idx, (benchmark_dataset_name, _) in enumerate(top_s_benchmarks):
            for _, benchmark_gain, benchmark_idx in benchmark_gains:
                if idx == benchmark_idx:
                    bayesian_data[benchmark_dataset_name]['observed_pairs'].append((unseen_gain, benchmark_gain))
                    if i > 10 and current_similarity[benchmark_dataset_name] < 0.2:
                        finetune_datasets.append(benchmark_dataset_name)
                    break
        if knowledge_estimator is not None:
            knowledge_estimator.add_feedback_observation(last_model, current_model, unseen_gain, current_similarity)
            if len(finetune_datasets) > 0:
                knowledge_estimator.feedback_integration(finetune_datasets)

        # Perform Bayesian updating
        likelihoods = {}
        total_likelihood = 0.0
        for benchmark_dataset_name, data in bayesian_data.items():
            observed_pairs = data['observed_pairs']
            
            if len(observed_pairs) >= 3:
                # Prepare data for linear regression
                if window_size:
                    delta_P_u = np.array([pair[0] for pair in observed_pairs[-1 * window_size:]])
                    delta_P_i = np.array([pair[1] for pair in observed_pairs[-1 * window_size:]])
                else:
                    delta_P_u = np.array([pair[0] for pair in observed_pairs])
                    delta_P_i = np.array([pair[1] for pair in observed_pairs])
                
                if np.all(delta_P_i == delta_P_i[0]):
                    beta_i = data['beta']
                    sigma2 = data['sigma2']
                else:
                    # Perform linear regression to estimate beta_i and sigma^2
                    beta_i, _, _, _, _ = linregress(delta_P_i, delta_P_u)
                    residuals = delta_P_u - (beta_i * delta_P_i)
                    sigma2 = np.var(residuals) if len(residuals) > 1 else np.var(delta_P_u - delta_P_u.mean())
                    if sigma2 == 0:
                        sigma2 = 1e-6  # Prevent division by zero
                    # Update beta and sigma2 in data
                    data['beta'] = beta_i
                    data['sigma2'] = sigma2
            else:
                # Not enough data, use prior estimates
                beta_i = data['beta']
                sigma2 = data['sigma2']

            # Compute likelihood for the latest observed pair
            delta_P_u_latest, delta_P_i_latest = observed_pairs[-1]
            likelihood = norm.pdf(delta_P_u_latest, loc=beta_i * delta_P_i_latest, scale=np.sqrt(sigma2))
            likelihoods[benchmark_dataset_name] = likelihood
            total_likelihood += likelihood * current_similarity[benchmark_dataset_name]
        
        # Update similarities using Bayesian update rule
        for benchmark_dataset_name in current_similarity:
            prior = current_similarity[benchmark_dataset_name]
            likelihood = likelihoods.get(benchmark_dataset_name, 0.0)
            posterior = (likelihood * prior) / total_likelihood if total_likelihood > 0 else prior
            current_similarity[benchmark_dataset_name] = posterior
        
        # Normalize similarities
        total_similarity = sum(current_similarity.values())
        if total_similarity > 0:
            for key in current_similarity:
                current_similarity[key] /= total_similarity
        
        # At the end of each iteration, store current similarities and ground truth
        iteration_numbers.append(i)
        for benchmark in current_similarity:
            dynamic_similarities[benchmark].append(current_similarity[benchmark])
    
    return initial_proposal, overall_best_model, accuracy_history

