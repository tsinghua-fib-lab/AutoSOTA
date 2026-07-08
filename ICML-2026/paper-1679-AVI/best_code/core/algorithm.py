import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
from .e_value import compute_e_value, compute_e_value_covariate
from .transitivity import propagate_transitivity
from .confidence_sets import compute_topk_confidence_set


def update_model_strength(win_records, total_comparisons, m):
    
    strengths = np.zeros(m)
    
    
    for i in range(m):
        total_wins = 0
        total_games = 0
        
        for j in range(m):
            if i != j and total_comparisons[i, j] > 0:
                total_wins += win_records[i, j]
                total_games += total_comparisons[i, j]
        
        if total_games > 0:
            
            win_rate = total_wins / total_games
            
            
            opponent_strength = 0
            opponent_count = 0
            
            for j in range(m):
                if i != j and total_comparisons[i, j] > 0:
                    opp_wins = 0
                    opp_games = 0
                    for k in range(m):
                        if j != k and total_comparisons[j, k] > 0:
                            opp_wins += win_records[j, k]
                            opp_games += total_comparisons[j, k]
                    
                    if opp_games > 0:
                        opp_strength = opp_wins / opp_games
                        opponent_strength += opp_strength
                        opponent_count += 1
            
            if opponent_count > 0:
                avg_opponent_strength = opponent_strength / opponent_count
                strengths[i] = win_rate * (1 + avg_opponent_strength) / 2
            else:
                strengths[i] = win_rate
        else:
            strengths[i] = 0.5  
    
    return strengths


def serpant_algorithm(m, alpha, true_probs, max_t=1000,
                     sampling_method="all_active", verbose=True,
                     max_tournament_samples=10, top_k=None, priority_mode="weighted",
                     uncertainty_weight=0.2, signal_strength_weight=0.8):
    
    
    t = 0
    K = m * (m - 1) / (2 * alpha)
    
    A = np.ones((m, m), dtype=bool)
    np.fill_diagonal(A, False)
    
    e_values = np.ones((m, m))
    np.fill_diagonal(e_values, 0)
    
    successes = np.zeros((m, m))
    trials = np.zeros((m, m))
    R = np.zeros((m, m), dtype=bool)
    
    
    tournament_round = 1
    current_tournament_pairs = []
    tournament_sample_counts = np.zeros((m, m))
    tournament_pair_index = 1
    
    win_records = np.zeros((m, m))
    total_comparisons = np.zeros((m, m))
    model_strength = np.zeros(m)
    
    results = []
    last_rank_confidence = None
    last_topk_set = None
    
    
    # Support callable observation functions for real-mode experiments
    _is_callable = callable(true_probs)
    def _get_obs(j, i, t_val):
        if _is_callable:
            return true_probs(j, i, t_val)
        return np.random.binomial(1, true_probs[j, i])
    
    while t < max_t and A.sum() > 0:
        t += 1
        R_t = np.zeros((m, m), dtype=bool)
        
        
        if sampling_method == "all_active":
            for j in range(1, m):
                for i in range(j):
                    if A[j, i]:
                        obs = _get_obs(j, i, t)
                        successes[j, i] += obs
                        trials[j, i] += 1
                        e_values[j, i] = compute_e_value(successes[j, i], trials[j, i])
                        if e_values[j, i] >= K:
                            R_t[j, i] = True
                    
                    if A[i, j]:
                        successes[i, j] += (1 - obs)
                        trials[i, j] += 1
                        e_values[i, j] = compute_e_value(successes[i, j], trials[i, j])
                        if e_values[i, j] >= K:
                            R_t[i, j] = True
        
        elif sampling_method == "random_pair":
            active_idx = np.argwhere(A & np.tri(m, k=-1, dtype=bool).T)
            if len(active_idx) > 0:
                sel = np.random.choice(len(active_idx))
                j, i = active_idx[sel]
                obs = _get_obs(j, i, t)
                
                successes[j, i] += obs
                trials[j, i] += 1
                e_values[j, i] = compute_e_value(successes[j, i], trials[j, i])
                if e_values[j, i] >= K:
                    R_t[j, i] = True
                
                if A[i, j]:
                    successes[i, j] += (1 - obs)
                    trials[i, j] += 1
                    e_values[i, j] = compute_e_value(successes[i, j], trials[i, j])
                    if e_values[i, j] >= K:
                        R_t[i, j] = True
        
        elif sampling_method == "tournament":
            
            if (len(current_tournament_pairs) == 0 or 
                (tournament_round > 1 and tournament_pair_index > len(current_tournament_pairs))):
                
                current_tournament_pairs = []
                tournament_pair_index = 1
                tournament_round += 1
                
                model_strength = update_model_strength(win_records, total_comparisons, m)
                active_pairs = np.argwhere(A & np.tri(m, k=-1, dtype=bool).T)
                
                if len(active_pairs) == 0:
                    continue
                
                
                is_exploration = (tournament_round % 4 == 1)
                
                if is_exploration:
                    
                    samples_so_far = trials[active_pairs[:, 0], active_pairs[:, 1]]
                    quantile_10 = np.percentile(samples_so_far, 10)
                    under_sampled = np.where(samples_so_far <= quantile_10)[0]
                    n_sample = min(20, len(under_sampled) if len(under_sampled) > 0 else len(active_pairs))
                    sel_idx = np.random.choice(
                        under_sampled if len(under_sampled) > 0 else range(len(active_pairs)),
                        n_sample, replace=False
                    )
                    selected_pairs = active_pairs[sel_idx]
                else:
                    
                    priorities = np.zeros(len(active_pairs))
                    dynamic_max = np.zeros(len(active_pairs))
                    
                    for idx in range(len(active_pairs)):
                        j, i = active_pairs[idx]
                        s = successes[j, i]
                        n = trials[j, i]
                        win_rate = s / n if n > 0 else 0.5
                        est_prob = (s + 1) / (n + 2)
                        e_val = e_values[j, i]
                        
                        uncertainty = 1 - abs(win_rate - 0.5) * 2
                        proximity_to_K = 1 / (1 + np.exp(5 * (np.log(K) - np.log(max(e_val, 1e-10)))))
                        signal_strength = abs(est_prob - 0.5) * 2
                        
                        if priority_mode == "max":
                            
                            priorities[idx] = max(uncertainty, signal_strength)
                        elif priority_mode == "weighted_no_proximity":
                            
                            priorities[idx] = uncertainty_weight*uncertainty + signal_strength_weight*signal_strength
                        else:
                            
                            priorities[idx] = 0.2*uncertainty + 0.4*proximity_to_K + 0.4*signal_strength
                            
                        
                        
                        diff = abs(est_prob - 0.5)
                        if diff > 0.3:
                            dynamic_max[idx] = np.inf
                        elif diff > 0.2:
                            dynamic_max[idx] = max_tournament_samples * 3
                        else:
                            dynamic_max[idx] = max_tournament_samples
                    
                    max_pairs = min(70, max(1, int(len(active_pairs) * 0.9)))
                    top_idx = np.argsort(priorities)[::-1][:max_pairs]
                    selected_pairs = active_pairs[top_idx]
                
                
                for k in range(len(selected_pairs)):
                    j, i = selected_pairs[k]
                    if is_exploration:
                        max_s = max_tournament_samples
                    else:
                        orig_idx = np.where((active_pairs[:, 0] == j) & (active_pairs[:, 1] == i))[0][0]
                        max_s = dynamic_max[orig_idx]
                    
                    current_tournament_pairs.append({
                        'pair': (j, i),
                        'max_samples': max_s
                    })
                
                if verbose and t % 200 == 0:
                    print(f"Time {t}: Tournament selected {len(current_tournament_pairs)} pairs", flush=True)
            
            
            if tournament_pair_index <= len(current_tournament_pairs):
                info = current_tournament_pairs[tournament_pair_index - 1]
                j, i = info['pair']
                max_s = info['max_samples']
                
                if A[j, i] and tournament_sample_counts[j, i] < max_s:
                    obs = _get_obs(j, i, t)
                    
                    successes[j, i] += obs
                    trials[j, i] += 1
                    tournament_sample_counts[j, i] += 1
                    
                    if obs == 1:
                        win_records[j, i] += 1
                    else:
                        win_records[i, j] += 1
                    total_comparisons[j, i] += 1
                    total_comparisons[i, j] += 1
                    
                    e_values[j, i] = compute_e_value(successes[j, i], trials[j, i])
                    if e_values[j, i] >= K:
                        R_t[j, i] = True
                    
                    if A[i, j] and tournament_sample_counts[i, j] < max_s:
                        successes[i, j] += (1 - obs)
                        trials[i, j] += 1
                        tournament_sample_counts[i, j] += 1
                        e_values[i, j] = compute_e_value(successes[i, j], trials[i, j])
                        if e_values[i, j] >= K:
                            R_t[i, j] = True
                
                tournament_pair_index += 1
        
        else:
            raise ValueError("Unknown sampling method.")
        

        prev_count = R_t.sum()
        while True:
            T_set = propagate_transitivity(R, R_t, m)
            R_t = R_t | T_set
            
            current_count = R_t.sum()
            if current_count == prev_count:
                
                break
            prev_count = current_count
        

        A = A & ~R_t
        R = R | R_t

        newly_rejected = np.argwhere(R_t)
        for j, i in newly_rejected:
            A[i, j] = False
        
        rank_confidence = None
        top_k_set = None
        if top_k is not None:
            topk_info = compute_topk_confidence_set(R, m, top_k)
            rank_confidence = topk_info['rank_confidence']
            top_k_set = topk_info['topk_models']
            last_rank_confidence = rank_confidence
            last_topk_set = top_k_set
        
        results.append({
            'time': t,
            'rejected_count': R.sum(),
            'active_count': A.sum(),
            'round_rejections': R_t.sum(),
            'sampling_method': sampling_method,
            'rejected_pairs': np.argwhere(R),
            'rank_confidence': rank_confidence,
            'top_k_set': top_k_set
        })
        
        if verbose and t % 100 == 0:
            tour_info = ""
            if sampling_method == "tournament":
                tour_info = f", Round: {tournament_round - 1}, Sampled: {tournament_pair_index - 1}/{len(current_tournament_pairs)}"
            print(f"Time {t}: Rejected {R.sum():.0f} (new: {R_t.sum():.0f}), Active {A.sum():.0f} [tournament{tour_info}]", flush=True)
    
    return {
        'final_rejected': R,
        'results': results,
        'e_values': e_values,
        'successes': successes,
        'trials': trials,
        'partial_order': np.argwhere(R),
        'sampling_method': sampling_method,
        'rank_confidence': last_rank_confidence if top_k is not None else None,
        'top_k_confidence_set': last_topk_set if top_k is not None else None,
        'top_k': top_k
    }


def fit_bt_estimator(X_history, z_history, use_sklearn=True):

    X = np.array(X_history)
    z = np.array(z_history)
    
    if len(X) == 0:
        return 0.0
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, d = X.shape
    
    if n_samples < 2:
        return np.zeros(d) if d > 1 else 0.0
    
    unique_z = np.unique(z)
    if len(unique_z) < 2:
        return np.zeros(d) if d > 1 else 0.0
    
    if use_sklearn:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                warnings.filterwarnings('ignore', category=FutureWarning)
                model = LogisticRegression(
                    l1_ratio=0,
                    C=100.0,
                    fit_intercept=False,
                    solver='lbfgs',
                    max_iter=200,
                    warm_start=False
                )
                model.fit(X, z)
                theta = model.coef_.flatten()
                return theta if d > 1 else theta[0]
        except Exception:
            pass
    
    theta = np.zeros(d)
    
    for _ in range(100):
        linear = X @ theta
        linear = np.clip(linear, -20, 20)
        p = 1 / (1 + np.exp(-linear))
        
        w = p * (1 - p)
        w = np.maximum(w, 1e-6)
        
        gradient = X.T @ (z - p)
        W = np.diag(w)
        H = -X.T @ W @ X
        H_reg = H - 1e-6 * np.eye(d)
        
        try:
            delta = np.linalg.solve(H_reg, gradient)
        except np.linalg.LinAlgError:
            break
        
        theta_new = theta - delta
        
        if np.max(np.abs(delta)) < 1e-6:
            break
        
        theta = theta_new
    
    return theta if d > 1 else theta[0] if d == 1 else 0.0


def predict_bt(X, theta):


    X = np.atleast_1d(X)
    theta = np.atleast_1d(theta)
    
    if X.ndim == 1 and theta.ndim == 1:
        linear = np.dot(X, theta)
    else:
        linear = X @ theta
    
    linear = np.clip(linear, -20, 20)
    p = 1 / (1 + np.exp(-linear))
    return p


def serpant_algorithm_covariate(m, alpha, true_probs, covariate_info, max_t=1000,
                                sampling_method="random_pair", verbose=True,
                                max_tournament_samples=10, top_k=None,
                                theta_update_interval=1, priority_mode="weighted",
                                uncertainty_weight=0.2, signal_strength_weight=0.8):

    t = 0
    K = m * (m - 1) / (2 * alpha)
    
    A = np.ones((m, m), dtype=bool)
    np.fill_diagonal(A, False)
    
    e_values = np.ones((m, m))
    np.fill_diagonal(e_values, 0)
    
    successes = np.zeros((m, m))
    trials = np.zeros((m, m))
    R = np.zeros((m, m), dtype=bool)
    
    observations_history = [[[] for _ in range(m)] for _ in range(m)]
    covariates_history = [[[] for _ in range(m)] for _ in range(m)]
    bt_predictions_history = [[[] for _ in range(m)] for _ in range(m)]
    
    theta_cache = [[None for _ in range(m)] for _ in range(m)]
    
    X = covariate_info.get('X', np.zeros(m))
    
    tournament_round = 1
    current_tournament_pairs = []
    tournament_sample_counts = np.zeros((m, m))
    tournament_pair_index = 1
    
    win_records = np.zeros((m, m))
    total_comparisons = np.zeros((m, m))
    model_strength = np.zeros(m)
    
    results = []
    last_rank_confidence = None
    last_topk_set = None
    
    # Support callable observation functions for real-mode experiments
    _is_callable = callable(true_probs)
    def _get_obs(j, i, t_val):
        if _is_callable:
            return true_probs(j, i, t_val)
        return np.random.binomial(1, true_probs[j, i])
    
    while t < max_t and A.sum() > 0:
        t += 1
        R_t = np.zeros((m, m), dtype=bool)
        
        if sampling_method == "random_pair":
            active_idx = np.argwhere(A & np.tri(m, k=-1, dtype=bool).T)
            if len(active_idx) > 0:
                sel = np.random.choice(len(active_idx))
                j, i = active_idx[sel]
                obs = _get_obs(j, i, t)
                
                x_t = X[j] - X[i]
                
                successes[j, i] += obs
                trials[j, i] += 1
                
                observations_history[j][i].append(obs)
                covariates_history[j][i].append(x_t)
                
                n_obs = len(observations_history[j][i])
                if n_obs >= 1:
                    should_update_theta = (n_obs % theta_update_interval == 0) or (n_obs == 1)
                    
                    if should_update_theta:
                        theta_ji = fit_bt_estimator(
                            covariates_history[j][i],
                            observations_history[j][i]
                        )
                        theta_cache[j][i] = theta_ji
                    else:
                        theta_ji = theta_cache[j][i] if theta_cache[j][i] is not None else 0.0
                    
                    predictions = []
                    for s_idx in range(n_obs):
                        x_s = covariates_history[j][i][s_idx]
                        pred_s = predict_bt(x_s, theta_ji)
                        predictions.append(pred_s)
                    
                    cumsum_preds = np.cumsum(predictions)
                    bt_avg_predictions = [cumsum_preds[s] / (s + 1) for s in range(n_obs)]
                    bt_predictions_history[j][i] = bt_avg_predictions
                    
                    e_values[j, i] = compute_e_value_covariate(
                        observations_history[j][i],
                        bt_predictions_history[j][i],
                        successes[j, i],
                        int(trials[j, i])
                    )
                
                if e_values[j, i] >= K:
                    R_t[j, i] = True
                
                if A[i, j]:
                    successes[i, j] += (1 - obs)
                    trials[i, j] += 1
                    
                    x_t_reverse = -x_t
                    observations_history[i][j].append(1 - obs)
                    covariates_history[i][j].append(x_t_reverse)
                    
                    n_obs_rev = len(observations_history[i][j])
                    if n_obs_rev >= 1:
                        should_update_theta_rev = (n_obs_rev % theta_update_interval == 0) or (n_obs_rev == 1)
                        
                        if should_update_theta_rev:
                            theta_ij = fit_bt_estimator(
                                covariates_history[i][j],
                                observations_history[i][j]
                            )
                            theta_cache[i][j] = theta_ij
                        else:
                            theta_ij = theta_cache[i][j] if theta_cache[i][j] is not None else 0.0
                        
                        predictions_rev = []
                        for s_idx in range(n_obs_rev):
                            x_s = covariates_history[i][j][s_idx]
                            pred_s = predict_bt(x_s, theta_ij)
                            predictions_rev.append(pred_s)
                        
                        cumsum_preds_rev = np.cumsum(predictions_rev)
                        bt_avg_predictions_rev = [cumsum_preds_rev[s] / (s + 1) for s in range(n_obs_rev)]
                        bt_predictions_history[i][j] = bt_avg_predictions_rev
                        
                        e_values[i, j] = compute_e_value_covariate(
                            observations_history[i][j],
                            bt_predictions_history[i][j],
                            successes[i, j],
                            int(trials[i, j])
                        )
                    
                    if e_values[i, j] >= K:
                        R_t[i, j] = True
        
        elif sampling_method == "tournament":
            if (len(current_tournament_pairs) == 0 or 
                (tournament_round > 1 and tournament_pair_index > len(current_tournament_pairs))):
                
                current_tournament_pairs = []
                tournament_pair_index = 1
                tournament_round += 1
                
                model_strength = update_model_strength(win_records, total_comparisons, m)
                active_pairs = np.argwhere(A & np.tri(m, k=-1, dtype=bool).T)
                
                if len(active_pairs) == 0:
                    continue
                
                is_exploration = (tournament_round % 4 == 1)
                
                if is_exploration:
                    samples_so_far = trials[active_pairs[:, 0], active_pairs[:, 1]]
                    quantile_10 = np.percentile(samples_so_far, 10)
                    under_sampled = np.where(samples_so_far <= quantile_10)[0]
                    n_sample = min(20, len(under_sampled) if len(under_sampled) > 0 else len(active_pairs))
                    sel_idx = np.random.choice(
                        under_sampled if len(under_sampled) > 0 else range(len(active_pairs)),
                        n_sample, replace=False
                    )
                    selected_pairs = active_pairs[sel_idx]
                else:
                    priorities = np.zeros(len(active_pairs))
                    dynamic_max = np.zeros(len(active_pairs))
                    
                    for idx in range(len(active_pairs)):
                        j, i = active_pairs[idx]
                        s = successes[j, i]
                        n = trials[j, i]
                        win_rate = s / n if n > 0 else 0.5
                        est_prob = (s + 1) / (n + 2)
                        e_val = e_values[j, i]
                        
                        uncertainty = 1 - abs(win_rate - 0.5) * 2
                        proximity_to_K = 1 / (1 + np.exp(5 * (np.log(K) - np.log(max(e_val, 1e-10)))))
                        signal_strength = abs(est_prob - 0.5) * 2
                        
                        if priority_mode == "max":
                            priorities[idx] = max(uncertainty, signal_strength)
                        elif priority_mode == "weighted_no_proximity":
                            priorities[idx] = uncertainty_weight*uncertainty + signal_strength_weight*signal_strength
                        else:
                            priorities[idx] = 0.2*uncertainty + 0.4*proximity_to_K + 0.4*signal_strength
                        
                        diff = abs(est_prob - 0.5)
                        if diff > 0.3:
                            dynamic_max[idx] = np.inf
                        elif diff > 0.2:
                            dynamic_max[idx] = max_tournament_samples * 3
                        else:
                            dynamic_max[idx] = max_tournament_samples
                    
                    max_pairs = min(70, max(1, int(len(active_pairs) * 0.9)))
                    top_idx = np.argsort(priorities)[::-1][:max_pairs]
                    selected_pairs = active_pairs[top_idx]
                
                for k in range(len(selected_pairs)):
                    j, i = selected_pairs[k]
                    if is_exploration:
                        max_s = max_tournament_samples
                    else:
                        orig_idx = np.where((active_pairs[:, 0] == j) & (active_pairs[:, 1] == i))[0][0]
                        max_s = dynamic_max[orig_idx]
                    
                    current_tournament_pairs.append({
                        'pair': (j, i),
                        'max_samples': max_s
                    })
                
                if verbose and t % 200 == 0:
                    print(f"Time {t}: Tournament selected {len(current_tournament_pairs)} pairs", flush=True)
            
            if tournament_pair_index <= len(current_tournament_pairs):
                info = current_tournament_pairs[tournament_pair_index - 1]
                j, i = info['pair']
                max_s = info['max_samples']
                
                if A[j, i] and tournament_sample_counts[j, i] < max_s:
                    obs = _get_obs(j, i, t)
                    x_t = X[j] - X[i]
                    
                    successes[j, i] += obs
                    trials[j, i] += 1
                    tournament_sample_counts[j, i] += 1
                    
                    if obs == 1:
                        win_records[j, i] += 1
                    else:
                        win_records[i, j] += 1
                    total_comparisons[j, i] += 1
                    total_comparisons[i, j] += 1
                    
                    observations_history[j][i].append(obs)
                    covariates_history[j][i].append(x_t)
                    
                    n_obs = len(observations_history[j][i])
                    if n_obs >= 1:

                        should_update_theta = (n_obs % theta_update_interval == 0) or (n_obs == 1)
                        
                        if should_update_theta:
                            theta_ji = fit_bt_estimator(
                                covariates_history[j][i],
                                observations_history[j][i]
                            )
                            theta_cache[j][i] = theta_ji
                        else:
                            theta_ji = theta_cache[j][i] if theta_cache[j][i] is not None else 0.0
                        
                        predictions = []
                        for s_idx in range(n_obs):
                            x_s = covariates_history[j][i][s_idx]
                            pred_s = predict_bt(x_s, theta_ji)
                            predictions.append(pred_s)
                        
                        cumsum_preds = np.cumsum(predictions)
                        bt_avg_predictions = [cumsum_preds[s] / (s + 1) for s in range(n_obs)]
                        bt_predictions_history[j][i] = bt_avg_predictions
                        
                        e_values[j, i] = compute_e_value_covariate(
                            observations_history[j][i],
                            bt_predictions_history[j][i],
                            successes[j, i],
                            int(trials[j, i])
                        )
                    
                    if e_values[j, i] >= K:
                        R_t[j, i] = True
                    
                    if A[i, j] and tournament_sample_counts[i, j] < max_s:
                        successes[i, j] += (1 - obs)
                        trials[i, j] += 1
                        tournament_sample_counts[i, j] += 1
                        
                        x_t_reverse = -x_t
                        observations_history[i][j].append(1 - obs)
                        covariates_history[i][j].append(x_t_reverse)
                        
                        n_obs_rev = len(observations_history[i][j])
                        if n_obs_rev >= 1:
                            should_update_theta_rev = (n_obs_rev % theta_update_interval == 0) or (n_obs_rev == 1)
                            
                            if should_update_theta_rev:
                                theta_ij = fit_bt_estimator(
                                    covariates_history[i][j],
                                    observations_history[i][j]
                                )
                                theta_cache[i][j] = theta_ij
                            else:
                                theta_ij = theta_cache[i][j] if theta_cache[i][j] is not None else 0.0
                            
                            predictions_rev = []
                            for s_idx in range(n_obs_rev):
                                x_s = covariates_history[i][j][s_idx]
                                pred_s = predict_bt(x_s, theta_ij)
                                predictions_rev.append(pred_s)
                            
                            cumsum_preds_rev = np.cumsum(predictions_rev)
                            bt_avg_predictions_rev = [cumsum_preds_rev[s] / (s + 1) for s in range(n_obs_rev)]
                            bt_predictions_history[i][j] = bt_avg_predictions_rev
                            
                            e_values[i, j] = compute_e_value_covariate(
                                observations_history[i][j],
                                bt_predictions_history[i][j],
                                successes[i, j],
                                int(trials[i, j])
                            )
                        
                        if e_values[i, j] >= K:
                            R_t[i, j] = True
                
                tournament_pair_index += 1
        
        else:
            raise ValueError("Unknown sampling method. Use 'random_pair' or 'tournament'.")
        
        prev_count = R_t.sum()
        while True:
            T_set = propagate_transitivity(R, R_t, m)
            R_t = R_t | T_set
            
            current_count = R_t.sum()
            if current_count == prev_count:
                break
            prev_count = current_count
        
        A = A & ~R_t
        R = R | R_t

        newly_rejected = np.argwhere(R_t)
        for j, i in newly_rejected:
            A[i, j] = False
        
        rank_confidence = None
        top_k_set = None
        if top_k is not None:
            topk_info = compute_topk_confidence_set(R, m, top_k)
            rank_confidence = topk_info['rank_confidence']
            top_k_set = topk_info['topk_models']
            last_rank_confidence = rank_confidence
            last_topk_set = top_k_set
        
        results.append({
            'time': t,
            'rejected_count': R.sum(),
            'active_count': A.sum(),
            'round_rejections': R_t.sum(),
            'sampling_method': sampling_method,
            'rejected_pairs': np.argwhere(R),
            'rank_confidence': rank_confidence,
            'top_k_set': top_k_set
        })
        
        if verbose and t % 100 == 0:
            tour_info = ""
            if sampling_method == "tournament":
                tour_info = f", Round: {tournament_round - 1}, Sampled: {tournament_pair_index - 1}/{len(current_tournament_pairs)}"
            print(f"Time {t}: Rejected {R.sum():.0f} (new: {R_t.sum():.0f}), Active {A.sum():.0f} [covariate{tour_info}]", flush=True)
    
    return {
        'final_rejected': R,
        'results': results,
        'e_values': e_values,
        'successes': successes,
        'trials': trials,
        'partial_order': np.argwhere(R),
        'sampling_method': sampling_method,
        'rank_confidence': last_rank_confidence if top_k is not None else None,
        'top_k_confidence_set': last_topk_set if top_k is not None else None,
        'top_k': top_k,
        'covariate_assisted': True
    }
