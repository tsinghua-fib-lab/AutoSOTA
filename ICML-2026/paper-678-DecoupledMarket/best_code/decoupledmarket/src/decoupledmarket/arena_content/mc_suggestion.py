import numpy as np
import pandas as pd

# ---------------------------

# ---------------------------
def random_policy(price, position):
    return np.random.choice([0, 1, -1])

# ---------------------------

# ---------------------------
def prepare_features(df: pd.DataFrame):

    feats = df[["last_price", "highest_price", "lowest_price", "begin_price"]].astype(float).values
    prices = df["last_price"].astype(float).values
    return feats, prices

# ---------------------------

# ---------------------------
def find_similar_windows_multifeat(features: np.ndarray,
                                   current_window: np.ndarray,
                                   top_k: int = 5,
                                   max_start: int = None):
    W = len(current_window)
    if max_start is None:
        max_start = len(features) - W
    max_start = max_start if max_start is not None else (len(features) - W)
    max_start = max(-1, min(max_start, len(features) - W))

    candidates = []

    for i in range(max_start + 1):
        past_window = features[i:i+W]

        pw_std = past_window.std(axis=0) + 1e-8
        cw_std = current_window.std(axis=0) + 1e-8
        pw_norm = (past_window - past_window.mean(axis=0)) / pw_std
        cw_norm = (current_window - current_window.mean(axis=0)) / cw_std
        dist = np.mean((pw_norm - cw_norm) ** 2)  # MSE
        candidates.append((dist, i))
    candidates.sort(key=lambda x: x[0])
    return [idx for _, idx in candidates[:top_k]]

# ---------------------------

# ---------------------------
def sample_future_paths(prices: np.ndarray,
                        indices: list,
                        window_size: int,
                        horizon: int,
                        current_last_price: float,
                        use_returns: bool = True):
    paths = []
    for idx in indices:
        future = prices[idx + window_size : idx + window_size + horizon]
        if len(future) != horizon:
            continue
        base = prices[idx + window_size - 1]
        path = [current_last_price]
        last = current_last_price
        for p in future:
            if use_returns:

                ret = (p - base) / base
                new_price = last * (1.0 + ret)
            else:

                delta = p - base
                new_price = last + delta
            path.append(float(new_price))
            last = new_price
        paths.append(path)
    return paths

# ---------------------------


# rollout_eval(features, prices, current_window, action, policy_fn,
#              n_rollouts=20, horizon=10, top_k=5, gamma=0.99, use_returns=True)
# ---------------------------
def rollout_eval(features: np.ndarray,
                 prices: np.ndarray,
                 current_window: np.ndarray,
                 action: int,
                 policy_fn,
                 n_rollouts: int = 20,
                 horizon: int = 10,
                 top_k: int = 5,
                 gamma: float = 0.99,
                 use_returns: bool = True):
    W = len(current_window)


    max_start = len(features) - W - horizon
    if max_start < 0:
        return {"mean": 0.0, "std": 0.0, "VaR_5": 0.0}


    indices = find_similar_windows_multifeat(features, current_window, top_k=top_k, max_start=max_start)
    if not indices:
        return {"mean": 0.0, "std": 0.0, "VaR_5": 0.0}


    current_last_price = prices[-1]
    candidate_paths = sample_future_paths(prices, indices, W, horizon, current_last_price, use_returns=use_returns)
    if not candidate_paths:
        return {"mean": 0.0, "std": 0.0, "VaR_5": 0.0}


    to_pos = {0: 0, 1: 1, -1: -1}

    # Rollout
    all_returns = []
    for path in candidate_paths:
        for _ in range(n_rollouts):
            position = to_pos.get(action, 0)
            G = 0.0
            discount = 1.0
            last_price = path[0]

            for price in path[1:]:
                if use_returns:
                    reward = position * (price / last_price - 1.0)
                else:
                    reward = position * (price - last_price)

                G += discount * reward
                discount *= gamma

                next_action = policy_fn(price, position)
                position = to_pos.get(next_action, position)
                last_price = price

            all_returns.append(G)

    return {
        "mean": float(np.mean(all_returns)),
        "std": float(np.std(all_returns)),
        "VaR_5": float(np.percentile(all_returns, 5))
    }

def daily_decision_narrative(df: pd.DataFrame,
                             stock_id: str,
                             window_size: int = 20,
                             horizon: int = 10,
                             top_k: int = 5,
                             n_rollouts: int = 20,
                             gamma: float = 0.99,
                             use_returns: bool = True,
                             policy_fn=random_policy):
    """Docstring."""
    sdf = df

    if len(sdf) < window_size + 1:
        return f"Not enough data for stock {stock_id}. At least {window_size+1} rows are required."

    features, prices = prepare_features(sdf)
    current_window = features[-window_size:]

    # Note: Sell is marked as -1
    actions = {0: "Hold", 1: "Buy", -1: "Sell"}
    results = {}

    for a in actions.keys():
        res = rollout_eval(features, prices, current_window, a, policy_fn,
                           n_rollouts=n_rollouts, horizon=horizon, top_k=top_k,
                           gamma=gamma, use_returns=use_returns)
        results[a] = res

    # choose the best action by mean return
    best_action, best_stats = max(results.items(), key=lambda x: x[1]['mean'])

    # Build narrative
    lines = [f"Analysis for stock {stock_id} based on the last {window_size} days:"]
    for a, name in actions.items():
        r = results[a]
    lines.append(
        f"Suggestion: {actions[best_action]} ({best_action}), "
        f"with an estimated mean return of {best_stats['mean']:.4f}."
    )
    return "\n".join(lines)
