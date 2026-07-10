import numpy as np
import os

ids_to_remove = [1, 3, 5, 7, 9]
# Constants
options = ["A", "B", "C", "D", "E", "F"]

# Utilities
def softmax(x):
    x = np.asarray(x, dtype=float)
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-12)

# Unweighted CP
def LAC_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    pred_sets_all = {}

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            if key not in logits_data_all:
                continue

            cal_scores = []
            cal_logits = logits_data_all[key]["cal"]

            for i, row in enumerate(cal_logits):
                probs = softmax(row["logits_options"])
                truth = cal_raw_data[i]["answer"]
                cal_scores.append(1 - probs[options.index(truth)])

            if len(cal_scores) == 0:
                pred_sets_all[key] = {}
                continue

            n = len(cal_scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method="higher")

            pred_sets = {}
            for row in logits_data_all[key]["test"]:
                probs = softmax(row["logits_options"])
                ps = [options[i] for i, p in enumerate(probs) if p >= 1 - qhat]
                if not ps:
                    ps = [options[int(np.argmax(probs))]]
                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets

    return pred_sets_all


def APS_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    pred_sets_all = {}

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            if key not in logits_data_all:
                continue

            cal_scores = []
            cal_logits = logits_data_all[key]["cal"]

            for i, row in enumerate(cal_logits):
                probs = softmax(row["logits_options"])
                truth = cal_raw_data[i]["answer"]

                order = np.argsort(probs)[::-1]
                csum = probs[order].cumsum()
                recovered = np.empty_like(csum)
                recovered[order] = csum

                cal_scores.append(recovered[options.index(truth)])

            if len(cal_scores) == 0:
                pred_sets_all[key] = {}
                continue

            n = len(cal_scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method="higher")

            pred_sets = {}
            for row in logits_data_all[key]["test"]:
                probs = softmax(row["logits_options"])
                order = np.argsort(probs)[::-1]
                csum = probs[order].cumsum()

                ps = []
                for i in range(len(csum)):
                    if csum[i] <= qhat:
                        ps.append(options[order[i]])
                if not ps:
                    ps = [options[order[0]]]

                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets

    return pred_sets_all

# Weighting helpers
def density_ratio(emb, clf, clip_min=1e-6, clip_max=1 - 1e-6):
    p = clf.predict_proba(emb)[:, 1]
    p = np.clip(p, clip_min, clip_max)
    return p / (1 - p + 1e-12)


def weighted_quantile(scores, weights, alpha, gamma=1.0, w_dir=None):
    """
    Weighted quantile with gamma inflation:
        cutoff = (1 - alpha) * (sum(w) + gamma * max(w))
    """
    scores = np.asarray(scores)
    weights = np.asarray(weights)

    if len(scores) == 0:
        return 1.0

    order = np.argsort(scores)
    s = scores[order]
    w = weights[order]

    csum = np.cumsum(w)
    cutoff = (1 - alpha) * (w.sum() + gamma * w.max())

    loc = np.searchsorted(csum, cutoff, side="left")
    qhat = 1.0 if loc >= len(s) else s[loc]

    if w_dir is not None:
        os.makedirs(w_dir, exist_ok=True)
        np.savez(
            os.path.join(w_dir, "w_stats.npz"),
            w_min=w.min(),
            w_median=np.median(w),
            w_max=w.max(),
            w_sum=w.sum(),
            alpha=alpha,
            gamma=gamma,
            cutoff=cutoff,
            qhat=qhat,
        )

    return qhat

# Weighted CP
def LAC_CP_W(
    logits_data_all,
    cal_raw_data,
    prompt_methods,
    icl_methods,
    clf,
    embed_model,
    alpha=0.1,
    gamma=1.0,
    w_dir=None,
):
    pred_sets_all = {}

    if len(cal_raw_data) == 0:
        return pred_sets_all
    cal_question = [x["question"] for x in cal_raw_data]
    cal_emb = embed_model.encode(cal_question)
    cal_weights = density_ratio(cal_emb, clf)

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            if key not in logits_data_all:
                continue

            cal_scores = []
            cal_logits = logits_data_all[key]["cal"]

            for i, row in enumerate(cal_logits):
                probs = softmax(row["logits_options"])
                truth = cal_raw_data[i]["answer"]
                cal_scores.append(1 - probs[options.index(truth)])

            if len(cal_scores) == 0:
                pred_sets_all[key] = {}
                continue

            qhat = weighted_quantile(
                cal_scores,
                cal_weights[: len(cal_scores)],
                alpha,
                gamma=gamma,
                w_dir=w_dir,
            )

            pred_sets = {}
            for row in logits_data_all[key]["test"]:
                probs = softmax(row["logits_options"])
                ps = [options[i] for i, p in enumerate(probs) if p >= 1 - qhat]
                if not ps:
                    ps = [options[int(np.argmax(probs))]]
                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets

    return pred_sets_all


def APS_CP_W(
    logits_data_all,
    cal_raw_data,
    prompt_methods,
    icl_methods,
    clf,
    embed_model,
    alpha=0.1,
    gamma=1.0,
    w_dir=None,
):
    pred_sets_all = {}

    if len(cal_raw_data) == 0:
        return pred_sets_all
    cal_question = [x["question"] for x in cal_raw_data]
    cal_emb = embed_model.encode(cal_question)
    cal_weights = density_ratio(cal_emb, clf)

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            if key not in logits_data_all:
                continue

            cal_scores, used_w = [], []
            cal_logits = logits_data_all[key]["cal"]

            for i, row in enumerate(cal_logits):
                probs = softmax(row["logits_options"])
                truth = cal_raw_data[i]["answer"]

                order = np.argsort(probs)[::-1]
                csum = probs[order].cumsum()
                recovered = np.empty_like(csum)
                recovered[order] = csum

                cal_scores.append(recovered[options.index(truth)])
                used_w.append(cal_weights[i])

            if len(cal_scores) == 0:
                pred_sets_all[key] = {}
                continue

            qhat = weighted_quantile(
                cal_scores,
                np.asarray(used_w),
                alpha,
                gamma=gamma,
                w_dir=w_dir,
            )

            pred_sets = {}
            for row in logits_data_all[key]["test"]:
                probs = softmax(row["logits_options"])
                order = np.argsort(probs)[::-1]
                csum = probs[order].cumsum()

                ps = []
                for i in range(len(csum)):
                    if csum[i] <= qhat:
                        ps.append(options[order[i]])
                if not ps:
                    ps = [options[order[0]]]

                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets

    return pred_sets_all
