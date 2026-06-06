import json, numpy as np, random
import argparse, copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from transformers import Trainer
from sklearn.neighbors import NearestNeighbors
import cvxpy as cp

MAX_LENGTH = 512


routerbench_cost = np.array([7.27031551e-05, 2.31895898e-04, 2.13621678e-03, 2.41047998e-03,
 2.42202363e-04, 3.28095445e-03, 1.71426650e-04, 2.01902332e-04, 4.55752123e-05, 1.34171861e-04, 1.84926716e-04])

routerbench_perf = np.array([0.43214326, 0.59844322, 0.63064309, 0.63556818, 0.6165698,  0.78112239, 0.20256822,
                              0.32848247, 0.30845001, 0.54979054, 0.64783749])

sprout_cost = np.array([7.65275499e-03, 5.63930860e-04, 4.92437851e-03, 3.39966149e-04,
    8.53533645e-05, 1.49579500e-04, 7.16633449e-04, 2.43288146e-04, 6.66874424e-05, 6.46840306e-05, 
    5.51861067e-04, 2.01393780e-03,3.74017428e-04])

sprout_perf = np.array([0.82663094, 0.5790882,  0.8458901,  0.80837548, 0.55251066, 0.65865551,
 0.81000868, 0.69046873, 0.46041816, 0.62884477, 0.80390988, 0.77590859, 0.61627788])

leaderboard_cost = np.array([6.56528875e-04, 4.77927953e-04, 8.89902323e-04, 6.67426742e-04,
 8.89902323e-04, 2.22475581e-04, 9.84793312e-04, 7.04621428e-04,
 6.13063606e-04, 2.29898852e-04, 7.66329507e-05, 2.47031414e-04,
 6.43716846e-04, 1.43048188e-04, 1.64160434e-04, 1.64160434e-04,
 4.92396656e-04, 7.38594984e-04])

leaderboard_perf = np.array([0.42801627, 0.40117488, 0.55237235, 0.56222323, 0.5608676,  0.42024401,
 0.49055581, 0.41283326, 0.46154541, 0.41943064, 0.19069137, 0.22747402,
 0.54794397, 0.25838229, 0.31098057, 0.33556258, 0.37903299, 0.50646182])


def budget_distribution(num_models, indice, dataset):
    B_uniform = np.array([10] * num_models)

    model_weights = []
    ## based on quality/cost
    if dataset == 1:
        weights = np.sqrt(routerbench_perf / routerbench_cost)
        if num_models < 11:
            weights = weights[indice]
    if dataset == 0:
        weights = np.sqrt(sprout_perf / sprout_cost)
        if num_models < 13:
            weights = weights[indice]
    if dataset == 2:
        weights = np.sqrt(leaderboard_perf / leaderboard_cost)
        if num_models < 18:
            weights = weights[indice]
    
    model_weights = weights

    # based on cost
    cost = []
    if dataset == 1:
        inverse_costs = np.sqrt(1 / routerbench_cost)
        if num_models < 11:
            inverse_costs = inverse_costs[indice]
    if dataset == 0:
        inverse_costs = np.sqrt(1 / sprout_cost)
        if num_models < 13:
            inverse_costs = inverse_costs[indice]
    if dataset == 2:
        inverse_costs = np.sqrt(1 / leaderboard_cost)
        if num_models < 18:
            inverse_costs = inverse_costs[indice]
    cost = inverse_costs

    # based on perf
    perf = []
    if dataset == 1:
        quality = routerbench_perf
        if num_models < 11:
            quality = quality[indice]
    if dataset == 0:
        quality = sprout_perf
        if num_models < 13:
            quality = quality[indice]
    if dataset == 2:
        quality = leaderboard_perf
        if num_models < 18:
            quality = quality[indice]
    perf = quality

    # worst cases
    extreme_cases = []
    ## low quality/cost models gain 80% budget 
    for k in range(1, 6):
        min_k_indices = np.argsort(model_weights)[:k]
        few_high_rest_low = np.array([50 if i in min_k_indices else (25 / 2 * k /(num_models - k)) for i in range(num_models)])
        extreme_cases.append(few_high_rest_low)

    # randomized allocation (100 cases)
    random_allocations = []
    for i in range(100):
        allocation = np.random.uniform(low=5, high=30, size=num_models)
        random_allocations.append(allocation)

    return model_weights, cost, perf, extreme_cases, random_allocations, B_uniform

def random_router(dataset, M, N, B, models):
    tg = np.zeros((M, N))
    obj = 0.0
    cost = 0.0
    lost = 0
    random.seed(42)
    for j, item in enumerate(dataset):
        i = np.random.randint(0, M)
        tg[i, j] = item[f"{models[i]}|total_cost"]
        if tg[i, :].sum() <= B[i]:
            obj += item[models[i]]
            cost += tg[i, j]
        else:
            tg[i, j] = 0
            lost +=1 

    return obj, cost, lost

def greedy(dataset, embeds, ann, M, N, B, base_data, op, models):
    hatg = np.zeros((M, N))
    tg = np.zeros((M, N))
    obj = 0.0
    cost = 0.0
    lost = 0
    for j, item in enumerate(dataset):
        index = item["index"]
        embed = embeds[index]
        indices, _ = ann.search(embed)
        indices = [int(i) for i in indices[0]]
        data = base_data.select(indices)
        d = []
        g = []
        # vanilla mean
        for model in models:
            d.append(np.mean(data[model]))
            g.append(np.mean(data[f"{model}|total_cost"]))
        
        if op == 1:
            r = np.argmax(np.array(d))
        elif op == 2:
            res = np.zeros(M)
            for i in range(0, M):
                res[i] = B[i] - hatg[i, :].sum()
            r = np.argmax(res)


        hatg[r, j] = g[r]
        tg[r, j] = item[f"{models[r]}|total_cost"]
        if tg[r, :].sum() <= B[r]:
            obj += item[models[r]]
            cost += tg[r, j]
        else:
            tg[r, j] = 0
            lost += 1 

    return obj, cost, lost

def true_optimal(dataset, M, N, B, models, show=False):
    td = np.zeros((M, N))
    tg = np.zeros((M, N))
    for j, item in enumerate(dataset):
        d = []
        g = []
        for model in models:
            d.append(item[model])
            g.append(item[f"{model}|total_cost"])
        for i in range(0, M):
            td[i, j] = d[i]
            tg[i, j] = g[i]

    # relaxation
    x, obj = solve_relaxed_assignment(np.array(td), np.array(tg), B, show=show)

    obj1 = 0.0
    cost = 0.0
    lost = 0
    tg = np.zeros((M, N))
    for j, item in enumerate(dataset):
        r = np.argmax(x[:, j])
        if x[r, j] > 0:
            tg[r, j] = item[f"{models[r]}|total_cost"]
            if tg[r, :].sum() <= B[r]:
                obj1 += item[models[r]]
                cost += tg[r, j]
            else:
                tg[r, j] = 0
                lost += 1 
        else: 
            lost += 1

    return obj, obj1, cost, lost

def est_optimal(dataset, embeds, ann, base, M, N, B, models, show=False):
    import gc
    hatd = np.zeros((M, N))
    hatg = np.zeros((M, N))
    for j, item in enumerate(dataset):
        index = item["index"]
        embed = embeds[index]
        indices, _ = ann.search(embed)
        indices = [int(i) for i in indices[0]]
        data = base.select(indices)
        d = []
        g = []
        # vanilla mean
        for model in models:
            d.append(np.mean(data[model]))
            g.append(np.mean(data[f"{model}|total_cost"]))

        for i in range(0, M):
            hatd[i, j] = d[i]
            hatg[i, j] = g[i]

    del data, indices, embed, d, g
    gc.collect()
    x, obj = solve_relaxed_assignment(np.array(hatd), np.array(hatg), B, show=show)

    obj1 = 0.0
    cost = 0.0
    lost = 0
    tg = np.zeros((M, N))
    for j, item in enumerate(dataset):
        r = np.argmax(x[:, j])
        if x[r, j] > 0:
            tg[r, j] = item[f"{models[r]}|total_cost"]
            if tg[r, :].sum() <= B[r]:
                obj1 += item[models[r]]
                cost += tg[r, j]
            else:
                tg[r, j] = 0
                lost += 1 
        else: 
            lost += 1


    return obj, obj1, cost, lost


def solve_binary_assignment(d_star, g, B, show):
    M, N = d_star.shape  # M models, N queries
    x = cp.Variable((M, N), name="x",  boolean=True)
    total_reward = cp.sum(cp.multiply(d_star, x))
    objective = cp.Maximize(total_reward)
    constraints = [
        cp.sum(cp.multiply(g, x), axis=1) <= B,
        cp.sum(x, axis=0) <= 1,
        x <= 1
    ]
    problem = cp.Problem(objective, constraints)
    obj_val = problem.solve(solver=cp.HIGHS, verbose=show)

    return x.value, obj_val

def solve_relaxed_assignment(d_star, g, B, show):
    M, N = d_star.shape  

    x = cp.Variable((M, N), name="x", nonneg=True)
    total_reward = cp.sum(cp.multiply(d_star, x))
    objective = cp.Maximize(total_reward)
    constraints = [
        cp.sum(cp.multiply(g, x), axis=1) <= B,
        cp.sum(x, axis=0) <= 1,
        x <= 1
    ]
    problem = cp.Problem(objective, constraints)
    obj_val = problem.solve(solver=cp.HIGHS, verbose=show)

    return x.value, obj_val

def knn(dataset, embeds, M, N, B, base_data, base_embeds, op, models, top_k):
    base_embeds = base_embeds / np.linalg.norm(base_embeds, axis=1, keepdims=True)
    knn = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    knn.fit(base_embeds)
    
    tg = np.zeros((M, N))
    hatg = np.zeros((M, N))
    obj = 0.0
    cost = 0.0
    lost = 0
    for j, item in enumerate(dataset):
        index = item["index"]
        embed = embeds[index]
        embed = embed / np.linalg.norm(embed)
        embed = embed.reshape(1, -1)
        _, indices = knn.kneighbors(embed)
        data = base_data.select(indices[0])
        d = []
        g = []
        # vanilla mean
        for model in models:
            d.append(np.mean(data[model]))
            g.append(np.mean(data[f"{model}|total_cost"]))

        
        if op == 1:
            r = np.argmax(np.array(d))
        elif op == 2:
            res = np.zeros(M)
            for i in range(0, M):
                res[i] = B[i] - hatg[i, :].sum()
            r = np.argmax(res)

        hatg[r, j] = g[r]
        tg[r, j] = item[f"{models[r]}|total_cost"]
        if tg[r, :].sum() <= B[r]:
            obj += item[models[r]]
            cost += tg[r, j]
        else:
            lost += 1
            tg[r, j] = 0

    return obj, cost, lost

def segment_local(dataset, embeds, ann, M, N, B, models, base_data, size=8):
    obj = 0.0
    cost = 0.0
    lost = 0
    rB = copy.deepcopy(B)
    tB = copy.deepcopy(B)
    for i in range(0, len(dataset), size):
        batch = dataset[i: i+ size]
        rsize = len(batch["index"])
        hatd = np.zeros((M, rsize))
        hatg = np.zeros((M, rsize))
        for j in range(rsize):
            index = batch["index"][j]
            embed = embeds[index]
            indices, _ = ann.search(embed)
            indices = [int(i) for i in indices[0]]
            data = base_data.select(indices)
            d = []
            g = []
            # vanilla mean
            for model in models:
                d.append(np.mean(data[model]))
                g.append(np.mean(data[f"{model}|total_cost"]))
                
            for i in range(0, M):
                hatd[i, j] = d[i]
                hatg[i, j] = g[i]


        x, _ = solve_binary_assignment(np.array(hatd), np.array(hatg), rB, show=False)

        for j in range(rsize):
            f = 1
            for k in range(M):
                if x[k, j] == 1:
                    rB[k] = rB[k] - hatg[k, j]

                    # eval
                    c = batch[models[k]][j]
                    r = batch[f"{models[k]}|total_cost"][j]
                    if tB[k] >= r:
                        tB[k] = tB[k] - r
                        obj += c
                        cost += r
                    else:
                        lost += 1
                    #### 
                    f = 0
                    break 

            if f == 1:
                lost += 1

    return obj, cost, lost

def prepare_datasets_prediction(test_texts,
                                tokenizer,
                                max_length=MAX_LENGTH):
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)
    test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"]})
    return test_dataset

def sigmoid(z):
    return 1/(1+np.exp(-z))

def roberta(dataset, M, N, B, models, datset, op):
    input_costs = [3.0, 0.5, 2.5, 0.15, 0.1, 0.2, 0.9, 0.2, 0.06, 0.06, 0.9, 3.5,0.6]
    output_costs = [15.0, 1.5, 10.0, 0.6, 0.1, 0.2, 0.9, 0.2, 0.06, 0.06, 0.9, 3.5, 0.6] 
    input_costs = [x / 1000000 for x in input_costs]
    output_costs = [x / 1000000 for x in output_costs]
    if datset == 0:
        model_cost = AutoModelForSequenceClassification.from_pretrained("roberta_sprout/cost", problem_type="regression", num_labels=M)
        tokenizer_cost = AutoTokenizer.from_pretrained("roberta-base")
        model_perf = AutoModelForSequenceClassification.from_pretrained("roberta_sprout/perf", problem_type="multi_label_classification", num_labels=M)
        tokenizer_perf = AutoTokenizer.from_pretrained("roberta-base")

    elif datset == 1:
        model_cost = AutoModelForSequenceClassification.from_pretrained("roberta_routerbench/cost/", problem_type="regression", num_labels=M)
        tokenizer_cost = AutoTokenizer.from_pretrained("roberta-base")
        model_perf = AutoModelForSequenceClassification.from_pretrained("roberta_routerbench/perf", problem_type="multi_label_classification", num_labels=M)
        tokenizer_perf = AutoTokenizer.from_pretrained("roberta-base")

    elif datset == 2:
        model_cost = AutoModelForSequenceClassification.from_pretrained("roberta_leaderboard/cost/", problem_type="regression", num_labels=M)
        tokenizer_cost = AutoTokenizer.from_pretrained("roberta-base")
        model_perf = AutoModelForSequenceClassification.from_pretrained("roberta_leaderboard/perf", problem_type="multi_label_classification", num_labels=M)
        tokenizer_perf = AutoTokenizer.from_pretrained("roberta-base")

    trainer_pref = Trainer(model=model_perf, tokenizer=tokenizer_perf)
    trainer_cost = Trainer(model=model_cost, tokenizer=tokenizer_cost)
    tg = np.zeros((M, N))
    hatg = np.zeros((M, N))
    obj = 0.0
    cost = 0.0
    lost = 0
    size = 1024
    for i in range(0, len(dataset), size):
        batch = dataset[i: i + size]
        rsize = len(batch["index"])
        p = sigmoid(trainer_pref.predict(prepare_datasets_prediction(batch['prompt'], tokenizer_perf)).predictions)
        c = trainer_cost.predict(prepare_datasets_prediction(batch['prompt'], tokenizer_cost)).predictions

        for j in range(rsize):
            d = []
            g = []
            d = p[j]
            g = c[j]

            if op == 1:
                r = np.argmax(np.array(d))
            elif op == 2:
                res = np.zeros(M)
                for k in range(0, M):
                    res[k] = B[k] - hatg[k, :].sum()
                r = np.argmax(res)

            hatg[r, i+j] = g[r]
            tg[r, i+j] = batch[f"{models[r]}|total_cost"][j]
            if tg[r, :].sum() <= B[r]:
                obj += batch[models[r]][j]
                cost += tg[r, i+j]
            else:
                lost += 1
                tg[r, i+j] = 0

    return obj, cost, lost

