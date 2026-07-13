# Comparison: Original UniRouter vs Uni-R2

This document provides a detailed comparison between the **original UniRouter** (from the paper) and **Uni-R2** (our implementation combining UniRouter + R2-Router).

---

## Table of Contents
1. [Quick Summary](#quick-summary)
2. [Problem Setting](#problem-setting)
3. [Architecture Comparison](#architecture-comparison)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Feature Representation](#feature-representation)
6. [Routing Decision](#routing-decision)
7. [Training Process](#training-process)
8. [Adding New Models](#adding-new-models)
9. [Complexity Analysis](#complexity-analysis)
10. [Pros and Cons](#pros-and-cons)
11. [Use Cases](#use-cases)

---

## Quick Summary

| Aspect | Original UniRouter | Uni-R2 |
|--------|-------------------|----------|
| **Problem** | Route to best LLM (single model selection) | Route to best (LLM, token_budget) pair |
| **Token Budgets** | ❌ No (always use unlimited tokens) | ✅ Yes (multiple budgets: 50, 100, ..., 3200) |
| **New Models** | ✅ Yes (run on validation set) | ✅ Yes (run on validation set × budgets) |
| **LLM Features** | Prediction error vector on S_val | Prediction error matrix on S_val × budgets |
| **Routing** | Pick best LLM | Pick best (LLM, budget) |
| **Cost Model** | Fixed per-model cost | Dynamic: tokens_used × model_size |
| **Clustering** | Optional (unsupervised or supervised) | Optional (unsupervised) |
| **Training** | Learn cluster map (optional) | No training (direct features) |

---

## Problem Setting

### Original UniRouter

**Goal**: Route each query to the best LLM from a dynamic pool

```
Given:
  - Query x
  - Pool of LLMs H = {GPT-3.5, GPT-4, Claude-2, ...}
  - Each LLM has fixed cost c(h)

Select:
  h* = argmin [P(h wrong on x | x) + λ · c(h)]
       h ∈ H

Output: Single LLM name
```

**Key assumption**: Each LLM generates response with **unlimited tokens** (or default setting)

### Uni-R2

**Goal**: Route each query to the best (LLM, token_budget) pair

```
Given:
  - Query x
  - Pool of LLMs H = {GPT-3.5, GPT-4, Claude-2, ...}
  - Token budgets B = {50, 100, 200, 400, 800, 1600, 3200}
  - Each (LLM, budget) has cost: tokens_used × model_size

Select:
  (h*, b*) = argmin [P(h wrong on x with budget b | x) + λ · cost(h, b)]
              h ∈ H, b ∈ B

Output: (LLM name, token budget)
```

**Key difference**: Jointly optimizes **which model** AND **how many tokens**

---

## Architecture Comparison

### Original UniRouter (Cluster-based)

```
┌─────────────────────────────────────────────────────────┐
│ TRAINING PHASE (one-time)                               │
│                                                          │
│ 1. Cluster training prompts into K clusters (K-means)   │
│    - Input: Training embeddings                         │
│    - Output: K cluster centroids                        │
│                                                          │
│ 2. (Optional) Learn cluster map Φ(x; θ)                 │
│    - Train neural network to map query → cluster probs  │
│    - Minimizes log loss on training LLMs                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ NEW MODEL REGISTRATION (per new LLM)                    │
│                                                          │
│ For new LLM h_new:                                       │
│   1. Run h_new on validation set S_val                   │
│   2. Compute per-cluster error rates:                    │
│      Ψ(h_new) = [err_cluster1, ..., err_clusterK]      │
│      where err_clusterk = avg error on cluster k        │
│   3. Store Ψ(h_new) ∈ R^K                               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ ROUTING PHASE (per query)                               │
│                                                          │
│ For query x:                                             │
│   1. Assign to cluster: k = argmax Φ_k(x)               │
│   2. For each LLM h ∈ pool:                              │
│      risk(h) = Ψ_k(h) + λ · c(h)                        │
│   3. Select: h* = argmin risk(h)                        │
└─────────────────────────────────────────────────────────┘
```

### Uni-R2

```
┌─────────────────────────────────────────────────────────┐
│ VALIDATION SET CREATION (one-time)                      │
│                                                          │
│ 1. Sample S_val ⊂ training set (e.g., 500 prompts)     │
│ 2. (Optional) Cluster S_val into K clusters            │
│    - Input: Validation embeddings                       │
│    - Output: K cluster centroids + assignments          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ NEW MODEL REGISTRATION (per new LLM)                    │
│                                                          │
│ For new LLM h_new:                                       │
│   1. For each budget b ∈ {50, 100, 200, ..., 3200}:    │
│      - Run h_new on S_val with max_tokens=b            │
│      - Record quality scores                            │
│   2. Compute feature matrix:                            │
│      Ψ(h_new) ∈ R^{K×B}                                 │
│      where Ψ[k, b] = avg score on cluster k, budget b  │
│   3. Store Ψ(h_new)                                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ ROUTING PHASE (per query)                               │
│                                                          │
│ For query x:                                             │
│   1. Compute similarity to clusters:                     │
│      Φ(x) = [sim(x, cluster1), ..., sim(x, clusterK)]  │
│   2. For each (LLM h, budget b) pair:                   │
│      quality = Φ(x)^T · Ψ(h)[:, b]                      │
│      cost = per_token_price(h) × b                      │
│      score = quality - λ · cost                         │
│   3. Select: (h*, b*) = argmax score                    │
└─────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### Original UniRouter

**LLM Representation**:
```
Ψ(h) ∈ R^K

Ψ_k(h) = (1 / |C_k|) Σ_{(x,y)∈C_k} 1{y ≠ h(x)}

where C_k = validation samples in cluster k
```

**Query Representation**:
```
Φ(x) ∈ {0, 1}^K  (hard assignment)
or
Φ(x) ∈ [0, 1]^K  (soft assignment if learned)

Φ_k(x) = { 1 if x assigned to cluster k
         { 0 otherwise
```

**Routing Score**:
```
For each LLM h:
  score(h) = Φ(x)^T · Ψ(h) + λ · c(h)
           = Ψ_k(h) + λ · c(h)  (if hard assignment to cluster k)

Select: h* = argmin score(h)
```

### Uni-R2

**LLM Representation**:
```
Ψ(h) ∈ R^{K×B}

Ψ[k, b] = (1 / |C_k|) Σ_{(x,y)∈C_k} quality_score(h(x, max_tokens=b), y)

where:
  K = number of clusters
  B = number of token budgets
  C_k = validation samples in cluster k
```

**Query Representation**:
```
Φ(x) ∈ [0, 1]^K

Φ_k(x) = cosine_similarity(embedding(x), centroid_k)

Normalized so Φ(x) sums to ~1
```

**Routing Score**:
```
For each (LLM h, budget b):
  quality = Φ(x)^T · Ψ(h)[:, b]
          = Σ_k Φ_k(x) × Ψ[k, b]

  cost = per_token_price(h) × b

  score = quality - λ · cost

Select: (h*, b*) = argmax score
```

---

## Feature Representation

### Original UniRouter

**Per-cluster error vector**:

```python
# For LLM h, validation set S_val with K clusters

Ψ(h) = np.zeros(K)

for k in range(K):
    cluster_samples = S_val[cluster_k]
    errors = [1 if y != h(x) else 0 for (x, y) in cluster_samples]
    Ψ(h)[k] = np.mean(errors)  # Error rate on cluster k

# Result: [K] vector
# Example: [0.12, 0.08, 0.25, ..., 0.15]
#           math  code  reason     trivia
```

**Interpretation**: "How often does LLM h fail on each type of query?"

### Uni-R2

**Per-cluster, per-budget quality matrix**:

```python
# For LLM h, validation set S_val with K clusters, B budgets

Ψ(h) = np.zeros((K, B))

for k in range(K):
    cluster_samples = S_val[cluster_k]
    for b_idx, budget in enumerate(budgets):
        scores = []
        for (x, y) in cluster_samples:
            response = h(x, max_tokens=budget)
            score = evaluate(response, y)  # 0-1 or continuous
            scores.append(score)
        Ψ(h)[k, b_idx] = np.mean(scores)

# Result: [K, B] matrix
# Example for K=3, B=4:
#           50    100   200   400  ← budgets
# math   [[0.65, 0.72, 0.85, 0.91],  ← cluster 1
# code    [0.55, 0.68, 0.78, 0.84],  ← cluster 2
# reason  [0.45, 0.58, 0.70, 0.80]]  ← cluster 3
```

**Interpretation**: "How well does LLM h perform on each type of query with each budget?"

---

## Routing Decision

### Example Query: "Solve the differential equation dy/dx = 2x"

**Original UniRouter**:

```python
# Step 1: Assign to cluster
query_embedding = embed("Solve dy/dx = 2x")
cluster_similarities = cosine_sim(query_embedding, cluster_centroids)
assigned_cluster = argmax(cluster_similarities)  # e.g., cluster 3 (math)

# Step 2: Look up error rates for each LLM on math cluster
errors = {
    'GPT-3.5': Ψ(GPT-3.5)[3] = 0.15,  # 15% error on math
    'GPT-4': Ψ(GPT-4)[3] = 0.08,      # 8% error on math
    'Claude': Ψ(Claude)[3] = 0.12      # 12% error on math
}

# Step 3: Compute routing scores
scores = {
    'GPT-3.5': 0.15 + λ×1.0 = 0.15 + λ,
    'GPT-4': 0.08 + λ×10.0 = 0.08 + 10λ,
    'Claude': 0.12 + λ×5.0 = 0.12 + 5λ
}

# Step 4: Select best (minimize score)
if λ = 0.01:
    'GPT-3.5': 0.16, 'GPT-4': 0.18, 'Claude': 0.17
    → Choose GPT-3.5

Output: GPT-3.5 (always with unlimited tokens)
```

**Uni-R2**:

```python
# Step 1: Compute similarity to ALL clusters
query_embedding = embed("Solve dy/dx = 2x")
Φ(x) = cosine_sim(query_embedding, all_cluster_centroids)
# e.g., [0.05, 0.12, 0.85, 0.03, ...]  ← High similarity to math cluster

# Step 2: Compute quality for each (LLM, budget)
# For GPT-4 @ 200 tokens:
quality = Φ(x)^T · Ψ(GPT-4)[:, 200_idx]
        = 0.05×0.92 + 0.12×0.68 + 0.85×0.88 + ...
        = 0.82

cost = 1e-4 × 200 = 0.02
score = 0.82 - λ×0.02

# Repeat for all combinations:
scores = {
    ('GPT-3.5', 50): 0.65 - λ×0.0025,
    ('GPT-3.5', 200): 0.72 - λ×0.01,
    ('GPT-4', 50): 0.70 - λ×0.005,
    ('GPT-4', 200): 0.82 - λ×0.02,
    ('GPT-4', 400): 0.88 - λ×0.04,
    ...
}

# Step 3: Select best (maximize score)
if λ = 1e-5:
    ('GPT-4', 400): 0.88 - 0.0004 = 0.8796
    → Choose (GPT-4, 400 tokens)

Output: (GPT-4, 400 tokens)
```

**Key Difference**: UniRouter picks GPT-3.5 (cheaper, worse). Uni-R2 picks GPT-4 with 400 tokens (better quality-cost tradeoff).

---

## Training Process

### Original UniRouter

**Training Phase** (Optional):

```python
# Only if using learned cluster map (supervised variant)

# Input:
training_prompts = [(x_1, y_1), ..., (x_N, y_N)]
training_LLMs = [h_1, h_2, ..., h_M]

# Cluster training prompts
centroids, assignments = kmeans(training_embeddings, K)

# Train cluster assignment network
θ = train_network(
    inputs=training_embeddings,
    outputs=cluster_assignments,
    loss=cross_entropy
)

# Φ(x; θ) = softmax(θ^T · embedding(x))
```

**Time**: Hours (neural network training)
**Data**: Need labeled data from training LLMs

### Uni-R2

**No Training Phase**:

```python
# Just cluster validation set (unsupervised)

validation_prompts = sample(training_set, 500)
centroids, assignments = kmeans(validation_embeddings, K)

# Done! No gradient updates, no optimization
```

**Time**: Seconds (just K-means)
**Data**: Only validation set (no labels needed for clustering)

---

## Adding New Models

### Original UniRouter

**Cost to add new LLM**:

```
1. Run new LLM on S_val:
   - 500 prompts
   - Unlimited tokens per prompt
   - API calls: 500

2. Compute Ψ(h_new):
   - Per-cluster error rates
   - Computation: ~1 second

3. Total:
   - Time: Minutes (API latency)
   - Cost: $1-5 (API costs)
   - Training: ZERO
```

### Uni-R2

**Cost to add new LLM**:

```
1. Run new LLM on S_val × budgets:
   - 500 prompts
   - 8 budgets per prompt
   - API calls: 500 × 8 = 4,000

2. Compute Ψ(h_new):
   - Per-cluster, per-budget quality matrix
   - Computation: ~1 second

3. Total:
   - Time: Minutes to hours (API latency)
   - Cost: $5-20 (8× more API calls)
   - Training: ZERO
```

**Tradeoff**: Uni-R2 costs 8× more to add a model (due to multiple budgets), but enables finer-grained routing.

---

## Complexity Analysis

### Validation Set Inference Cost

| Method | Prompts | Budgets | Total API Calls |
|--------|---------|---------|-----------------|
| Original UniRouter | 500 | 1 (unlimited) | **500** |
| Uni-R2 | 500 | 8 | **4,000** |

### Routing Latency (per query)

**Original UniRouter**:
```
1. Assign to cluster: O(K × D)  (K=100, D=768)
2. Compute scores: O(M)  (M = number of LLMs)
3. Select best: O(M)

Total: O(K×D + M) ≈ 77,000 operations
```

**Uni-R2**:
```
1. Compute Φ(x): O(K × D)  (K=100, D=768)
2. Compute scores: O(M × B × K)  (M=5, B=8, K=100)
3. Select best: O(M × B)

Total: O(K×D + M×B×K) ≈ 81,000 operations
```

**Difference**: Negligible (<5% slower)

### Memory Footprint

**Original UniRouter**:
```
LLM features: M × K floats
Example: 5 models × 100 clusters = 500 floats ≈ 2 KB
```

**Uni-R2**:
```
LLM features: M × K × B floats
Example: 5 models × 100 clusters × 8 budgets = 4,000 floats ≈ 16 KB
```

**Difference**: 8× larger (but still tiny!)

---

## Pros and Cons

### Original UniRouter

**Pros**:
- ✅ Simple: Only need to pick best LLM
- ✅ Fast validation: 500 API calls per new model
- ✅ Cheap: $1-5 per new model
- ✅ Small memory: ~2 KB per model
- ✅ Proven: Published in peer-reviewed paper

**Cons**:
- ❌ No token budget optimization
- ❌ Always uses unlimited tokens (wasteful)
- ❌ Can't trade quality for cost within a model
- ❌ Binary choice: use model or don't
- ❌ Misses opportunity to use expensive models with small budgets

### Uni-R2

**Pros**:
- ✅ Token budget optimization: 8 budgets per model
- ✅ Fine-grained cost-quality tradeoff
- ✅ Can use expensive models with small budgets on simple queries
- ✅ Can use cheap models with large budgets on complex queries
- ✅ No training (simpler than learned cluster map)
- ✅ Combines two proven ideas (UniRouter + R2-Router)

**Cons**:
- ❌ Slower validation: 4,000 API calls per new model (8× more)
- ❌ More expensive: $5-20 per new model
- ❌ Larger memory: ~16 KB per model (8× more)
- ❌ More complex routing logic
- ❌ Not yet published/validated

---

## Use Cases

### When to Use Original UniRouter

**Scenario 1**: Simple routing with fixed token budgets
```
Problem: Route customer support queries to {GPT-3.5, GPT-4}
Budget: Always use default tokens (~500)
Priority: Minimize API costs for validation

→ Use UniRouter: Only 500 calls per model
```

**Scenario 2**: LLMs with different specializations
```
Problem: Route to {CodeLlama, MathGPT, GeneralGPT}
Budget: Unlimited tokens OK
Priority: Pick right specialist

→ Use UniRouter: Focuses on model selection
```

**Scenario 3**: Rapidly changing model pool
```
Problem: 10 new models per week
Budget: Unlimited tokens
Priority: Fast validation (<1 hour)

→ Use UniRouter: 500 calls = fast validation
```

### When to Use Uni-R2

**Scenario 1**: Cost-sensitive applications
```
Problem: Serving 1M queries/day
Budget: $10k/month budget
Priority: Minimize cost while maintaining quality

→ Use Uni-R2: Route simple queries to cheap+small budgets
```

**Scenario 2**: Diverse query complexity
```
Problem: Queries range from "2+2" to "prove Fermat's theorem"
Budget: Variable (willing to spend on hard queries)
Priority: Match budget to difficulty

→ Use Uni-R2: Small budgets for simple, large for complex
```

**Scenario 3**: Quality-cost pareto frontier
```
Problem: Need to trace full quality-cost curve
Budget: Sweep λ from 0 to 1e-3
Priority: Understand tradeoff options

→ Use Uni-R2: Multiple budgets enable fine-grained curve
```

---

## Empirical Comparison (Hypothetical)

### Setup
- 5 models: {GPT-3.5, GPT-4, Claude-2, Llama-70B, Gemini}
- Validation set: 500 prompts
- Test set: 10,000 queries
- λ swept from 0 to 1e-3

### Expected Results

**Original UniRouter**:
```
Peak accuracy: 0.85 (always GPT-4)
Min cost: 1.2 (always GPT-3.5)
AUDC: 0.72
Pareto points: 5 (one per model)
```

**Uni-R2**:
```
Peak accuracy: 0.85 (GPT-4 @ unlimited)
Min cost: 0.3 (GPT-3.5 @ 50 tokens)
AUDC: 0.81 (+12%)
Pareto points: 40 (5 models × 8 budgets)
```

**Explanation**: More routing options → better cost-quality tradeoff

---

## Conclusion

| Criterion | Winner | Reasoning |
|-----------|--------|-----------|
| **Simplicity** | UniRouter | Only pick model, not budget |
| **Validation Cost** | UniRouter | 8× fewer API calls |
| **Routing Quality** | Uni-R2 | More options → better tradeoff |
| **Cost Efficiency** | Uni-R2 | Can use cheap models with small budgets |
| **Flexibility** | Uni-R2 | Supports variable token budgets |
| **Memory** | UniRouter | 8× smaller features |
| **Maturity** | UniRouter | Published and validated |

**Recommendation**:
- Use **UniRouter** if token budgets are fixed and you need fast validation
- Use **Uni-R2** if you want to optimize cost-quality tradeoff with variable budgets

**Best of both worlds**: Start with UniRouter for rapid prototyping, then migrate to Uni-R2 once you've validated the model pool and want finer control over costs.
