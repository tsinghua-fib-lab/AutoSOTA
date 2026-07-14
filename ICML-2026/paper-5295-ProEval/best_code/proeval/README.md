# ProEval — Active Evaluation with Bayesian Quadrature

A Python library for efficient LLM evaluation through Bayesian Quadrature active sampling, topic-aware test case generation, structured LLM prediction, and neural encoder training.

## Installation

```bash
# From the project root
pip install numpy pandas scikit-learn torch requests tqdm bertopic hdbscan
```

## Quick Start

```python
from proeval import BQPriorSampler, BQEncoderSampler, TopicAwareGenerator, LLMPredictor
from proeval.encoder import EncoderTrainer
```

---

## 1. BQPriorSampler — Active Sampling

Efficiently estimate an LLM's accuracy using far fewer labeled samples than random sampling. Uses Bayesian Quadrature with a learned prior from other models' predictions.

### Initialisation

```python
from proeval.sampler import BQPriorSampler

sampler = BQPriorSampler(
    noise_variance=0.3,       # GP observation noise variance (default: 0.3)
    n_init=0,                 # Number of random initial samples (default: 0)
)
```

| Parameter        | Type    | Default | Description                                                                           |
| ---------------- | ------- | ------- | ------------------------------------------------------------------------------------- |
| `noise_variance` | `float` | `0.3`   | GP observation noise. Lower = more trust in observations, higher = smoother estimates |
| `n_init`         | `int`   | `0`     | Random initial samples before active acquisition starts. Useful for cold-start        |

### Sampling

```python
result = sampler.sample(
    predictions="svamp",                  # Dataset name or pandas DataFrame
    target_model="gemini25_flash",        # Model name to evaluate
    budget=50,                             # Number of samples to acquire
    data_dir=None,                         # Path to data directory (default: data/)
    pretrain_indices=None,                 # Specific model indices for pre-training prior
    seed=42,                               # Random seed for reproducibility
)
```

| Parameter          | Type                 | Default              | Description                                                                                                                             |
| ------------------ | -------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `predictions`      | `str` or `DataFrame` | _required_           | Dataset name (e.g. `"svamp"`, `"gsm8k"`, `"strategyqa"`) loaded from `data_dir`, or a pre-loaded DataFrame with `label_<model>` columns |
| `target_model`    | `int` or `str`       | `"gemini25_flash"`   | Name (preferred) or index of the model to target for testing                                                                           |
| `budget`           | `int`                | `50`                 | Number of samples to actively acquire                                                                                                   |
| `data_dir`         | `str`                | `None`               | Directory with `<dataset>_predictions.csv` files. Defaults to `data/`                                                               |
| `pretrain_indices` | `list[int]`          | `None`               | Explicit list of model indices to use as prior features. If `None`, uses all models except the target model                              |
| `seed`             | `int`                | `None`               | Random seed for reproducibility                                                                                                         |

### Pretrain Model Selection (GMM)

By default, the sampler uses **all** available models (except the target) as pretrain features. For better accuracy, you can use **GMM clustering** to automatically select only the models most similar to the target model:

```python
from proeval.sampler.pretrain_selector import select_pretrain_models_gmm

# Auto-select source models via GMM clustering on reference benchmarks
selected_models, selected_indices = select_pretrain_models_gmm(
    target_benchmark="svamp",
    target_model="gemini25_flash",
    # reference_benchmarks=None,  # Auto-discovers from data_dir
    # n_clusters=None,            # Auto-selects via BIC
)

# Pass selected indices to the sampler
result = sampler.sample(
    predictions="svamp",
    target_model="gemini25_flash",
    pretrain_indices=selected_indices,
    budget=50,
)
```

The GMM selector clusters models by their performance patterns across reference benchmarks and returns models in the same cluster as the target model. This is especially useful when you have many diverse models and want to avoid noisy pretrain signals.

### SamplingResult

```python
# SamplingResult attributes
result.estimates         # np.ndarray (budget,) — posterior mean estimate at each step
result.selected_indices  # list[int] — indices selected in acquisition order
result.posterior_mean    # np.ndarray (n_samples,) — final posterior mean
result.posterior_var     # np.ndarray (n_samples,) — final posterior variance
result.prior_mean        # np.ndarray (n_samples,) — prior mean from pretrain models

# Convenience methods
result.mae(true_mean)        # float — final |estimate - true_mean|
result.mae_curve(true_mean)  # np.ndarray (budget,) — MAE at every step
```

### Full Example: BQ vs Random Sampling

```python
import numpy as np
from proeval.sampler import BQPriorSampler, load_predictions, extract_model_predictions

# Load data to get true mean
df = load_predictions("svamp")
pred_matrix, model_names = extract_model_predictions(df)
true_mean = np.mean(pred_matrix[:, 0])

# BQ active sampling
sampler = BQPriorSampler(noise_variance=0.3)
result = sampler.sample(predictions="svamp", target_model="gemini25_flash", budget=50, seed=42)

print(f"True accuracy: {true_mean:.4f}")
print(f"BQ estimate:   {result.estimates[-1]:.4f}")
print(f"BQ MAE:        {result.mae(true_mean):.4f}")

# Compare with random baseline
random_mae = np.mean([
    abs(np.mean(pred_matrix[:, 0][np.random.choice(len(pred_matrix), 50, replace=False)]) - true_mean)
    for _ in range(10)
])
print(f"Random MAE:    {random_mae:.4f}")
print(f"Improvement:   {(1 - result.mae(true_mean) / random_mae) * 100:.1f}%")
```

---

## 1b. BQEncoderSampler — Active Sampling with Neural Encoder (BQ-TPF)

Uses a pre-trained neural encoder's φ embeddings and kernel for GP posterior updates instead of the linear prior.

> [!CAUTION]
> The encoder must be trained **without** the target dataset. For example, if the encoder was trained on `gsm8k + strategyqa` with `svamp` as holdout, you must sample on `svamp` — sampling on `gsm8k` would be data leakage.

### Initialisation

```python
from proeval.sampler import BQEncoderSampler

# Encoder trained on gsm8k+strategyqa, holdout=svamp
sampler = BQEncoderSampler(
    encoder_path="path/to/encoder_holdout_svamp.pth",
    noise_variance=0.3,
    n_init=0,
)
```

### Sampling

```python
# predictions MUST be the holdout benchmark from training
result = sampler.sample(
    predictions="svamp",                # ← holdout benchmark
    target_model="gemini25_flash",      # ← target model from training
    budget=50,
    seed=42,
)
```

### End-to-End: Train → Sample

```python
from proeval.encoder import EncoderTrainer
from proeval.sampler import BQEncoderSampler

# Step 1: Train encoder (holdout = svamp)
trainer = EncoderTrainer(
    train_benchmarks=["gsm8k", "strategyqa"],
    holdout_benchmark="svamp",
    target_model="gemini25_flash",
    hidden_dim=16, kernel_type="matern", epochs=200,
)
encoder_path = trainer.train(data_dir="data")

# Step 2: Sample on the holdout benchmark
sampler = BQEncoderSampler(encoder_path=encoder_path)
result = sampler.sample(
    predictions="svamp",
    target_model="gemini25_flash",
    budget=50,
)
```

Returns the same `SamplingResult` object as `BQPriorSampler`.

---

## 2. TopicAwareGenerator — Test Case Generation

Generate hard, diverse test cases guided by BQ posterior and topic structure. The generator manages its own GP state and automatically selects hard anchors via SS acquisition on each `generate()` call.

### Initialisation

The generator supports two prior modes:

**BQ-SF: Learned prior** (from other models' predictions)

```python
from proeval.generator import TopicAwareGenerator
from proeval.sampler import load_predictions, extract_model_predictions
from proeval.sampler.data import setup_train_test_split

df = load_predictions("gsm8k")
pred_matrix, model_names = extract_model_predictions(df)
_, _, _, prior_u, prior_S = setup_train_test_split(
    pred_matrix, target_model="gemini25_flash", model_names=model_names
)

gen = TopicAwareGenerator(
    df=df,
    dataset="gsm8k",
    n_topics=11,                          # BERTopic runs internally
    prior_u=prior_u,                      # Prior mean from pretrain models
    prior_S=prior_S,                      # Prior covariance
    noise_variance=0.3,
    model="google/gemini-3-flash-preview",
)
```

**BQ-TPF: Encoder prior** (from trained neural encoder)

```python
gen = TopicAwareGenerator(
    df=df,
    dataset="gsm8k",
    n_topics=11,
    encoder_path="path/to/encoder.pth",
    embeddings_path="path/to/embeddings.npy",
    model="google/gemini-3-flash-preview",
)
```

| Parameter         | Type            | Default                           | Description                                                        |
| ----------------- | --------------- | --------------------------------- | ------------------------------------------------------------------ |
| `df`              | `DataFrame`     | _required_                        | Dataset DataFrame with `question`, `ground_truth` columns          |
| `dataset`         | `str`           | `"gsm8k"`                        | Dataset name (determines prompt format)                            |
| `n_topics`        | `int`           | `11`                              | Number of topics for BERTopic (runs internally)                    |
| `prior_u`         | `np.ndarray`    | `None`                            | Prior mean from pretrain models (BQ-SF)                            |
| `prior_S`         | `np.ndarray`    | `None`                            | Prior covariance (BQ-SF)                                           |
| `noise_variance`  | `float`         | `0.3`                             | GP noise variance (BQ-SF)                                         |
| `encoder_path`    | `str`           | `None`                            | Path to encoder `.pth` (BQ-TPF)                                    |
| `embeddings_path` | `str`           | `None`                            | Path to embeddings `.npy` (BQ-TPF)                                 |
| `model`           | `str`           | `"google/gemini-3-flash-preview"` | LLM for test case generation                                      |
| `ss_threshold`    | `float`         | `0.0`                             | SS acquisition threshold λ                                        |
| `ss_beta`         | `float`         | `1.96`                            | UCB exploration parameter β                                       |

### Generate → Evaluate → Update Loop

The generator has a built-in GP posterior that updates after each evaluation:

```python
for i in range(num_cases):
    # 1. Generate — auto-selects hard anchors via SS acquisition
    case = gen.generate(strategy="tss", k_examples=5)
    print(f"Topic: {case['topic']}, Question: {case['question']}")

    # 2. Evaluate the generated question (your evaluation logic)
    score = evaluate(case["question"])  # 1.0 = failure, 0.0 = correct

    # 3. Update GP posterior with the result
    gen.update(score)
```

### Full Example: Sampler → Generator Pipeline

```python
import numpy as np
from proeval.sampler import BQPriorSampler, load_predictions, extract_model_predictions
from proeval.sampler.data import setup_train_test_split
from proeval.generator import TopicAwareGenerator

# Step 1: Load data and compute prior
df = load_predictions("gsm8k")
pred_matrix, model_names = extract_model_predictions(df)
_, _, _, prior_u, prior_S = setup_train_test_split(
    pred_matrix, target_model="gemini25_flash", model_names=model_names
)

# Step 2: Create generator (BERTopic + GP prior initialized internally)
gen = TopicAwareGenerator(
    df=df,
    dataset="gsm8k",
    n_topics=11,
    prior_u=prior_u,
    prior_S=prior_S,
    noise_variance=0.3,
    model="google/gemini-3-flash-preview",
)

# Step 3: Generate → Evaluate → Update loop
for i in range(5):
    case = gen.generate(strategy="tss", k_examples=5)
    print(f"\n--- Case {i+1} (Topic: {case['topic']}) ---")
    print(f"Question: {case['question']}")
    print(f"Answer:   {case['ground_truth']}")

    # Evaluate and update GP posterior
    score = evaluate(case["question"])  # your eval function: 1.0=failure, 0.0=correct
    gen.update(score)
```

---

## 3. Dataset — Bring Your Own Data

`Dataset` is the single object that flows through the whole pipeline —
**prediction → sampling → generation**. It bundles **questions + ground truths
+ a `DatasetConfig`**, and (when built from a predictions CSV) also exposes the
prediction matrix and embeddings the sampler/generator need. Use it whenever you
want to evaluate or sample on data that isn't already wired into
`DATASET_CONFIGS`.

### Constructors

```python
from proeval import Dataset, DATASET_CONFIGS, LLMPredictor

# (a) Built-in: load one of the 10 built-in datasets from HuggingFace
ds = Dataset.from_builtin("svamp")

# (b) From a pre-computed predictions CSV — offline, and the bridge to sampling.
#     Carries questions/ground_truths AND the label_<model> matrix + embeddings.
ds = Dataset.from_predictions("svamp")

# (c) From in-memory lists (simplest custom case)
ds = Dataset.from_lists(
    name="my_yesno",
    questions=["Is the sky blue?", "Is fire cold?"],
    ground_truths=["yes", "no"],
    prompt_template=lambda q: f"{q} Respond JSON: {{'answer': 'yes'|'no'}}",
    extract_prediction=lambda d: d["answer"],
    extract_ground_truth=lambda gt: str(gt).lower(),
    compare_predictions=lambda p, g: 0.0 if str(p).lower() == g else 1.0,
)

# (d) From a CSV file
ds = Dataset.from_csv(
    "my_data.csv",
    question_col="question",
    ground_truth_col="answer",
    config=DATASET_CONFIGS["strategyqa"],   # reuse an existing config
)
```

If you already have a built-in `DatasetConfig` that fits your scoring needs,
pass it via `config=...` and skip the four eval-function arguments. A `config`
is only required for `predict()`; a `Dataset` built purely for sampling can omit
it.

### Use a Dataset everywhere

A `Dataset` can be passed directly to the sampler and generator in place of a
dataset-name string — one object carries the data through every stage:

```python
from proeval import BQPriorSampler, TopicAwareGenerator

ds = Dataset.from_predictions("svamp")

# Sampling: equivalent to sample(predictions="svamp", ...), but the Dataset
# also supplies its name (for GMM selection) and cached predictions.
result = BQPriorSampler(noise_variance=0.3).sample(
    predictions=ds, target_model="gemini25_flash", budget=50,
)

# The accessors the sampler relies on are also available directly:
matrix, model_names = ds.prediction_matrix()   # (n_samples, n_models), 1=failure
embeddings = ds.embeddings()                    # (n_samples, d)

# Generation: pass the Dataset as `df`; its name drives the prompt format.
gen = TopicAwareGenerator(df=ds, prior_u=prior_u, prior_S=prior_S)
```

Passing a name string or a raw DataFrame still works exactly as before.

### Predict

```python
predictor = LLMPredictor(model="google/gemma-3-4b-it")

# Either direction works — they're equivalent
results = ds.predict(predictor, parallel=True, workers=10)
results = predictor.predict_dataset(ds, parallel=True, workers=10)
# results: list of (question, ground_truth, raw_response, prediction, score)
```

`Dataset` also supports `len(ds)`, `ds[i]`, and iteration — returning
`(question, ground_truth)` tuples.

---

## 4. LLMPredictor — LLM Evaluation

Evaluate LLMs on supported datasets with structured JSON parsing, retry logic, and parallel batching.

### Initialisation

```python
from proeval.evaluator import LLMPredictor, DATASET_CONFIGS

predictor = LLMPredictor(
    model="google/gemma-3-27b-it",   # Model identifier
    api_key=None,                     # OpenRouter API key (default: env var)
)
```

### Single Evaluation

```python
config = DATASET_CONFIGS["strategyqa"]
raw_response, prediction, score = predictor.evaluate(
    question="Is the sky blue?",
    ground_truth=True,
    dataset_config=config,
    max_parse_retries=3,
)
# score: 0.0 = correct, 1.0 = failure
```

### Batch Evaluation (Parallel)

```python
results = predictor.predict_batch_parallel(
    questions=["Is fire hot?", "Is ice warm?"],
    ground_truths=[True, False],
    dataset_config=DATASET_CONFIGS["strategyqa"],
    max_workers=10,       # Concurrent threads
    show_progress=True,   # tqdm progress bar
    skip_error=False,     # True: mark parse errors as NaN; False: mark as 1.0
)
# results: list of (question, ground_truth, raw_response, prediction, score)
```

### Supported Datasets

| Dataset    | Config Key     | Task Type               |
| ---------- | -------------- | ----------------------- |
| StrategyQA | `"strategyqa"` | Yes/No reasoning        |
| GSM8K      | `"gsm8k"`      | Math problem solving    |
| SVAMP      | `"svamp"`      | Math word problems      |
| MMLU       | `"mmlu"`       | Multiple choice         |
| MMLU (Law) | `"mmlu_professionallaw"` | Multiple choice |
| Jigsaw     | `"jigsaw"`     | Toxicity classification |
| ToxicChat  | `"toxicchat"`  | Toxicity classification |
| GQA        | `"gqa"`        | Visual QA               |
| DICES      | `"dices"`      | Response quality rating |
| DICES-T2I  | `"dices_t2i"`  | Image quality rating    |

### UnifiedCSVManager — Multi-Model CSV with Resume & Fix-Error

The `UnifiedCSVManager` manages multi-model prediction CSVs with **checkpoint-based resume** and **fix-error mode**.

#### Full Evaluation with Resume

```python
from proeval.evaluator import LLMPredictor, DATASET_CONFIGS, UnifiedCSVManager, load_dataset_data

# Load dataset
questions, ground_truths = load_dataset_data("gsm8k")

# Create CSV manager (loads existing CSV or creates new one)
csv_mgr = UnifiedCSVManager("gsm8k", output_dir="./data")
csv_mgr.load_or_create(questions, ground_truths)

# Run evaluation with checkpoint resume
predictor = LLMPredictor(model="google/gemma-3-27b-it")
csv_mgr.run_evaluation(
    predictor,
    model_name="gemma3_27b",
    dataset_config=DATASET_CONFIGS["gsm8k"],
    questions=questions,
    ground_truths=ground_truths,
    parallel=True,           # Use parallel API calls
    workers=10,              # Thread count
    skip_error=False,        # True: NaN for parse errors, False: 1.0
    checkpoint_interval=50,  # Save every 50 items (sequential mode)
)
# If interrupted, re-run the same code — it resumes from checkpoint automatically
```

| Parameter             | Type   | Default | Description                                                                               |
| --------------------- | ------ | ------- | ----------------------------------------------------------------------------------------- |
| `parallel`            | `bool` | `True`  | Use `predict_batch_parallel` for faster processing                                        |
| `workers`             | `int`  | `10`    | Thread count for parallel mode                                                            |
| `skip_error`          | `bool` | `False` | `True`: mark parse errors as NaN (excluded from accuracy). `False`: mark as 1.0 (failure) |
| `rerun`               | `bool` | `False` | Force re-evaluation even if model columns exist                                           |
| `checkpoint_interval` | `int`  | `50`    | Save checkpoint every N items (sequential mode only)                                      |

#### Fix-Error Mode

Re-run only predictions that failed (`SKIPPED`, `ERROR`, `RATE_LIMITED`, `PARSE_ERROR`):

```python
# Fix errors in existing CSV
csv_mgr = UnifiedCSVManager("gsm8k", output_dir="./data")
csv_mgr.load_or_create(questions, ground_truths)

# Check how many errors exist
errors = csv_mgr.get_error_indices("gemma3_27b")
print(f"Found {len(errors)} errors to fix")

# Re-run only the failed predictions
csv_mgr.fix_errors(
    predictor, "gemma3_27b", DATASET_CONFIGS["gsm8k"],
    questions, ground_truths, parallel=True,
)
```

#### CSV Structure

The unified CSV has one row per question, with columns per model:

```
index, question, ground_truth,
prediction_gemma3_27b, label_gemma3_27b, raw_response_gemma3_27b,
prediction_claude35, label_claude35, raw_response_claude35,
...
```

#### Useful Methods

```python
csv_mgr.has_model("gemma3_27b")           # Check if model columns exist
csv_mgr.get_model_accuracy("gemma3_27b")   # float: 1 - mean(labels)
csv_mgr.get_error_indices("gemma3_27b")    # List[int]: rows with errors
csv_mgr.save()                             # Write DataFrame to CSV
```

---

## 5. EncoderTrainer — Train a Neural Encoder

Train a neural encoder for cross-benchmark BQ prior (Setting 1).

### Python API

```python
from proeval.encoder import EncoderTrainer

trainer = EncoderTrainer(
    train_benchmarks=["gsm8k", "strategyqa"],
    holdout_benchmark="svamp",
    target_model="gemini25_flash",
    hidden_dim=16,
    kernel_type="matern",
    init_lengthscale=1.0,
    epochs=200,
)
encoder_path = trainer.train(data_dir="data", output_dir="results")
```

| Parameter            | Type        | Default                  | Description                                    |
| -------------------- | ----------- | ------------------------ | ---------------------------------------------- |
| `train_benchmarks`   | `list[str]` | _required_               | Benchmarks for training                        |
| `holdout_benchmark`  | `str`       | _required_               | Benchmark to hold out for testing              |
| `target_model`      | `str`       | `"gemini25_flash"`       | Model to evaluate for BQ testing               |
| `hidden_dim`         | `int`       | `16`                     | Encoder hidden dimension                       |
| `learning_rate`      | `float`     | `0.01`                   | Adam learning rate                             |
| `epochs`             | `int`       | `200`                    | Training epochs                                |
| `kernel_type`        | `str`       | `"matern"`               | `"linear"`, `"matern"`, or `"rbf"`             |
| `init_lengthscale`   | `float`     | `1.0`                    | Initial lengthscale for matern/rbf             |
| `matern_nu`          | `float`     | `2.5`                    | Matérn smoothness (0.5, 1.5, 2.5)              |
| `init_var`           | `float`     | `0.3`                    | Initial noise variance                         |
| `embedding_model`    | `str`       | `"text-embedding-3-large"` | Embedding model for file lookup              |

### CLI

```bash
python -m experiment.exp_train_encoder \
    --train-benchmarks gsm8k strategyqa \
    --holdout-benchmark svamp \
    --target-model gemini25_flash \
    --hidden-dim 16 --kernel-type matern --init-lengthscale 1.0 --epochs 200
```

Use `--checkpoint-path path/to/encoder.pth` to specify the exact save location.

---

## 6. Utility Functions

### Data Loading

```python
from proeval.sampler import load_predictions, load_embeddings, extract_model_predictions

df = load_predictions("svamp", data_dir="path/to/data")
embeddings = load_embeddings("svamp", embedding_model="text_embedding_3_large")
pred_matrix, model_names = extract_model_predictions(df)
```

### Neural Encoder

```python
from proeval.encoder import QuestionEncoder, load_encoder, compute_kernel_matrix, get_phi_embeddings
import torch

# Load pre-trained encoder
encoder, checkpoint = load_encoder("path/to/encoder.pth", device=torch.device("cpu"))

# Compute phi embeddings
phi = get_phi_embeddings(encoder, embeddings)
```

### Diversity Metrics

```python
from proeval.utils import topic_entropy, embedding_coverage, failure_rate

te = topic_entropy(["math", "geometry", "algebra", "math"])  # 0-100%
ec = embedding_coverage(embeddings_array)                     # 0-1
fr = failure_rate([1.0, 0.0, 1.0, 0.0, 0.0])                # 60.0%
```

### Model Name Resolution

```python
from proeval.evaluator import resolve_model_name

resolve_model_name("gemma3_27b")  # → "google/gemma-3-27b-it"
resolve_model_name("claude35_sonnet")  # → "anthropic/claude-3.5-sonnet"
```

---

## 7. Experiment CLI Scripts

All experiment scripts live in `experiment/` and are run as Python modules:

| Script                               | Description                            |
| ------------------------------------ | -------------------------------------- |
| `exp_performance_estimation`          | Unified performance estimation (all methods) |
| `exp_failure_discovery`               | Active failure discovery experiments  |
| `exp_train_encoder`                  | Train a neural encoder for TPF         |

Example:

```bash
# Train encoder
python -m experiment.exp_train_encoder \
    --train-benchmarks gsm8k strategyqa --holdout-benchmark svamp --epochs 200

# Performance estimation with encoder
python -m experiment.exp_performance_estimation \
    --dataset svamp --encoder-path path/to/encoder.pth --n-runs 5
```

---

## Environment Variables

| Variable             | Description                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY` | Required for `LLMPredictor`, `TopicAwareGenerator`, and embedding metrics |

## Available Data Files

The `data/` directory contains pre-computed prediction CSVs and embeddings for
8 datasets:
`gsm8k`, `svamp`, `strategyqa`, `mmlu`, `jigsaw`, `gqa`, `dices`, `dices_t2i`.

(`mmlu_professionallaw` and `toxicchat` have `DATASET_CONFIGS` entries for
prediction via `LLMPredictor`, but no pre-computed files ship in `data/`.)
