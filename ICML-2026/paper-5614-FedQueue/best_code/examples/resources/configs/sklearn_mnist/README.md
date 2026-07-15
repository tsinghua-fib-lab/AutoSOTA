# Federated Learning with a scikit-learn Trainer (MNIST)

## High-level idea

APPFL's server infrastructure expects a PyTorch model (i.e. `nn.Module`) — it calls
`state_dict()`, `load_state_dict()`, and aggregates `OrderedDict[str, Tensor]`.
On the other hand, `sklearn` stores learned weights as plain `numpy` arrays (`coef_`, `intercept_`) with
no such interface.  Making the two work together requires two additional wrappers to make them compatible,
with **no changes to any existing server or aggregator code (minimizing modifications)**:

- **`SklearnModelWrapper`** — a thin `nn.Module` subclass that wraps a fitted
  `sklearn` estimator and overrides `state_dict()` and `load_state_dict()` to
  convert between numpy arrays and tensors. This lets the `FedAvgAggregator`
  initialise the global model and broadcast it to clients exactly as it would
  for any PyTorch model. **For any `sklearn` model you want to use, you only need to wrap it in a `SklearnModelWrapper`  when you define the model factory function.**

- **`SklearnTrainer`** — a new prototype and simple client-side trainer with three methods that mirror
  the standard `BaseTrainer` interface:
  - `train()` — calls `fit` or `partial_fit` on the `sklearn` estimator directly
    using the data from the dataset.
  - `get_parameters()` — extracts the fitted numpy arrays (`coef_`,
    `intercept_`, …) and returns them as an `OrderedDict[str, Tensor]` that the
    server aggregator can average directly.
  - `load_parameters()` — receives the aggregated tensors from the server,
    converts them back to numpy, and writes them into the `sklearn` model's
    attributes so the next training round starts from the global model.

  The `get_parameters()` and `load_parameters()` methods are the key to making the `sklearn` model compatible with the server's expectations, while `train()` allows us to use `sklearn`'s training methods directly on the client side.

---

## Design overview

### 1. `SklearnTrainer` — client-side trainer

`SklearnTrainer` follows the same `BaseTrainer` interface as `VanillaTrainer`.
The three key methods convert between sklearn's numpy arrays and the
`OrderedDict[str, torch.Tensor]` format that FedAvg expects:

| Method | What it does |
|--------|-------------|
| `train()` | Calls `fit` or `partial_fit` on the sklearn estimator |
| `get_parameters()` | Walks fitted `*_` attributes, converts each numpy array to a torch tensor to send to the server |
| `load_parameters()` | Converts torch tensors received from the FL server back to numpy, writes into the sklearn model's attributes |

Currently, this simple and prototyping trainer supports the following training modes (`train_configs.mode`):
- **`partial_fit`** (default) — incremental update each round; natural for FL
- **`fit`** — full refit from scratch each round

### 2. `SklearnModelWrapper` — `nn.Module` shim

Some server-side code (e.g., loading the global model so that the server can
have the initial model to send to clients) still expects a PyTorch `nn.Module`,
so it can call `state_dict()`, `load_state_dict()`, and `named_parameters()`.
`SklearnModelWrapper` overrides the first two to delegate to the shared
`_sklearn_get_state` / `_sklearn_set_state` helpers, letting the code treat
sklearn params exactly like PyTorch params.

Because the wrapper has no `nn.Parameter` objects, `named_parameters()` yields
nothing.  The aggregator then falls through to its direct-averaging path — which is
exactly what we want for sklearn weights.

`SklearnTrainer.__init__` automatically **unwraps** the model so the trainer
always operates on the raw sklearn estimator:

```python
if isinstance(self.model, SklearnModelWrapper):
    self.model = self.model.sklearn_model
```

### 3. `sgd_classifier.py` — model factory

In this example, we provide two functions selectable via `model_name` in the server config:

| Function | When to use |
|----------|------------|
| `get_sgd_classifier(n_features, n_classes, ...)` | Fresh zero-weight start |
| `load_sgd_classifier(checkpoint_path)` | Pre-trained checkpoint start |

Both return a `SklearnModelWrapper` ready for the server aggregator.

**Why a dummy fit?** sklearn attributes like `coef_` and `intercept_` do not
exist until after the first `fit` / `partial_fit` call.  `get_sgd_classifier`
runs one pass on zero dummy data so the arrays are created with the correct
shapes, then zeros them out.  This gives the server a valid `state_dict()` to
send, and gives each client pre-existing attributes that `load_parameters()` can
overwrite with the server's values before round 0 begins.

**Why a joblib checkpoint?**  `load_sgd_classifier` skips the dummy fit entirely
and loads a fully trained model instead.  The server broadcasts those weights to
all clients, which then fine-tune via federated learning rather than starting
from zero.

### 4. Flat MNIST dataset (`mnist_dataset_flat.py`)

sklearn expects 2-D input `(N, features)`.  This loader flattens each
`(1, 28, 28)` MNIST image to a `(784,)` numpy array and returns a `NumpyDataset`
— a minimal wrapper holding `(X, y)` numpy arrays with `__len__` and
`__getitem__`.  No torch tensors are involved on the data path.

`NumpyDataset` satisfies the two things `ClientAgent` needs from a dataset:
- `len(dataset)` — used by `get_sample_size()`
- `dataset[i]` — used by `SklearnTrainer._to_numpy()` to extract arrays

Because `SklearnTrainer` no longer uses a `DataLoader`, there is no torch
dependency on the client data path at all.  A custom dataset loader for sklearn
only needs to return any object that supports `len()` and `__getitem__`.

---

## How to run

All commands are run from the `examples/` directory.

### Option 1 — zero-initialised global model

All clients start from the same zero-weight `SGDClassifier` distributed by the
server.  No preparation is needed.

```bash
# Terminal 1 — server
python grpc/run_server.py --config resources/configs/sklearn_mnist/server_fedavg.yaml

# Terminal 2 — client 1
python grpc/run_client.py --config resources/configs/sklearn_mnist/client_1.yaml

# Terminal 3 — client 2
python grpc/run_client.py --config resources/configs/sklearn_mnist/client_2.yaml
```

### Option 2 — pre-trained checkpoint as global model

The server loads a checkpoint trained on the full MNIST dataset and distributes
its weights to all clients as the starting point.  Clients then fine-tune via
federated learning.

**Step 1 — generate the checkpoint** (only needed once):

```bash
python resources/configs/sklearn_mnist/pretrain.py \
    --output checkpoints/sgd_mnist.joblib \
    --epochs 20
```

The script prints train/test accuracy each epoch.  After 20 epochs you should
see roughly 90 % test accuracy.  The checkpoint is saved to
`examples/checkpoints/sgd_mnist.joblib`.

> The checkpoint file is not committed to the repository.  Every user must
> generate it locally before running Option 2.

**Step 2 — start the federation**:

```bash
# Terminal 1 — server (loads checkpoint)
python grpc/run_server.py --config resources/configs/sklearn_mnist/server_fedavg_checkpoint.yaml

# Terminal 2 — client 1
python grpc/run_client.py --config resources/configs/sklearn_mnist/client_1.yaml

# Terminal 3 — client 2
python grpc/run_client.py --config resources/configs/sklearn_mnist/client_2.yaml
```

> Client configs are identical for both options.  The only difference is which
> server config is used — `server_fedavg.yaml` vs `server_fedavg_checkpoint.yaml`.

---

## Configuration reference

### `server_fedavg.yaml` vs `server_fedavg_checkpoint.yaml`

The two server configs are identical except for `model_configs`:

```yaml
# server_fedavg.yaml — zero-init
model_configs:
  model_path: "./resources/model/sgd_classifier.py"
  model_name: "get_sgd_classifier"
  model_kwargs:
    n_features: 784
    n_classes: 10
    loss: "log_loss"
    alpha: 0.0001
    random_state: 42

# server_fedavg_checkpoint.yaml — pre-trained start
model_configs:
  model_path: "./resources/model/sgd_classifier.py"
  model_name: "load_sgd_classifier"
  model_kwargs:
    checkpoint_path: "./checkpoints/sgd_mnist.joblib"
```

### `client_N.yaml`

Client configs contain no `model_configs` — the server broadcasts the model
source automatically via `get_configuration()`.

```yaml
data_configs:
  dataset_path: "./resources/dataset/mnist_dataset_flat.py"
  dataset_name: "get_mnist_flat"
  dataset_kwargs:
    num_clients: 2
    client_id: 0          # 0 for client_1, 1 for client_2
    partition_strategy: "iid"
```

---

## Files added / changed

| File | Role |
|------|------|
| `src/appfl/algorithm/trainer/sklearn_trainer.py` | `SklearnTrainer` + `SklearnModelWrapper` |
| `examples/resources/model/sgd_classifier.py` | Model factory (`get_sgd_classifier`, `load_sgd_classifier`) |
| `examples/resources/dataset/mnist_dataset_flat.py` | Flat (784-dim) MNIST dataset |
| `examples/resources/configs/sklearn_mnist/pretrain.py` | Script to generate a joblib checkpoint |
| `examples/resources/configs/sklearn_mnist/server_fedavg.yaml` | Server config — zero-init start |
| `examples/resources/configs/sklearn_mnist/server_fedavg_checkpoint.yaml` | Server config — checkpoint start |
| `examples/resources/configs/sklearn_mnist/client_1.yaml` | Client 1 config |
| `examples/resources/configs/sklearn_mnist/client_2.yaml` | Client 2 config |
