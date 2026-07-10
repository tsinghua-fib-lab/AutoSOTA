# Code Analysis — Paper 2727: Hyperbolic Neural Population Geometry

## Evaluation Path
- Entry: /repo/eval_mnist.py — parses CLI args, runs N trials, averages
- Training loop: /repo/ml/hyphop/test_mnist.py — standard MNIST train/test loop
- Model wrapper: /repo/ml/hyphop/models/wrappers.py -> SingleInstanceClassifier
- Core module: /repo/ml/hyphop/models/KFCore.py -> KFCore (KarcherFlow hyperbolic attention)
- Dataset: MNIST auto-downloaded by torchvision to /repo/ml/hyphop/datasets/
- Working directory: /repo/ml/hyphop (set by eval_mnist.py via os.chdir)

## Train/Inference Pipeline
1. MNIST images (28x28=784) -> embedder (Linear(784,4) + ReLU) -> 4D representation
2. 4D embeddings mapped to hyperboloid via expmap0 (4D + 1 time coordinate = 5D)
3. KFAttention: Q,K projected to d=4 -> expmap0 -> hyperbolic inner product -> softmax(-beta*sims)
4. Karcher flow: weighted Frechet mean in hyperbolic space -> logmap0 -> 4D Euclidean
5. Classifier: Linear(4,10) -> cross-entropy loss

## Config Path
- CLI args in eval_mnist.py: --model, --hidden-dim, --epochs, --lr, --gamma, --batch-size, --beta, --trials, --seed
- CLI args in test_mnist.py: additional --num-states, --num-memories
- Fixed: optimizer=AdamW, weight_decay=1e-4, StepLR scheduler

## Metric Parser
- eval_mnist.py prints: RESULT: accuracy=X.XXXX accuracy_std=Y.YYYY
- Accuracy is mean test accuracy across trials (0-100 scale)

## Safe Modification Targets
- KFCore.py: __init__, forward — add learnable params, change attention mechanism
- wrappers.py: SingleInstanceClassifier.__init__ — pass new params to KFCore
- test_mnist.py: training loop — add gradient clipping, EMA, MixUp, LR schedule
- eval_mnist.py: add new CLI args (must NOT change metric computation)

## Risky Files (do NOT modify)
- geoopt/ (vendored hyperbolic operations)
- datasets/loader.py (data loading)
- hflayers/ (Hopfield baseline — not the optimization target)

## Reusable Resources
- None pre-downloaded. MNIST auto-downloads.

## Key Hyperparameters
- hidden_dim=4 (paper core claim)
- lr=0.001, gamma=0.96 (StepLR), epochs=14, batch_size=64
- beta=1/sqrt(d)=0.5
- weight_decay=1e-4 (hardcoded)
- seed=42
