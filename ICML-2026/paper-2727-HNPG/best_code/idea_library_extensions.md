
### ALGO-8: Multi-Seed Ensemble with Temperature Scaling
- **ID**: ALGO-8
- **Type**: ALGO
- **Priority**: P1
- **Hypothesis**: Trial variance is high (±2.82%). Instead of single-model EMA, train 3 models with different seeds and ensemble their logits. This directly addresses the seed sensitivity observed in ALGO-1 trial 4 (80.27% vs ~86% for other trials).
- **Target**: eval_mnist.py — add --ensemble flag that trains multiple seeds and averages logits at test time.
- **Expected**: +0.5-1.0% mean accuracy, significantly reduced std.
- **Red-line check**: Pass. Only changes training initialization, not eval protocol.
- **Source**: Standard ensemble technique; TraceCard-S-ALGO-4.

### CODE-5: Orthogonal Initialization for W_Q and W_K
- **ID**: CODE-5
- **Type**: CODE
- **Priority**: P1
- **Hypothesis**: The Q/K projections at d=4 produce highly correlated queries and keys, reducing attention discriminability. Orthogonal initialization ensures Q and K projections explore different subspaces, improving attention quality.
- **Target**: KFCore.__init__() — apply torch.nn.init.orthogonal_ to W_Q.weight and W_K.weight.
- **Expected**: +0.3-0.7% from better attention diversity.
- **Red-line check**: Pass. Only changes weight initialization.
- **Source**: Standard attention best practice; validated in SparseMixer.

### CODE-6: Two-Stage Training with Frozen Embedder
- **ID**: CODE-6
- **Type**: CODE
- **Priority**: P2
- **Hypothesis**: At d=4, the embedder (Linear(784,4) + ReLU) may dominate training, leaving KFAttention undertrained. Two-stage training: first freeze KFAttention and train embedder+classifier for 5 epochs, then unfreeze all for 9 epochs.
- **Target**: test_mnist.py training loop.
- **Expected**: +0.3-0.6% if KFAttention was undertrained.
- **Red-line check**: Pass. Same total epochs, same eval.
- **Source**: Curriculum learning / progressive training.
