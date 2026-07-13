# Code Analysis for Paper 5411: DisCoVR

## Evaluation Path
- **eval.py** at /repo/eval.py: Standalone ParametricModel evaluation.
  - Synthetic data: z,w ~ N(0,1), x = z+w, y = I(w>0)
  - DLVAE(DisCoVR) + AdversarialThresholdPyroTrainer
  - N_REPS=3 repetitions (seeds 0,1,2)
  - Output: FINAL: NLL=X.XXXX+/-Y.YYYY, FINAL: Delta-Bayes=X.XXXX+/-Y.YYYY

## Training Path
1. make_data(seed): synthetic ParametricModel
2. DLVAE(1, [1], latent_dim=1, w_dim=1, hidden_dim=8, num_layers=2, ...)
   - _DoubleLossMixin + _AdversarialEntropyMixin
   - encoder: GaussianMLP(1->[8,8]->1) for z
   - encoder_w: GaussianMLP(2->[8,8]->1) for w (takes x,y concat)
   - decoder: MLP(2->[8,8]->1) for joint recon
   - decoder_z: MLP(1->[8,8]->1) for z-only recon
   - classifier: MLP(1->[8]->2) for adversarial entropy
3. AdversarialThresholdPyroTrainer(CONV_THRESH=5e-3, PATIENCE=30)
   - Trace_ELBO (single sample)
   - Alternates classifier/model steps per batch
   - AdamW(lr=1e-3), best model by test loss

## Hyperparameters (eval.py top-level)
N_SAMPLES=30000, BATCH_SIZE=256, LATENT_DIM=1, W_DIM=1
HIDDEN_DIM=8, NUM_LAYERS=2, LR=1e-3
REC_W=0.75, REC_Z=0.25, Z_KL_W=0.9, W_KL_W=0.2, ADV_W=0.8
PATIENCE=30, CONV_THRESH=5e-3, N_REPS=3

## Metric Parsing
stdout: FINAL: NLL=X.XXXX+/-Y.YYYY
stdout: FINAL: Delta-Bayes=X.XXXX+/-Y.YYYY

## Safe Mod Targets
- src/VAE_trainers.py: ELBO, optimizer config, training loop, grad clipping
- src/VAE_mixins.py: Model/guide loss weights, schedules
- src/VAE_variants.py DLVAE: architecture, init, layers
- eval.py: hyperparameter constants only
- src/MLP_variants.py: MLP layers

## Risky (DO NOT MODIFY)
- /tools/record_score.sh
- eval.py metric computation (post-training)
- make_data() function
- Test set creation, label computation
- Delta-Bayes/GaussianNB metric

## Red-line
1. No metric/test/data/scoring changes
2. No hard-coded outputs
3. Training-side only (loss, optimizer, architecture, init)
4. All metrics from real eval output
5. Both NLL and Delta-Bayes must be reported
